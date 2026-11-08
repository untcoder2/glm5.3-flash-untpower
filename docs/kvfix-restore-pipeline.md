# Checkpoint-restore pipeline: drain outside the copy lock

Engine line: local-inference-lab/vLLM (GLM-5.3-Flash hybrid) + LMCache
`0.5.7+lil.rebase1`, transfer mode `engine_driven`, boundary checkpoints
(`--recurrent-checkpoint-policy request_boundaries`).

## Symptom

Warm turns of a ~130 K-token session sporadically paid **9–13 s TTFT while the
engine reported ~99% cache hit** and sat otherwise idle (no prefill bursts,
`num_requests_waiting = 0`, zero preemptions). Client telemetry (per-request
`cached_tokens`) proved the checkpoint lookup succeeded — the cost was in the
restore itself.

## Root cause

`CheckpointPageCopier.__call__`
(`lmcache/integration/vllm/checkpoint_copy.py`) held a `threading.Lock` across
the **entire** enqueue burst *and* `self._stream.synchronize()`:

```python
with (self._lock, torch_dev.device(...), torch_dev.stream(self._stream)):
    buffers = self._transfer.checkpoint_slot_views(...)   # borrow SHM views
    for ...:                                              # enqueue page copies
        page.copy_(self._pool[block], non_blocking=True)
    finally:
        self._stream.synchronize()                        # drain INSIDE the lock
```

The executor has 2 workers, but the lock serializes them into one effective
worker: lease *N+1* cannot even start enqueueing until lease *N*'s DMA fully
drains. Measured in the engine container:

| Path | Rate |
|---|---|
| Raw H2D memcpy (pinned or pageable, 12–128 MB) | **28.8 GB/s** |
| Effective restore rate (6.4 GB / 8.9–12.7 s) | **~725 MB/s (×38 slower)** |

A ~130 K-token checkpoint (~500 pages × 2048 tokens, fp8 KV) should cost
~0.25 s of DMA; the pipeline turned it into 8–13 s. This also explains why
raising the transfer-worker count on 06.11 (`max-gpu-workers 2→6`) changed
nothing — the bottleneck was the engine-side serialization, not capacity.

## Fix

Three changes, all inside `CheckpointPageCopier`:

1. **Event-based drain outside the lock.** After the enqueue burst, record a
   CUDA event and release the lock; `event.synchronize()` runs outside, so the
   next lease enqueues its own burst while the previous one drains on the
   (single, FIFO-ordered) stream. The call-level contract "returned ⇒ drained"
   is preserved.
2. **In-flight guard for the remap path.** `ShmPoolMapping.remap()` (the only
   remap call site is this file) may unmap the pinned SHM mapping; with drains
   now live outside the lock, a concurrent remap would unmap bytes under DMA
   (use-after-unmap → silent KV corruption). The remap path waits on a
   `Condition` for the in-flight counter to reach zero (bounded 30 s); past the
   deadline it raises `UnsafeCheckpointCopyError` ("worker termination
   required") **without** remapping. A double `is_current()` check prevents two
   workers from remapping twice.
3. **Legacy drain on the exception path.** If the enqueue loop raises mid-burst,
   the partial burst is drained under the lock (as before) before unwinding, so
   the caller cannot release the lease/SHM slots under live DMA. The in-flight
   increment happens only after a successful event record; the decrement after
   the drain in a `finally`.

## Verification pipeline

- Review round 1: 4 findings (2 per reviewer) — counter leak + un-drained
  partial burst on the exception path (CWE-667); remap under live DMA
  (CWE-362/416). All closed in round 2: both reviewers **PASS**.
- Independent verifier: compile, 52+/2− diff scope check, spot-checks of the
  review findings, deploy-script gates.
- Arbiter: GO. Deployed as a one-file overlay image on the battle engine
  (`FROM` battle + `COPY checkpoint_copy.py`); gates: byte-identical source
  tree outside the one file, md5, `py_compile`, patch marker.
- Acceptance: port, live md5, SJF module intact, generation, models, listener;
  5-minute recheck clean; watchdog healthy.

## Notes / caveats

- `vllm:kv_cache_usage_perc` drops to 0.0 at idle **by design** — cached
  (retained) blocks live in the free queue and do not count as usage. Do not
  treat it as evidence of cache loss.
- L2 on the cache side is deliberately absent in our deployment (post-incident
  clone after a memcg-OOM); enabling `--l2-adapter`/`checkpoint_on_evict`
  without re-validating host RAM is not recommended.
- Concurrency after the fix is bounded by the caller's executor
  (`CheckpointTransferWorker`, default `workers=2`), not by this class.
- **Live effect on the slow class is not yet measured** — the trigger requires
  real multi-session load; the patch is strictly non-regressive (same stream,
  FIFO order, only more overlap). Next candidates if the class persists:
  batching the per-page H2D copies, raising the worker count in code.
