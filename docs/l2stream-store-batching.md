# L2 store pipeline: byte-capped batches with inline backpressure

Engine line: local-inference-lab/vLLM (GLM-5.3-Flash hybrid) + LMCache
`0.5.7+lil.rebase1`, transfer mode `engine_driven`, L2 fs adapter
(`checkpoint_on_evict` store policy) on NVMe, L1 = 50 GiB SHM pool.

## Symptom

The cache container was killed by a cgroup OOM (`Exit 137`, `OOMKilled=true`)
during ordinary eviction churn: L1 sits at its watermark (0.80), evictions
trigger `checkpoint_on_evict` writes to L2, and the container's memcg
(52 GiB for a 50 GiB pool) blows up. Two production incidents (07.11 and
08.11), second one ~4 minutes into a 6×~200K-token cold-prefill burst.
The L2 disk kept receiving data right up to death (231 GB total) — the write
path itself worked; the container died of memory, not of I/O.

## Root cause: the whole burst is pinned at reservation time

`StoreController._submit_plan`
(`lmcache/v1/distributed/storage_controllers/store_controller.py`) took the
entire eviction burst from the store loop in one go:

1. `l1_mgr.reserve_read(keys)` — **read-pins every page of the burst in SHM
   at once** (there are no data copies anywhere: the fs adapter writes
   O_DIRECT from SHM views);
2. a single `submit_store_task` for the whole batch;
3. the pins live until one giant sequential NVMe write drains.

Meanwhile new stores allocate fresh SHM pages on top of the still-pinned
burst, so SHM (live L1 + pinned burst + new pages) grows past the memcg
limit. No existing lever bounds this: there is no cap on pending bytes,
`checkpoint_write_timeout_seconds` is an eviction timeout, and
`_SYNC_FLUSH_BATCH_SIZE` only covers the sync path (the fs adapter does not
use it).

## Fix: bounded batches, the wait IS the backpressure

`_submit_plan` now delegates to `_submit_bounded_batches`:

- reserve keys in slices of 8 (`_L2_STORE_RESERVE_SLICE`);
- accumulate into a batch capped at 2 GiB (`_L2_STORE_BATCH_BYTES`);
- submit, publish `L2_STORE_SUBMITTED` (per-batch totals), then
  `_wait_own_store_task` — a **blocking inline drain on the store-loop
  thread** (50 ms poll, 600 s deadline, 30 s warning interval). The next
  batch's `reserve_read` therefore cannot pin pages on top of the previous
  batch's still-pinned pages: the wait is the backpressure. Peak pin
  footprint ≈ batch cap + one reserve slice.

Failure paths (each found by the review pair — 4 rounds, 5 real bugs):

- `stop_flag` checked first in the poll loop, otherwise `stop()` hangs for
  N×600 s on a stuck adapter;
- `_advance_request` under try/except, otherwise an exception in
  `_finalize_store` silently drops the rest of the burst;
- a wait failure returns immediately and does **not** reserve the remaining
  keys (same semantics as the `KEY_NOT_EXIST`/`KEY_NOT_READABLE` skip path),
  so a dead adapter cannot keep collecting pins;
- `submitted` is the single source of truth for pinned keys — a repeated
  flush must not double-`finish_read` keys from an earlier batch;
- keys whose submit failed are **not** added to `submitted` (sync-finished
  keys stay re-storable) and are released exactly once via a `released` set.

L1 read failures (`not_found` / `write_locked`) are still published as
`L1_READ_FAILED` anomaly events, once at the end of the burst.

## Deployment note (second-order lesson)

The batching closes the burst-pin OOM, but a fully pre-fallocated L1 pool
leaves the memcg razor-thin: our orchestrator pre-fallocates the whole pool
(`posix_fallocate`) into the cache cgroup, so the pool charge is permanent
and the container limit must cover pool **plus** non-reclaimable overhead
(anon + kernel + pgtables + dirty page cache ≈ 2.1 GiB measured at OOM
time). Keep `--memory` at pool + ≥6 GiB — a `pool + 2 GiB` container OOM'd
after hours of churn once the whole pool was touched.

## Verification

- Review pair (correctness + security), 4 rounds, 5 findings — all closed;
  verifier pass over baked artifacts; arbiter GO.
- Baked into the battle image (`FROM` the kvfix overlay, md5-gated), window
  executed, acceptance green, LKG updated.
- Burst soak on the battle stack: 13 eviction loops, 47 GB written to L2,
  cache alive throughout, zero `Exit 137`, zero "pages lost"; RAM stable.
- Cold prefills during evictions: 33–48 s (dead-cache era: 93–116 s);
  warm re-probes of evicted sessions: 0.7 s vs 103 s.
