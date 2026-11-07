# SJF waiting-queue for prefill admissions (with aging)

Env-gated replacement for the FCFS waiting queue (`vllm/v1/core/sched/request_queue.py`).

## Problem

The waiting queue is FCFS: an interactive short turn queues behind whatever long
re-prefill arrived earlier (head-of-line blocking). Under mixed load — our case is
agent swarms generating bursts of heavy prefills alongside small interactive turns —
this dominates start latency:

- live inter-token/TTFT row (mixed load, pp=1, chunked prefill 4096):
  **p50 0.75 s / p90 2.5 s / p99 80 s**;
- the p99 tail is almost entirely "small request stuck behind a deep prefill".

## Design

`SJFRequestQueue(deque[Request], RequestQueue)`:

- **Shortest-job-first** among waiting admissions, keyed by `num_tokens`
  (ties → arrival order).
- **Aging bypass**: any request waiting longer than
  `VLLM_SCHED_PREFILL_SJF_AGING_S` (default 30 s) is admitted FIFO before the
  SJF order — long prefills cannot starve.
- **Prepend-front semantics**: re-queued (preempted/aborted-and-restarted)
  requests go to the front in prepend order, ahead of the SJF selection. Each
  request lives in exactly one structure (front list OR deque) — no ghosting.
- **Tolerant `remove_requests`**: the scheduler passes the same abort list to
  both the running and the waiting queues, and a request lives in at most one of
  them — absent requests must not raise (the strict contract crashes the engine
  on *any* client abort).
- **`__bool__` / `__len__` / `__contains__` / `__iter__`** account for both
  structures (front-blind `__bool__` would hang preempted requests forever).

## Usage

```
VLLM_SCHED_PREFILL_SJF=1                      # factory override (FCFS→SJF)
VLLM_SCHED_PREFILL_SJF_AGING_S=30             # optional, default 30
```

The configured `--scheduling-policy` is ignored while the flag is set.

Diff: [diffs/20261031-sjf-prefill-queue.diff](../diffs/20261031-sjf-prefill-queue.diff)
Full module: [files/sjf/request_queue.py](../files/sjf/request_queue.py)

## Verification

- `tests/test_sjf_queue.py` — 7 cases (SJF order, aging bypass, prepend
  semantics, tolerant abort-mix removal, front-member `__bool__`/`__len__`,
  env-gate factory both ways), all green in the container.
- Two independent review rounds on our side (correctness + security) with the
  abort-crash PoC fixed and re-verified.
- Status: **live in production since 06.11** (env-gated, carried by the
  watchdog/LKG restore chain on restarts).

## v2 (08.11): admission key = uncached remainder

The total-length key above bit in production: a warm fat session (252 692-token
prompt, 58 tokens of actual work) queued as the "longest job" through the full
30 s aging — observed **TTFT 45.9 s at a 99.98% cache hit** (no prefill, no
eviction: pure queue wait; the p99 row above was the same effect at smaller
scale).

v2 keys the queue by the request's **uncached remainder**:

- the scheduler injects a read-only probe at construction —
  `coordinator.find_longest_cache_hit(request.block_hashes, max(0, num_tokens -
  max(1, coordinator.prefill_replay_tokens)))`, an admission-time radix lookup
  with no pool mutation (`get_computed_blocks` is deliberately NOT used: it
  mutates request state);
- probe results are memoized for 1 s per request;
- a single fallback point (`SJFRequestQueue._work`): a probe exception logs a
  rate-limited warning (60 s) with a fallback counter and falls back to the
  total prompt length — a broken radix cannot crash admission;
- aging bypass, prepend-front semantics and tolerant removal are unchanged.

Status: **live in production since 08.11**. Review: correctness PASS, security
PASS (probe purity verified against the coordinator source). Tests:
`tests/test_sjf_v2.py` — 8 cases (warm-fat-first regression, both fallback
paths, aged bypass, prepend, tolerant removal, iteration order, probe-aware
factory), all green in the container. Live check: 15 consecutive interactive
turns on the warm fat session — 0.87–1.06 s turn-to-turn, zero stalls.

Diff (v1 → v2): [diffs/20261108-sjf-v2-uncached-key.diff](../diffs/20261108-sjf-v2-uncached-key.diff)
