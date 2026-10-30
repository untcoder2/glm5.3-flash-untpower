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
- Status: **staged for a live window** (not yet in production).

## Known limitation

The sort key is the request's **total** prompt length, not the uncached
remainder — a long session with a warm prefix cache is de-prioritized even
though its actual re-prefill is small. Bounded by queue depth; the true-uncached
ordering needs admission-time radix lookups. Documented in the module docstring.
