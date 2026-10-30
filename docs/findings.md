# Measured findings on the production stack

Observations from serving GLM-5.3-Flash continuously (mixed interactive +
agent-swarm load). Numbers from `/metrics` and container logs; method notes
included where the measurement form matters.

## 1. KV/L1 eviction thrash under working-set pressure

When the live working set exceeds the cache pool, the LMCache L1 server sits on
its watermark and evicts continuously:

- `L1 memory usage 0.80 above watermark 0.80; triggering eviction` —
  217 events in 3 h, 72 in the busiest hour, ~8 per 10 min while active;
- cumulative prefix-cache hit over a 10 h engine window: internal 62.5%
  (234.4M/375.1M tokens), external (LMCache) 84.5% (118.9M/140.7M);
- each eviction of a still-live session's prefix = full re-prefill on its
  return, which both burns prefill budget and further displaces other sessions.

Not a code bug — capacity vs working set — but worth noting because it
interacts with copy-admission pressure (see §3). Mitigation that works for us:
cutting idle/oversized sessions; expanding the pool is not viable on our VRAM
margin (see the 8K-chunk OOM history in the Penny-era notes).

## 2. Live confirmation of issue #1020 (chunk-lift race → decode stalls)

[Issue #1020](https://github.com/local-inference-lab/vllm/issues/1020)
(adaptive long-prefill chunk lift races with momentary decode absence) matches
what we see live:

- multi-second inter-token stalls during deep prefills under mixed load;
- live inter-token gap row: p50 0.75 s / p90 2.5 s / **p99 80 s**;
- measurement form that catches it: **inter-token gap distribution under mixed
  load**, not tok/s or TTFT alone (a "queue is fine" picture from p50 hides the
  tail completely).

Mitigations that helped us: LMCache offload (deferred re-prefills become cheap
cache hits), prefix hygiene, and keeping the working set under the pool (§1).
The SJF module (see README) targets the remaining head-of-line tail directly.

## 3. LMCache delta relevant to our deployment

Our wheel: `0.5.3.dev453+lil.cu134` (11.10, includes #112 "reload a preempted
request's prefix from LMCache" — directly relevant under our eviction pressure).
Branch head `dde2c874b3` (19.10) adds #113: **checkpoint copy admission
capacity 16 → 34** — rank transfer workers could reject scheduler-admitted
(up to 32) store/restore tasks under copy pressure, reported as *misses*.

The checkpoint subsystem is **active** in our deployment (1158 checkpoint log
mentions, live `checkpoint-*` prefetch requests), so #113 is a candidate
contributor to our external miss rate (15.5%). Upgrade pending — it needs the
new wheel in both the standalone server and the engine-side connector, and the
warm 50 GiB L1 is lost on cache-container recreation.

## 4. Draft-acceptance telemetry (MTP k=3, greedy sampling of drafts)

A 30-minute-tick collector (`tools/collect_spec.py`) tracks per-position draft
acceptance on the live stack:

- accept rate 55.5%, **1.66 tokens/step** at conversation temperature 1.0
  (sum over k=3 positions);
- greedy-era numbers from the SGLang stack (accept ≈ 5.07) are **not
  comparable** at temp 1.0 — acceptance is structurally lower there; per-position
  telemetry is the honest baseline for any draft-side experiment (k depth,
  NVFP4 draft head, probabilistic-vs-greedy draft sampling).

`tools/collect_spec.py` runs against the engine's `/metrics` + a periodic
quality prompt; `DASH_SPEC_BASE` points at the engine endpoint.
