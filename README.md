# GLM-5.3-Flash on local-inference-lab/vLLM — modules & measured notes

Production-stack notes and self-developed modules for serving **GLM-5.3-Flash**
(`glm5_next`: hybrid MLA + KDA, MTP draft k=3) on
[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)
(`jovian-cu134-beta`, 15.10.2026 release). Everything here is offered upstream;
all diffs are against `release-10152026` and Apache-2.0 (vLLM-derived).

**Stack**: 2× RTX PRO 6000 (SM120, PCIe, no NVLink), TP=2 + DCP=2, GLM-5.3-Flash
FP4 (modelopt_mixed), 45 layers hybrid MLA+KDA, MTP k=3 greedy (fused multi-step
draft), chunked prefill 4096, pp=1, auto-KV ~8.02 GiB/GPU, 1M context, standalone
LMCache server (50 GiB CPU L1, LRU, `engine_driven` transfer) with the semantic
offloading connector at request boundaries.

## Modules

1. **Split-CKV gather for mixed DCP batches on `glm5_next`** — extends
   [PR #1005](https://github.com/local-inference-lab/vllm/pull/1005), which ships
   the mechanism for `glm_moe_dsa` only. Under DCP>1 a mixed batch (some requests
   prefilling, some decoding) needs prefill rows to attend over their own gathered
   cache while decode rows keep the cheap query exchange; `glm5_next` routes through
   a different attention path (hybrid MLA+KDA), so the upstream gate never fires for it.
   **Status: live in production since 30.10**, acceptance battery green, 14/14 tests.
   → [docs/split-ckv-glm5next.md](docs/split-ckv-glm5next.md),
   [diff](diffs/20261029-split-ckv-glm5next.diff)

2. **SJF waiting-queue for prefill admissions (with aging)** — env-gated replacement
   for the FCFS waiting queue: shortest-job-first by prompt length, 30 s aging bypass
   so long prefills cannot starve. Targets head-of-line start latency under mixed
   load (agent swarms: an interactive turn queues behind a neighbour's long
   re-prefill; live inter-token row p50 0.75 s / p90 2.5 s / **p99 80 s**).
   **Status: written, reviewed (2 rounds), staged for a live window.**
   → [docs/sjf-prefill-queue.md](docs/sjf-prefill-queue.md),
   [diff](diffs/20261031-sjf-prefill-queue.diff)

3. **Request-boundary-checkpoints gate: dynamic speculative decoding × Semantic
   LMCache** — enabling adaptive spec depth with the semantic LMCache connector dies
   at startup with `ValueError: Semantic LMCache requires a supported
   request-boundary adapter`. Root cause: `config/vllm.py` couples "dynamic spec
   decoding" to forbidding request-boundary checkpoints, while the real
   incompatibility is only per-batch spec schedules. 2-hunk fix.
   **Status: mechanism verified live; adaptive depth itself measured −15–20% on
   short prompts (our workload) and is kept off — the gate fix stands on its own.**
   → [docs/dynamic-sd-boundary-gate.md](docs/dynamic-sd-boundary-gate.md),
   [diff](diffs/20261030-dynamic-sd-boundary-gate.diff)

## Tests

`tests/` run inside the container without pytest:

```
docker run --rm -v $PWD/tests:/tests:ro -w /tests \
  --entrypoint python3 <engine-image> /tests/run_without_pytest.py
```

14 split-CKV cases + 7 SJF cases, all green on `release-10152026`.

## Speed measurements (updated 31.10)

Three measurement forms, **not** cross-comparable — details, methods and rules
in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md):

- **F1** single-stream greedy decode (temp 0, cached prompt, median n=5);
- **F2** bench suite (temp 1.0, short/med/long, idle-gated);
- **F3** live inter-token row under mixed agent-swarm load.

### F2 — bench suite (tok/s)

| Config | short | med | long | accept |
|---|---|---|---|---|
| k=3 probabilistic, pp=1, chunk 4096 (28.10 base) | 150.22 | 159.09 | 180.17* | 0.44/0.51/0.68 |
| k=2 probabilistic, pp=1, chunk 4096 (28.10) | ~parity | +5.7% | +9.3%* | 0.64/0.64/0.76 |
| k=3 **greedy** draft, pp=1, chunk 4096 (30.10) | 153.40 | 159.80 | 225.49* | 0.41/0.43/0.68 |
| adaptive depth window 64 / initial 3 (30.10, canary A) | 123–130 (**−15–20%**) | −11% | — | — |
| split-CKV `MIXED=1`, adaptive off (30.10, canary B) | −3–4% (noise) | 162 | 232* | — |
| greedy + local-argmax + SJF + split-CKV, all on (31.10 battle, warm cache, 2 runs) | 140.6 / 143.1 | 164.9 / 162.2 | 213.7 / 190.3* | 0.34–0.36 / 0.42–0.47 / 0.65–0.66 |

\* long deltas unconfirmed — cross-run prefix-cache warmth contaminates long
rows; short/med are gated and decisive.

### F1 — single-stream greedy decode

| Config | Decode tok/s |
|---|---|
| k=3, probabilistic draft (28.10) | **259.2** median (256.1–261.2, n=5), TTFT 23 ms |
| k=2, probabilistic draft (28.10) | 217 (**−16%**) |
| k=3 greedy + local-argmax + SJF + split-CKV, all on (31.10 battle, warm cache) | 256.6 (255.4–256.9, n=5), TTFT 23 ms — parity |

### Prefill / refill (TTFT, stream, cold = unique header + seeded filler)

| Probe | TTFT | Prefill tok/s |
|---|---|---|
| 9K cold | 1.06 s (etalon 1.048 s — parity) | 8 500 |
| 60K cold | 5.9 s | 10 100 |
| 60K repeat, same prompt (cache refill) | **0.056 s** (cached 59 790/59 790) | **×106 vs cold** |
| 200K cold | 25.0 s | 8 000 |

### F3 — live mixed load

| Era | p50 | p90 | p99 |
|---|---|---|---|
| pp=2 era (29.10) | 0.75 s | 2.5 s | **80 s** |

Tail = small interactive turns behind deep prefills ([issue #1020](https://github.com/local-inference-lab/vllm/issues/1020));
single-start probes did not degrade (pp=1 vs pp=2: warm 0.05/0.033 s, cold 9K 1.06/1.24 s).

### Other configs

| Config | Result |
|---|---|
| Prefill chunk 8192 (vs 4096) | decode parity (218.1 vs 217.3), queue worse (TTFT 3.9 s vs 1.1 s), KV −17% — **closed** |
| reasoning_effort low / high / max | 0.6–1.1 s / 0.8–2.6 s / 2.6–10.3 s per fix task; quality 6/6, 6/6, 5/6 (max overthinks) |
| Draft acceptance (live, temp 1.0) | 55.5% accept, **1.66 tok/step** over k=3 positions |
| Adaptive depth verdict | FAIL by speed (kept off); correctness perfect (lossless A/B, quality 20/20) |

## Measured findings

[docs/findings.md](docs/findings.md) — KV/L1 eviction thrash under working-set
pressure; live confirmation of
[issue #1020](https://github.com/local-inference-lab/vllm/issues/1020)
(multi-second decode stalls during deep prefills, inter-token gap p99 ≈ 80 s);
LMCache delta note (our wheel `0.5.3.dev453` vs beta `dde2c874b3` — checkpoint
copy admission #113 is relevant for our deployment); draft-acceptance telemetry
(55.5% / 1.66 tok/step at temp 1.0).
