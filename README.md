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

## Measured findings

[docs/findings.md](docs/findings.md) — KV/L1 eviction thrash under working-set
pressure; live confirmation of
[issue #1020](https://github.com/local-inference-lab/vllm/issues/1020)
(multi-second decode stalls during deep prefills, inter-token gap p99 ≈ 80 s);
LMCache delta note (our wheel `0.5.3.dev453` vs beta `dde2c874b3` — checkpoint
copy admission #113 is relevant for our deployment); draft-acceptance telemetry
(55.5% / 1.66 tok/step at temp 1.0).
