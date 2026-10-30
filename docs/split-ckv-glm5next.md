# Split-CKV gather for mixed DCP batches on `glm5_next` (hybrid MLA+KDA)

Extends [PR #1005](https://github.com/local-inference-lab/vllm/pull/1005),
which adds "CKV gather for mixed DCP batches" — for `glm_moe_dsa` only.
GLM-5.3-Flash (`glm5_next`) shares the DCP problem but routes through a
different attention path (hybrid MLA + KDA), so the upstream gate never fires
for it.

## Problem

Under DCP>1 a batch can mix prefill rows and decode rows:

- prefill rows need to attend over their own KV — with the split-CKV trick they
  gather **only their own** cache instead of paying the full-CKV all-gather;
- decode rows must keep the cheap query exchange (query_gather + LSE-combine).

Without the routing, either correctness suffers for the prefill rows of the
mixed batch or every mixed batch pays the full gather.

## Our change

Same idea as #1005, extended to the `glm5_next` attention path:

- `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` — the sparse-GLM gate now
  recognizes `glm5_next`; `_use_b12x_split_ckv_gather` decides per batch;
  `forward_mqa` routes per-row (prefill → gathered own cache, decode → query
  exchange).
- `vllm/model_executor/layers/attention/mla_attention.py` — `split_ckv_dcp`
  branch + `_split_ckv_dcp_mqa` for the hybrid path.
- `vllm/v1/attention/ops/dcp.py`, `vllm/models/deepseek_v32/nvidia/mtp.py`,
  `vllm/models/glm5next/nvidia/mtp_draft_head.py` — taken from #1005 as-is.
- `vllm/envs.py` — `VLLM_B12X_MLA_CKV_GATHER_MIXED` (default off; the module is
  inert without it).

Diff: [diffs/20261029-split-ckv-glm5next.diff](../diffs/20261029-split-ckv-glm5next.diff)

## Verification

- Tests: `tests/test_split_ckv_glm5next.py` — 14 cases (builder families
  `glm_dsa`/`glm5next` × dcp_rank × split, plus the gate heuristic), all green
  in the container on `release-10152026`.
- Live: in production since 30.10 (DCP=2, TP=2). Standard acceptance battery
  green: bench ×2 (short/med/long parity), quality battery 20/20, TTFT probe,
  VRAM peak margin unchanged, logs clean (no OOM / illegal address).

## Notes

- The module is fully inert unless `VLLM_B12X_MLA_CKV_GATHER_MIXED=1`.
- Independent review pass (correctness + security) was done on our side before
  production; happy to walk through any hunk.
