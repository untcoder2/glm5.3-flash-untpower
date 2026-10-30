# Request-boundary-checkpoints gate: dynamic speculative decoding × Semantic LMCache

## Repro

Serve with the semantic LMCache offloading connector (request-boundary
checkpoints required) and enable dynamic/adaptive speculative decoding depth
(window-based). Startup dies with:

```
ValueError: Semantic LMCache requires a supported request-boundary adapter
```

## Root cause

In `vllm/config/vllm.py` the `use_request_boundary_checkpoints` property couples
"dynamic speculative decoding is in use" (`uses_dynamic_speculative_decoding()`)
to *forbidding* request-boundary checkpoints. But window-based adaptive depth
does not actually remove request boundaries — the real incompatibility is only
with per-batch spec schedules
(`speculative_config.num_speculative_tokens_per_batch_size`), which make
checkpoint placement ambiguous.

## Fix (2 hunks, config/vllm.py)

1. The property's condition keys on
   `num_speculative_tokens_per_batch_size is None` instead of
   `not uses_dynamic_speculative_decoding()` — window-based adaptive depth keeps
   request-boundary checkpoints (and therefore semantic LMCache compatibility).
2. The `_set_cudagraph_sizes` spec-tier block is guarded with
   `if schedule is not None:` — when no per-batch schedule is configured, the
   tier table is not touched.

Diff: [diffs/20261030-dynamic-sd-boundary-gate.diff](../diffs/20261030-dynamic-sd-boundary-gate.diff)

## Verification

- Mechanism verified live: with the fix the engine starts and runs with
  adaptive depth + semantic LMCache together (previously impossible).
- Honest caveat: in our A/B, adaptive depth itself measured **−15–20% on short
  prompts** (draft phase runs unfused, dispatch padding), so we kept adaptive
  depth off in production. The gate fix stands on its own as upstream-correct —
  the two features should not be mutually exclusive for window-based depth.
