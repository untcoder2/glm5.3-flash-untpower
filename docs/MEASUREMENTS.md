# MEASUREMENTS — GLM-5.3-Flash speed on local-inference-lab/vLLM

Measured notes on the production stack (see [README](../README.md) for the full
configuration). Same discipline as our Qwen3.8-Flash-Next notes: every number
carries its form, its config and its date — and forms are **not**
cross-comparable.

## Forms and comparison rules

| Form | What it measures | Tool |
|---|---|---|
| **F1: single-stream greedy decode** | best-case decode rate, short fully-cached prompt, temp 0, max_tokens 512, median of n=5 after warmup | `short_bench_vllm.py` |
| **F2: bench suite** | short/med/long mixed batch, temp 1.0, idle-gated (`running=0, waiting=0` sampled before and after) | `bench_suite_vllm.py` |
| **F3: live inter-token row** | what users actually feel under mixed agent-swarm load | telemetry, gap distribution |

Rules that cost us mistakes, recorded here so they cost nobody else:

- F1 ≠ F2 ≠ F3 — three different answers, never deltas across forms. Example:
  the F1 "etalon" 259 tok/s was measured at k=3; later k=2 measured +9.3% on F2
  long and **−16% on F1** (259→217). Both true simultaneously.
- SGLang-era baselines (408 tok/s @ the previous engine/model) are a different
  engine and a different model — indicative only, never deltas.
- Stream chunk rate ≠ tokens: chunks carry ~3.5 tokens (MTP); compute stream
  speed from the final chunk's `usage` (`stream_options.include_usage`).
- Measure only at idle: `running=0, waiting=0` gated before and after the series.

## F1 — single-stream greedy decode

| Date | Config | Decode tok/s | TTFT |
|---|---|---|---|
| 28.10 | k=3, probabilistic draft, pp=1 | **259.2** median (256.1–261.2, n=5) | 23 ms |
| 28.10 | k=2, probabilistic draft, pp=1 | **217** (−16% vs k=3) | — |

## F2 — bench suite (short / med / long, tok/s; accept = accepted tokens/step)

| Date | Config | short | med | long | accept |
|---|---|---|---|---|---|
| 28.10 | k=3, probabilistic, pp=1, 4096 (base) | 150.22 | 159.09 | 180.17 | 0.44/0.51/0.68 |
| 28.10 | k=2, probabilistic, pp=1, 4096 (wave A2) | ~parity | +5.7% (commit record; wave noted +9.3% on long) | +9.3% / +4.6% (two runs recorded) | 0.64/0.64/0.76 |
| 30.10 | k=3, **greedy** draft, pp=1, 4096 (run3, clean) | 153.40 | 159.80 | 225.49* | 0.41/0.43/0.68 |
| 30.10 | canary A: adaptive depth window 64 / initial 3 | 123–130 (**−15–20%**) | −11% | — | — |
| 30.10 | canary B: split-CKV `VLLM_B12X_MLA_CKV_GATHER_MIXED=1`, adaptive off | −3–4% (noise) | 162 | 232* | — |

\* long-row deltas are **unconfirmed**: cross-run prefix-cache warmth
(repeated prompts) contaminates long rows between runs; base was measured cold.
short/med are gated and treated as decisive.

Notes per row:

- **greedy draft (30.10)**: no harm vs probabilistic — short +2%, med +0.4%
  vs the 28.10 base; accept slightly lower on short/med (expected at temp 1.0)
  while tps holds parity. Kept (it is also the upstream default).
- **adaptive depth (canary A)**: correctness perfect (lossless greedy A/B
  bit-identical, quality 20/20), but the acceptance controller held depth 3
  almost always (mean_accepted ~1.7–2.2 never triggered downshift) while the
  fused multi-step draft was disabled and per-depth cudagraphs were captured —
  the tax outweighed the gain at our acceptance levels. Verdict: FAIL by speed.
- **split-CKV (canary B)**: parity on F2, lossless bit-identical, live in
  production. F2 does not exercise mixed prefill/decode contention — the real
  benefit (TTFT/TPOT when prefills land during decodes) is a live-telemetry
  question, not a suite question.

## F3 — live inter-token row (mixed agent-swarm load)

| Era | p50 | p90 | p99 |
|---|---|---|---|
| pp=2 era (29.10) | 0.75 s | 2.5 s | **80 s** |

The p99 tail is almost entirely small interactive turns queued behind deep
prefills (see [issue #1020 confirmation](../README.md) and the SJF module).
Single-start probes of the same era did **not** degrade (warm 0.05 vs 0.033 s,
cold 9K 1.06 vs 1.24 s for pp=1 vs pp=2) — the difference is queue behavior
under concurrency, not per-request cost.

## Prefill chunk 4096 vs 8192 (28.10, F2, pp=2 era)

| Chunk | Decode | Queue behavior | KV pool |
|---|---|---|---|
| 4096 (base) | 217.3 | capacity ≤5, waiting 0–1 | baseline |
| 8192 | 218.1 (parity) | waiting up to 7, small-turn TTFT up to 3.9 s (vs 1.1 s) | −17% |

Decode does not care about the chunk; admission granularity does. 8K closed
definitively for this revision (allocator warnings at startup, zero benefit).

## reasoning_effort levels (29.10, temp 0, 3 fix-tasks × 3 runs)

| Level | Quality | Reasoning tokens | Task time |
|---|---|---|---|
| low | 6/6 PASS | 0–9 | 0.6–1.1 s |
| high | 6/6 PASS | 17–30 | 0.8–2.6 s |
| max | 5/6 (1 of 3 runs broken: un-awaited coroutine) | 254–1229 | 2.6–10.3 s |

max is 5–8× slower and occasionally overthinks to a wrong answer; low == high
== max quality on these tasks. Note: hidden thinking phase is off at low — the
model reasons in visible text. Nondeterminism exists at temp 0 (probabilistic
MTP draft): a single run proves nothing, need ≥3.

## Draft-acceptance telemetry (live, temp 1.0)

Per-position draft acceptance collector (30-min tick): **55.5% accept,
1.66 tokens/step** summed over k=3 positions. Greedy-era SGLang acceptance
(~5.07) is not comparable at temp 1.0 — acceptance is structurally lower there.

## Prefill / refill battery (31.10, battle config, warm cache, idle-gated)

Method: each probe = unique header at position 0 (guarantees coldness against
prefix cache) + seeded filler, stream `/v1/completions`, TTFT = time to first
text chunk, usage from the final chunk (`stream_options.include_usage`).
The 60K probe runs twice on the **same** text: pass b = refill from cache
(LMCache + radix). Idle gate between probes (running==0 ∧ waiting==0, two
samples 10 s apart).

| Probe | TTFT | prompt tok | cached | Prefill tok/s |
|---|---|---|---|---|
| 9K cold | 1.057 s | 8 995 | 0 | 8 508 |
| 60K cold (a) | 5.926 s | 59 790 | 0 | 10 089 |
| 60K refill (b, same text) | **0.056 s** | 59 790 | 59 790 | — (×106 vs cold) |
| 200K cold | 25.04 s | 199 595 | 0 | 7 970 |

Scaling is mildly superlinear (200K/60K = 3.34× tokens → 4.2× time), consistent
with attention cost growth over long contexts. The 9K cold number matches the
30.10 etalon (1.048 s) to 1% — the method is comparable across days.

## Open questions (measured nothing yet, deliberately)

- k=3 vs k=2 on **both** F1 and F2 in one quiet window — F1 agent traffic is
  single-stream, where k=2 costs −16%; whether that outweighs the F2 long gain
  is unresolved.
- True mixed-load effect of split-CKV (needs live TTFT/TPOT telemetry, not F2).
- SJF queue effect on the F3 p99 tail (module staged, live window pending).
