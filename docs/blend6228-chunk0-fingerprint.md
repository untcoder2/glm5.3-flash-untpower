# Blend store-leg: register chunk 0 fingerprints (backport of LMCache #6228)

## Problem

The standalone LMCache sidecar's blend module skipped chunk 0 when a
position-0 store enqueued its fingerprints: the store leg assumed the
standard prefix-lookup path owns the first chunk. In practice blend-sliced
stores never get that chunk indexed anywhere — the blend matcher serves
token-range lookups *inside* blend slices, and the prefix path does not
cover them. Result: the first chunk of the shared preamble (the identical
system prompt every session starts with) is a permanent cache miss, and
every new session re-prefills exactly the most shared content.

## Fix

Backport of [LMCache #6228](https://github.com/LMCache/LMCache/pull/6228)
to the local-inference-lab lmcache line (`0.5.3.dev453`, main @
`8ef537d6481b`): the store leg registers chunk 0 natively. The
`start_chunk_idx` plumbing is removed from `BlendTokenRangeMatcher.on_new_token_hashes`
and from the async fingerprint-queue job tuple — the matcher indexes every
chunk it is handed. Dedup semantics are unchanged: a re-store of already
registered content still publishes no registration event.

→ [diff](../diffs/20261106-blend6228-chunk0-fingerprint.diff)
(14+/23−, `lmcache/v1/multiprocess/modules/blend.py`)

## Verification

- Live in production since 06.11 on the GLM-5.3-Flash stack (50 GiB CPU L1
  sidecar, `engine_driven` transfer). The store path was confirmed via the
  sidecar's Prometheus metrics on `:8091/metrics` (L1 write/read/eviction
  counters advancing with blend registrations; CB events live on the event
  bus, not in container logs — don't grep for them).
- Semantic check: a first store of shared content now publishes a
  registration including chunk 0; a re-store of known content still
  publishes nothing.

## Status

Superseded in our own deployment line by the v0.5.7 rebase (upstream #6228
is native in the `blend/` package; our lookup leg additionally passes
`chunk_hashes` into `begin_lookup`). Kept for deployments still on the
`0.5.3.dev453` line. No standalone test file — verified on the live stack.
