# SPDX-License-Identifier: Apache-2.0
# Regression tests for the glm5_next split-CKV extension of PR #1005.
# Ported from local-inference-lab/vllm tests/v1/attention/test_b12x_sparse_mla_api.py
# (branch canon/glm53-744b-dcp-draft-head) and adapted: the upstream heuristic test
# asserts "GLM5Next keeps its own policy"; our overlay intentionally inverts that
# for glm5_next, so the non-sparse case below is the only remaining negative.
# Run inside the canary image: pytest tests/test_split_ckv_glm5next.py

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder,
)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseMetadataBuilder,
    _use_b12x_split_ckv_gather,
)


@pytest.mark.parametrize(
    ("is_sparse_glm", "num_decode_tokens", "num_prefills", "prefill_tokens", "expected"),
    [
        (True, 8, 1, 8192, True),
        (True, 0, 1, 8192, False),  # pure prefill takes the full-batch gather
        (True, 8, 0, 0, False),  # pure decode keeps the query exchange
        (True, 8, 1, 8, False),  # too few prefill rows to pay for a gather
        (True, 8, 1, 600000, False),  # beyond the gathered-cache capacity
        # Upstream expected False here ("GLM5Next keeps its own policy");
        # our overlay extends the split to glm5_next, so the negative case
        # is now only a non-sparse-GLM backend.
        (False, 8, 1, 8192, False),
    ],
)
def test_b12x_split_ckv_gather_needs_a_mixed_sparse_glm_batch(
    is_sparse_glm: bool,
    num_decode_tokens: int,
    num_prefills: int,
    prefill_tokens: int,
    expected: bool,
) -> None:
    assert (
        _use_b12x_split_ckv_gather(
            enabled=True,
            is_sparse_glm=is_sparse_glm,
            dcp_world_size=2,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=prefill_tokens,
            min_tokens=16,
            max_tokens=524288,
        )
        is expected
    )


def _bare_glm_dsa_ckv_metadata_builder(
    *, dcp_rank: int, max_tokens: int = 64, max_reqs: int = 4, split: bool = False
) -> B12xMLASparseMetadataBuilder:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False
    builder._is_glm_dsa = True
    builder._ckv_gather_requested = True
    builder._ckv_split_requested = split
    builder.dcp_world_size = 2
    builder.dcp_rank = dcp_rank
    builder.cp_kv_cache_interleave_size = 1
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder._ckv_max_reqs = max_reqs
    builder._max_speculative_decode_query_len = 4
    builder.cache_seq_lens_per_token_buffer = torch.zeros(max_tokens, dtype=torch.int32)
    builder.ckv_selected_indices_buffer = torch.empty(
        (max_tokens, 2048), dtype=torch.int32
    )
    builder.ckv_active_counts_buffer = torch.empty(max_tokens, dtype=torch.int32)
    builder.dcp_rank_req_lens_buffer = torch.empty((2, max_reqs), dtype=torch.int32)
    builder.dcp_rank_req_starts_buffer = torch.empty((2, max_reqs), dtype=torch.int32)
    builder.dcp_local_cu_seq_lens_buffer = torch.empty(max_reqs + 1, dtype=torch.int32)
    return builder


def _bare_glm5next_ckv_metadata_builder(
    *, dcp_rank: int, max_tokens: int = 64, max_reqs: int = 4, split: bool = False
) -> B12xMLASparseMetadataBuilder:
    builder = _bare_glm_dsa_ckv_metadata_builder(
        dcp_rank=dcp_rank, max_tokens=max_tokens, max_reqs=max_reqs, split=split
    )
    builder.requires_glm_next_selector_metadata = True
    builder._is_glm_dsa = False
    builder._capture_state_slot_ids = torch.zeros(max_reqs, dtype=torch.int64)
    builder._capture_state_is_fresh = torch.zeros(max_reqs, dtype=torch.bool)
    builder._capture_num_accepted_tokens = torch.zeros(max_reqs, dtype=torch.int64)
    builder._capture_is_prefilling = torch.zeros(max_reqs, dtype=torch.bool)
    return builder


def _mixed_batch_common(query_lens, seq_lens, dcp_rank: int) -> SimpleNamespace:
    rows = sum(query_lens)
    query_start_loc = torch.tensor([0, query_lens[0], rows], dtype=torch.int32)
    global_seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=rows,
        max_query_len=max(query_lens),
        seq_lens=global_seq_lens,
        dcp_local_seq_lens=(global_seq_lens + 1 - dcp_rank) // 2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens_cpu=global_seq_lens,
        seq_lens_cpu_upper_bound=global_seq_lens,
        positions=None,
        is_prefilling=torch.tensor([False, True]),
    ), query_start_loc, global_seq_lens


@pytest.mark.parametrize("dcp_rank", [0, 1])
@pytest.mark.parametrize("split", [False, True])
def test_glm_dsa_builder_gathers_only_the_prefill_requests_of_a_mixed_batch(
    monkeypatch: pytest.MonkeyPatch, dcp_rank: int, split: bool
) -> None:
    # Upstream regression: request 0 decodes 4 rows over 50 tokens; request 1
    # prefills a 24-token chunk of a 40-token prompt.
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=1,
            num_decodes=1,
            num_decode_tokens=4,
            dcp_ckv_gather_eligible=False,
            dcp_ckv_split_eligible=False,
        ),
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    builder = _bare_glm_dsa_ckv_metadata_builder(dcp_rank=dcp_rank, split=split)
    common, _, global_seq_lens = _mixed_batch_common([4, 24], [50, 40], dcp_rank)

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert metadata.dcp_ckv_gather_eligible is False
    assert metadata.dcp_ckv_split_eligible is split
    if not split:
        return
    assert metadata.dcp_rank_req_lens.tolist() == [[0, 20], [0, 20]]
    assert metadata.dcp_rank_req_starts.tolist() == [[0, 0], [0, 0]]
    assert metadata.dcp_local_cu_seq_lens.tolist() == [0, 0, 20]
    assert metadata.dcp_local_total_tokens == 20
    assert torch.equal(metadata.global_cache_seq_lens_per_req, global_seq_lens)


@pytest.mark.parametrize("dcp_rank", [0, 1])
@pytest.mark.parametrize("split", [False, True])
def test_glm5next_builder_gathers_only_the_prefill_requests_of_a_mixed_batch(
    monkeypatch: pytest.MonkeyPatch, dcp_rank: int, split: bool
) -> None:
    # Our extension: the same mixed batch through a glm5_next builder
    # (requires_glm_next_selector_metadata=True) must stage the same
    # decode-excluded gather spans.
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=1,
            num_decodes=1,
            num_decode_tokens=4,
            dcp_ckv_gather_eligible=False,
            dcp_ckv_split_eligible=False,
        ),
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    builder = _bare_glm5next_ckv_metadata_builder(dcp_rank=dcp_rank, split=split)
    common, _, global_seq_lens = _mixed_batch_common([4, 24], [50, 40], dcp_rank)

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        selector_state_slot_ids=torch.zeros(2, dtype=torch.int64),
        selector_state_is_fresh=torch.ones(2, dtype=torch.bool),
        selector_num_accepted_tokens=torch.zeros(2, dtype=torch.int64),
        selector_is_prefilling=torch.tensor([False, True]),
    )

    assert metadata.dcp_ckv_gather_eligible is False
    assert metadata.dcp_ckv_split_eligible is split
    if not split:
        return
    assert metadata.dcp_rank_req_lens.tolist() == [[0, 20], [0, 20]]
    assert metadata.dcp_rank_req_starts.tolist() == [[0, 0], [0, 0]]
    assert metadata.dcp_local_cu_seq_lens.tolist() == [0, 0, 20]
    assert metadata.dcp_local_total_tokens == 20
    assert torch.equal(metadata.global_cache_seq_lens_per_req, global_seq_lens)
