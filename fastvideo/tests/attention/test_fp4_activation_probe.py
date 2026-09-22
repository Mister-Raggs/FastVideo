import torch

from fastvideo.attention.fp4_activation_probe import (
    output_error,
    resolve_block_indices,
    sampled_logit_distribution,
    tensor_distribution,
)


def test_resolve_block_indices_defaults_to_first_middle_last() -> None:
    assert resolve_block_indices(28, None) == [0, 14, 27]
    assert resolve_block_indices(1, None) == [0]


def test_resolve_block_indices_validates_and_deduplicates() -> None:
    assert resolve_block_indices(4, [3, 0, 3]) == [0, 3]

    try:
        resolve_block_indices(4, [4])
    except ValueError as exc:
        assert "outside [0, 3]" in str(exc)
    else:
        raise AssertionError("out-of-range block index was accepted")


def test_tensor_and_logit_statistics_are_json_safe() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 2, 17, 8)
    key = torch.randn(1, 2, 9, 8)

    distribution = tensor_distribution(query, token_samples=5)
    logits = sampled_logit_distribution(query, key, 8**-0.5, query_samples=4, key_samples=3)

    assert distribution["sampled_elements"] <= 1 * 2 * 5 * 8
    assert len(distribution["per_head_rms"]) == 2
    assert logits["sampled_query_tokens"] == 4
    assert logits["sampled_key_tokens"] == 3
    assert len(logits["per_head_abs_max"]) == 2


def test_output_error_reports_global_and_per_head_metrics() -> None:
    reference = torch.randn(1, 3, 7, 8)
    metrics = output_error(reference.clone(), reference)

    assert metrics["finite"] is True
    assert abs(metrics["cosine_similarity"] - 1.0) < 1e-6
    assert metrics["relative_l2"] == 0.0
    assert len(metrics["per_head_cosine"]) == 3
