# SPDX-License-Identifier: Apache-2.0

import json
import math
import pickle

import pytest
import torch
from torch import nn

from fastvideo.benchmarks.dynamic_precision_probe import (
    DynamicPrecisionProbeLinearMethod,
    ProbeRecord,
    ProbeSpec,
    ProbeState,
    analyze_records,
    block_index_from_prefix,
    relative_l1_block_change,
    relative_l2_error,
    run_block_probe,
    spearman_correlation,
)


class _ProbeLayer(nn.Module):

    def __init__(self, method):
        super().__init__()
        self.quant_method = method


class _ProbeBlock(nn.Module):

    def __init__(self, state):
        super().__init__()
        self.first = _ProbeLayer(DynamicPrecisionProbeLinearMethod("blocks.5.to_q", state))
        self.second = _ProbeLayer(DynamicPrecisionProbeLinearMethod("blocks.5.ffn.fc_in", state))


def test_relative_metrics_are_normalized():
    reference = torch.tensor([1.0, 2.0])
    candidate = torch.tensor([2.0, 4.0])

    assert relative_l1_block_change(reference, candidate) == pytest.approx(1.0)
    assert relative_l2_error(reference, candidate) == pytest.approx(1.0)


def test_block_index_parsing_is_scoped_to_blocks():
    assert block_index_from_prefix("transformer.blocks.12.attn2.to_q") == 12
    assert block_index_from_prefix("transformer.layers.12.to_q") is None


def test_probe_state_survives_worker_serialization(tmp_path):
    state = ProbeState(ProbeSpec(output_path=str(tmp_path / "probe.jsonl"), block_indices=(5, ),
                                 error_step_indices=(2, )))

    restored = pickle.loads(pickle.dumps(state))

    assert restored.spec == state.spec
    # Use the public writer after unpickling to prove its lock was rebuilt.
    restored.write(ProbeRecord(kind="block", step_index=1, block_index=5, gamma=0.1))
    assert (tmp_path / "probe.jsonl").exists()


def test_block_probe_returns_dense_baseline_and_restores_state(tmp_path):
    state = ProbeState(ProbeSpec(output_path=str(tmp_path / "probe.jsonl"), block_indices=(5, ),
                                 error_step_indices=(2, )))
    block = _ProbeBlock(state)

    def forward(_block, hidden_states):
        offset = 1.0 if state.active_prefix is None else 2.0
        return hidden_states + offset

    result = run_block_probe(forward, block, (torch.ones(2), ), {}, step_index=2)

    assert torch.equal(result, torch.full((2, ), 2.0))
    assert state.active_prefix is None
    assert state.in_block_replay is False
    records = [json.loads(line) for line in (tmp_path / "probe.jsonl").read_text().splitlines()]
    assert [record["kind"] for record in records] == ["block", "projection", "projection"]
    assert all(record.get("relative_l2_error", 0.0) > 0 for record in records[1:])


def test_block_probe_restores_state_after_replay_failure(tmp_path):
    state = ProbeState(ProbeSpec(output_path=str(tmp_path / "probe.jsonl"), block_indices=(5, ),
                                 error_step_indices=(2, )))
    block = _ProbeBlock(state)

    def forward(_block, hidden_states):
        if state.active_prefix is not None:
            raise RuntimeError("synthetic FP4 failure")
        return hidden_states + 1

    with pytest.raises(RuntimeError, match="synthetic FP4 failure"):
        run_block_probe(forward, block, (torch.ones(2), ), {}, step_index=2)

    assert state.active_prefix is None
    assert state.in_block_replay is False


def test_analyzer_requires_exact_previous_step_gamma():
    records = [
        {"kind": "block", "step_index": 1, "block_index": 0, "gamma": 0.1},
        {"kind": "block", "step_index": 2, "block_index": 0, "gamma": 0.2},
        {
            "kind": "projection",
            "step_index": 2,
            "block_index": 0,
            "projection": "blocks.0.to_q",
            "gamma": 0.2,
            "relative_l2_error": 0.3,
        },
        {"kind": "block", "step_index": 4, "block_index": 0, "gamma": 0.4},
        {
            "kind": "projection",
            "step_index": 4,
            "block_index": 0,
            "projection": "blocks.0.to_q",
            "gamma": 0.4,
            "relative_l2_error": 0.5,
        },
    ]

    summary = analyze_records(records)

    assert summary["paired_samples"] == 1
    assert summary["skipped_without_previous_step"] == 1


def test_spearman_handles_ties_without_scipy():
    assert spearman_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 8.0]) == pytest.approx(1.0)
    assert math.isnan(spearman_correlation([1.0], [2.0]))
