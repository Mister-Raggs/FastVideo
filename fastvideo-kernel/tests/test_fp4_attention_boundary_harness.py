import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "examples/inference/optimizations/fp4_attention_boundary.py"


def _list_cases(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--list-cases"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cross_preset_contains_only_unequal_noncausal_shapes() -> None:
    result = _list_cases("--preset", "cross", "--heads", "16")

    assert result.returncode == 0, result.stderr
    cases = json.loads(result.stdout)
    assert cases
    assert all(case["heads"] == 16 for case in cases)
    assert all(case["q_len"] != case["kv_len"] for case in cases)
    assert not any(case["causal"] for case in cases)


def test_custom_case_round_trips() -> None:
    result = _list_cases("--case", "cosmos-cross,1,16,32768,512,128,false,0.3")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [{
        "label": "cosmos-cross",
        "batch": 1,
        "heads": 16,
        "q_len": 32768,
        "kv_len": 512,
        "head_dim": 128,
        "causal": False,
        "input_scale": 0.3,
    }]


def test_padding_preset_straddles_tile_boundaries() -> None:
    result = _list_cases("--preset", "padding")

    assert result.returncode == 0, result.stderr
    cases = json.loads(result.stdout)
    lengths = {(case["q_len"], case["kv_len"]) for case in cases}
    assert (127, 127) in lengths
    assert (129, 129) in lengths
    assert (4095, 127) in lengths
    assert (4096, 129) in lengths


def test_unequal_causal_case_is_rejected() -> None:
    result = _list_cases("--case", "bad-cross,1,4,4096,512,128,true,0.3")

    assert result.returncode != 0
    assert "causal cases require q_len == kv_len" in result.stderr
