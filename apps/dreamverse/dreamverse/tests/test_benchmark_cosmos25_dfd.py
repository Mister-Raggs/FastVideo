from __future__ import annotations

from argparse import Namespace

import numpy as np

import benchmark_cosmos25_dfd as benchmark


def _args(tmp_path) -> Namespace:
    return Namespace(
        bootstrap_model=str(tmp_path / "bootstrap"),
        continuation_model=str(tmp_path / "continuation"),
        image=str(tmp_path / "image.png"),
        role="continuation",
        prompt="prompt",
        prompt_file=None,
        arm="core",
        output_dir=str(tmp_path / "output"),
        warmups=1,
        runs=2,
        seed=42,
        height=704,
        width=1280,
        bootstrap_frames=77,
        continuation_frames=81,
        fps=24,
        quality_scale=4,
        dry_run=True,
    )


def test_core_matrix_is_quality_safe_and_residency_focused() -> None:
    assert benchmark._selected_arms("core") == (
        "lazy_sdpa",
        "resident_sdpa_regional",
        "hybrid_sdpa_regional",
    )
    assert all(benchmark.ARMS[name].attention_backend == "TORCH_SDPA" for name in benchmark.CORE_ARMS)
    assert benchmark.ARMS["lazy_sdpa"].continuation_lazy_module_load is True
    assert benchmark.ARMS["resident_sdpa"].continuation_lazy_module_load is False
    assert benchmark.ARMS["resident_sdpa_regional"].continuation_inference_torch_compile is True
    assert benchmark.ARMS["hybrid_sdpa_regional"].bootstrap_lazy_module_load is True
    assert benchmark.ARMS["hybrid_sdpa_regional"].bootstrap_inference_torch_compile is False
    assert benchmark.ARMS["hybrid_sdpa_regional"].continuation_compile_vae is False
    assert benchmark.ARMS["hybrid_sdpa_regional_vae"].continuation_compile_vae is True


def test_model_config_keeps_the_generation_contract_fixed(tmp_path) -> None:
    args = _args(tmp_path)

    config = benchmark._model_config(args, benchmark.ARMS["resident_sdpa_regional"])

    assert config["height"] == 704
    assert config["width"] == 1280
    assert config["bootstrap_num_frames"] == 77
    assert config["continuation_num_frames"] == 81
    assert config["num_inference_steps"] == 4
    assert config["seed"] == 42
    assert config["bootstrap_lazy_module_load"] is False
    assert config["continuation_lazy_module_load"] is False
    assert config["bootstrap_inference_torch_compile"] is True
    assert config["continuation_inference_torch_compile"] is True
    assert config["bootstrap_compile_vae"] is False
    assert config["continuation_compile_vae"] is False
    assert config["startup_warmup"] is False


def test_quality_comparison_reports_exact_and_changed_samples(tmp_path) -> None:
    reference_path = tmp_path / "reference.npz"
    exact_path = tmp_path / "exact.npz"
    changed_path = tmp_path / "changed.npz"
    reference = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    changed = reference.copy()
    changed[0, 0, 0, 0] = 10
    np.savez_compressed(reference_path, frames=reference)
    np.savez_compressed(exact_path, frames=reference)
    np.savez_compressed(changed_path, frames=changed)

    exact = benchmark._quality_comparison(str(reference_path), str(exact_path))
    different = benchmark._quality_comparison(str(reference_path), str(changed_path))

    assert exact["identical"] is True
    assert exact["psnr_db"] == float("inf")
    assert different["identical"] is False
    assert different["mean_abs"] > 0
    assert different["psnr_db"] < float("inf")
