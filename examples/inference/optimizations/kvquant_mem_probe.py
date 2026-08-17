"""D6 Step-0: which pipeline stage owns peak memory on the unwindowed KV path?

D1 measured the SFWan KV caches at ~5.6 GiB, ~69% of peak, and D6's memory
thesis rests on that share. But share-of-bytes is not the same as owning the
high-water mark: on Matrix-Game, D1 shrank the KV allocation and peak did not
move (6029.46 MB @w5 vs 6029.3 MB @w6).

The KV caches here are locals in ``CausalDMDDenosingStage.forward`` -- they are
released before VAE decode runs. So if decode owns the peak, quantizing KV
cannot reduce peak at all, and D6's memory argument closes here.

Run this before writing any quantizer:

    FASTVIDEO_KV_MEM_PROBE=1 \
    FASTVIDEO_KV_MEM_PROBE_OUTPUT=$HOME/kvquant/step0_sfwan21.json \
        python examples/inference/optimizations/kvquant_mem_probe.py

Knobs:

    FV_KVQ_MODEL=...  FV_KVQ_FRAMES=81  FV_KVQ_OUTPUT_DIR=...
    FV_KVQ_VAE_TILING=1|0        enable/disable VAE tiling (Wan ships it OFF)
    FV_KVQ_VAE_FEATURE_CACHE=0   required for tiling to actually engage on Wan
    FV_KVQ_NO_OFFLOAD=1          pin every module resident (all-resident arm)

Offload is left at FastVideo's defaults (all True) unless FV_KVQ_NO_OFFLOAD is
set. The first run of this probe mirrored basic_self_forcing_causal.py, which
disables text-encoder and DiT offload -- that pinned ~21.7 GiB of weights
resident and measured the least favourable configuration for the KV question.
Defaults describe the path a commodity-GPU user actually runs.
"""

import os
import time

from fastvideo import SamplingParam, VideoGenerator
from fastvideo.configs.pipelines import PipelineConfig

DEFAULT_MODEL = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"

PROMPT = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
          "wide with interest. The playful yet serene atmosphere is complemented by soft "
          "natural light filtering through the petals. Mid-shot, warm and cheerful tones.")


def main() -> None:
    # Set here rather than relying on the caller so a bare run still measures
    # something. Workers are spawned, and a spawned child inherits os.environ,
    # so this reaches the worker that actually holds the pipeline.
    os.environ.setdefault("FASTVIDEO_KV_MEM_PROBE", "1")

    model_name = os.getenv("FV_KVQ_MODEL", DEFAULT_MODEL)
    output_dir = os.getenv("FV_KVQ_OUTPUT_DIR", "video_samples_kvquant_step0")

    pipeline_config = PipelineConfig.from_pretrained(model_name)
    tiling = os.getenv("FV_KVQ_VAE_TILING")
    if tiling is not None:
        # Wan configs override the base default to False, so SFWan decodes all
        # frames in one shot -- that untiled transient is what owns peak memory.
        pipeline_config.vae_tiling = tiling != "0"

    if os.getenv("FV_KVQ_VAE_FEATURE_CACHE") == "0":
        # WanVAE.decode() branches on use_feature_cache and never consults
        # use_tiling, so enable_tiling() is a silent no-op on the Wan family.
        # Turning the feature cache off routes decode through
        # ParallelTiledVAE.decode, which is the only path that honours tiling.
        # This changes the decode algorithm -- SSIM-gate before believing it.
        pipeline_config.vae_config.use_feature_cache = False

    kwargs: dict[str, object] = {}
    if os.getenv("FV_KVQ_NO_OFFLOAD", "0") != "0":
        kwargs.update(text_encoder_cpu_offload=False, dit_cpu_offload=False, dit_layerwise_offload=False)

    generator = VideoGenerator.from_pretrained(
        model_name,
        num_gpus=1,
        use_fsdp_inference=False,
        pipeline_config=pipeline_config,
        **kwargs,
    )

    sampling_param = SamplingParam.from_pretrained(model_name)
    frames = os.getenv("FV_KVQ_FRAMES")
    if frames:
        sampling_param.num_frames = int(frames)
    sampling_param.seed = 42

    print(f"[step0] model={model_name} num_frames={sampling_param.num_frames} seed={sampling_param.seed} "
          f"vae_tiling={pipeline_config.vae_tiling} "
          f"feature_cache={getattr(pipeline_config.vae_config, 'use_feature_cache', None)} "
          f"no_offload={bool(kwargs)}")

    start = time.perf_counter()
    generator.generate_video(PROMPT, output_path=output_dir, save_video=True, sampling_param=sampling_param)
    print(f"[step0] wall {time.perf_counter() - start:.1f}s")
    print("[step0] see the [kv-mem] lines above and the JSON report for the stage breakdown")


if __name__ == "__main__":
    main()
