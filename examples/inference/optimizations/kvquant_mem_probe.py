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

Model/frames are overridable:

    FV_KVQ_MODEL=...  FV_KVQ_FRAMES=81  FV_KVQ_OUTPUT_DIR=...

Generation settings mirror ``basic_self_forcing_causal.py`` (offload off, single
GPU) so the measurement describes the path a single-GPU user actually runs;
enabling offload moves weights off the device and changes the peak composition.
"""

import os
import time

from fastvideo import SamplingParam, VideoGenerator

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

    generator = VideoGenerator.from_pretrained(
        model_name,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=False,
        dit_cpu_offload=False,
    )

    sampling_param = SamplingParam.from_pretrained(model_name)
    frames = os.getenv("FV_KVQ_FRAMES")
    if frames:
        sampling_param.num_frames = int(frames)
    sampling_param.seed = 42

    print(f"[step0] model={model_name} num_frames={sampling_param.num_frames} seed={sampling_param.seed}")

    start = time.perf_counter()
    generator.generate_video(PROMPT, output_path=output_dir, save_video=True, sampling_param=sampling_param)
    print(f"[step0] wall {time.perf_counter() - start:.1f}s")
    print("[step0] see the [kv-mem] lines above and the JSON report for the stage breakdown")


if __name__ == "__main__":
    main()
