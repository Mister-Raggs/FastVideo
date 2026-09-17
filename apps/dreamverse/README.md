# Dreamverse

Dreamverse is the FastVideo realtime video generation & editing platform. It lives in this monorepo under `apps/dreamverse/`.

**Deploy on:** [local GPU](#quick-start-local-gpu) · [self-hosted B200 (SSH)](#server-b200-deployment-ssh) · [Docker](docker/README.md) · [Modal](scripts/modal/README.md)

## Install Dreamverse

You can install Dreamverse using one of the methods below.

### Method 1: With uv pip

```bash
pip install --upgrade pip
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
UV_TORCH_BACKEND=cu126 uv pip install "fastvideo[dreamverse]"
```

### Method 2: From source

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo

pip install --upgrade pip
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
UV_TORCH_BACKEND=cu126 uv pip install -e ".[dreamverse]"
```

Use `UV_TORCH_BACKEND=cu130` instead on CUDA 13.

### Method 3: Using Docker

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo

apps/dreamverse/docker/docker_build.sh
```

See `apps/dreamverse/docker/README.md` for Docker build and run option details.

## Optional: Building FFmpeg For Better Performance

For full streaming performance in a non-Docker install, build a custom FFmpeg
binary from a FastVideo source checkout. The command below is repo-relative,
so run it from the repository root:

```bash
bash apps/dreamverse/scripts/install_native_ffmpeg.sh
```

The installer supports Linux `x86_64` and `aarch64`. It prefers conda-forge
triplet compilers when those commands are on `PATH`, otherwise it falls back to
system `gcc`/`g++` (plain venv). On `x86_64`, x264's hand-tuned SIMD also
requires `nasm`; install via whichever path fits your host:

```bash
sudo apt install nasm                       # Debian/Ubuntu
conda install -c conda-forge nasm           # inside an active conda env
```

No sudo and no conda? Build `nasm` from source (~30s, installs into `$HOME`):

```bash
(
  mkdir -p "$HOME/src" "$HOME/opt" && cd "$HOME/src"
  curl -fsSL -O https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz
  tar -xf nasm-2.16.03.tar.gz && cd nasm-2.16.03
  ./configure --prefix="$HOME/opt/nasm" && make -j"$(nproc)" && make install
)
export PATH="$HOME/opt/nasm/bin:$PATH"      # add to ~/.bashrc to persist
```

The installer writes to `~/opt/ffmpeg-native/` and emits
`apps/dreamverse/scripts/ffmpeg-env.sh`. Source it before starting the backend
so Dreamverse uses the custom FFmpeg binary:

```bash
source apps/dreamverse/scripts/ffmpeg-env.sh
dreamverse-server
```

Docker images already run this FFmpeg build during image creation and source the
generated environment file at container startup.

## Launch Dreamverse

Start the backend with the installed Dreamverse commands:

```bash
dreamverse-server --port 8009
dreamverse-mock-server --port 8009
```

### Run Dreamverse with FastH3

Select the VSA data-free FastH3 Preview profile when you start the backend:

```bash
DREAMVERSE_MODEL_ID=fast-h3 dreamverse-server --port 8009
```

The `fast-h3` profile uses four visible GPUs by default. It loads the `MiniMaxAI/MiniMax-H3` base checkpoint and the
`vsa-datafree/adapter_model.safetensors` adapter from
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`. Each request generates a 124-frame, 768×1344 video with
synchronized audio and five sigma-grid points. Dreamverse uses the last frame of each segment as first-frame
conditioning for the following segment.

Set `CUDA_VISIBLE_DEVICES` when you need to choose the four physical GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DREAMVERSE_MODEL_ID=fast-h3 dreamverse-server --port 8009
```

> **Expect a slow first boot.** With `torch.compile` and startup warmup enabled
> (the default), the backend compiles the segment 1 and segment 2 inference
> paths before it reports ready — this can take **tens of minutes on a cold
> cache**, regardless of how you deploy (local, server, Docker, or Modal).
> `/healthz` responds as soon as the process is up; `/readyz` stays `503` until
> warmup finishes. To defer compilation until the first generated request while
> testing, set `FASTVIDEO_ENABLE_STARTUP_WARMUP=0` before starting the backend.

## Frontend Setup

Install the web dependencies once from the FastVideo checkout:

```bash
cd apps/dreamverse/web
npm ci
```

The frontend package uses `package-lock.json`; use npm for installs and scripts.

## Quick Start: Local GPU

### Start Backend

Export the API keys used for prompt rewrite and prompt enhancement:

```bash
export CEREBRAS_API_KEY=...
export GROQ_API_KEY=...
```

If you built the optional native FFmpeg binary above, source its environment
file in the same shell before starting the backend:

```bash
source apps/dreamverse/scripts/ffmpeg-env.sh
dreamverse-server --host 0.0.0.0 --port 8009
```

The Dreamverse backend defaults to `0.0.0.0:8009` and starts one GPU worker on
the first visible GPU by default.

### Cosmos Predict2.5 DFD continuation (experimental)

Dreamverse can combine two converted Cosmos Predict2.5 2B packages: the
distilled Text2World student creates an unconditioned first segment, then the
Data-Forcing Distillation (DFD) Video2World student conditions each later
segment on the prior terminal frame. Point the runtime at both local converted
packages:

```bash
export DREAMVERSE_MODEL_ID=cosmos25-dfd
export DREAMVERSE_MODEL_PATH=/path/to/Cosmos-Predict2.5-2B-Distilled-TrigFlow-FastVideo
export DREAMVERSE_COSMOS25_DFD_MODEL_PATH=/path/to/Cosmos-Predict2.5-2B-DFD-FastVideo
export ENABLE_TORCH_COMPILE=0
dreamverse-server --host 0.0.0.0 --port 8009
```

The backend loads both model roles before reporting ready. Both use BF16, Torch
SDPA, 704x1280 output, 24 FPS, and four steps. On unified-memory GB10 systems,
the one-use T2W role loads lazily and releases its components after bootstrap;
the repeatedly used DFD role stays resident and uses regional DiT compile. It
also skips the two generated startup warmups by default; the first real request
is therefore a cold request. Bootstrap segments contain 77 frames. DFD segments
contain 81 decoded frames, but Dreamverse drops the repeated conditioning frame
before streaming, leaving 80 new frames. An initial user image selects DFD
immediately without treating that first frame as a cross-segment overlap.

The runtime profile can be changed without editing the registry:

```bash
export DREAMVERSE_COSMOS25_LAZY_MODULE_LOAD=1        # reproduce the old GB10 policy
export DREAMVERSE_COSMOS25_INFERENCE_TORCH_COMPILE=1 # regional BF16 DiT compile
export DREAMVERSE_COSMOS25_STARTUP_WARMUP=1          # warm both generation paths
export DREAMVERSE_COSMOS25_ATTENTION_BACKEND=flash_attn
```

The two shared lifecycle variables override both roles. More specific
`DREAMVERSE_COSMOS25_BOOTSTRAP_*` and
`DREAMVERSE_COSMOS25_CONTINUATION_*` variants override one role; their suffixes
are `LAZY_MODULE_LOAD` and `INFERENCE_TORCH_COMPILE`.

`DREAMVERSE_COSMOS25_BOOTSTRAP_COMPILE_VAE` and
`DREAMVERSE_COSMOS25_CONTINUATION_COMPILE_VAE` are experimental decoder-only
compile gates. They remain disabled until the fixed-shape GB10 decode benchmark
demonstrates a warm latency win and decoded-frame parity.

Use benchmark selection `--arm decode` to compare the validated hybrid profile
with the otherwise-identical continuation VAE-compile arm.

On a GB10 DFD continuation, regional compile reduced a correct-input request
from 147.25 to 125.17 seconds (15.0%). Its decoded all-frame sample measured
48.18 dB PSNR and 0.45 mean absolute pixel error against eager SDPA, and passed
visual review. The profile does not change the model weights, four-step
schedule, attention implementation, seed, resolution, or frame count.

The profile uses a 30-minute session lease because sequential generation on
GB10-class hardware can exceed Dreamverse's five-minute default while the GPU
is still making progress. Deployments can override the lease with
`FASTVIDEO_SESSION_TIMEOUT_SECONDS`.

Cosmos does not produce audio, so the backend supplies duration-matched silent
24 kHz audio for the existing browser streaming contract and trims 1,000 audio
samples with each repeated DFD boundary frame. Runtime LoRA changes are not
supported.

To compare the prior lazy-loading baseline with resident inference and
precision-preserving regional compile, run the isolated DreamVerse matrix on
the target GPU. Each arm preserves BF16, the same model packages, schedule,
seed, dimensions, and frame count. The harness reports initialization and stage
medians, writes a review MP4 and boundary image, and computes decoded-frame
parity against the baseline:

```bash
PYTHONPATH="$PWD/apps/dreamverse:$PWD" python \
  apps/dreamverse/dreamverse/benchmarks/benchmark_cosmos25_dfd.py \
  --bootstrap-model "$DREAMVERSE_MODEL_PATH" \
  --continuation-model "$DREAMVERSE_COSMOS25_DFD_MODEL_PATH" \
  --image /path/to/conditioning.png \
  --arm core
```

Use `--arm all` only when FlashAttention is installed. Promote an optional arm
only after both its numerical receipt and generated review video are accepted;
the benchmark intentionally makes no unmeasured latency claim.

### Check Readiness

In another shell, verify that the backend process is alive:

```bash
curl http://localhost:8009/healthz
```

Then wait for GPU workers and startup warmup to finish:

```bash
curl http://localhost:8009/readyz
```

You can also run the same readiness path with:

```bash
BACKEND_HOST=localhost BACKEND_PORT=8009 apps/dreamverse/scripts/smoke_local.sh
```

If a backend is already running and you only want the script to probe it:

```bash
DREAMVERSE_SMOKE_START_BACKEND=0 apps/dreamverse/scripts/smoke_local.sh
```

### Start Frontend

Start the frontend:

```bash
cd apps/dreamverse/web
BACKEND_HOST=localhost BACKEND_PORT=8009 npm run dev
```

Open `http://localhost:5299`.

## Server B200 deployment (SSH)

Deploying on a remote GPU host (for example a B200 box) is a local install run
over SSH, plus a few server-specific concerns. Two paths:

### Option A: Native (source install)

SSH in, then follow [Install → From source](#method-2-from-source) and
(recommended) [Building FFmpeg](#optional-building-ffmpeg-for-better-performance),
then start the backend as in [Quick Start: Local GPU](#quick-start-local-gpu).
For a remote host, a few things differ from localhost:

- Bind all interfaces: `dreamverse-server --host 0.0.0.0 --port 8009`.
- Point the frontend/client at the host: `BACKEND_HOST=<b200-host> BACKEND_PORT=8009 npm run dev`.
- Keep the backend alive across SSH sessions (`tmux` / `systemd` / `nohup`).
- Expose / firewall port `8009`, or front it with a reverse proxy + auth.

### Option B: Docker (on the server)

SSH in, then follow [Install → Using Docker](#method-3-using-docker) and the run
steps in [`docker/README.md`](docker/README.md):

```bash
CEREBRAS_API_KEY="<key>" GROQ_API_KEY="<key>" apps/dreamverse/docker/docker_run.sh
```

## Quick Start: Mock Backend (For UI development)

The mock server emulates the Dreamverse backend protocol and streams a
synthetic FFmpeg-generated fMP4 clip, so the frontend can run without a GPU.

```bash
dreamverse-mock-server --latency 200 --port 8009
```

## Tests

Run the focused backend tests that validate local startup wiring, config, GPU
selection, and mock-server behavior:

```bash
pytest apps/dreamverse/dreamverse/tests/test_config.py \
  apps/dreamverse/dreamverse/tests/test_entrypoints.py \
  apps/dreamverse/dreamverse/tests/test_gpu_pool.py \
  apps/dreamverse/dreamverse/tests/test_minimax_h3_generation.py \
  apps/dreamverse/dreamverse/tests/test_mock_server.py -q
```

Run the broader Dreamverse backend suite:

```bash
pytest apps/dreamverse/dreamverse/tests -q
```

Run the frontend tests:

```bash
cd apps/dreamverse/web
npm test
```

Run the frontend e2e tests:

```bash
cd apps/dreamverse/web
npm run e2e
```

## Troubleshooting

`dreamverse-server` exits with an install hint

- install the Dreamverse extra with `UV_TORCH_BACKEND=cu126 uv pip install -e ".[dreamverse]"` from a
  source checkout, or `UV_TORCH_BACKEND=cu126 uv pip install "fastvideo[dreamverse]"` from PyPI
  (`cu130` on CUDA 13).

Prompt-provider environment variable errors

- set `CEREBRAS_API_KEY`
- set `GROQ_API_KEY`
- direct `dreamverse-server` launches do not source `~/.env`; export the keys
  in the shell or use the bundled launch scripts, which source `~/.env`.

`/readyz` stays at `503`

- wait for model loading and startup warmup to finish
- confirm a compatible CUDA GPU is visible to the process
- check backend logs for worker startup or warmup failures
- for a startup/debug pass without warmup, set
  `FASTVIDEO_ENABLE_STARTUP_WARMUP=0` before starting the backend

Only one GPU is used

- this is the default local behavior
- set `FASTVIDEO_GPU_COUNT=<N>` to start N GPU worker subprocesses inside one
  backend instance
- set `FASTVIDEO_GPU_COUNT=all` to start one worker for every visible GPU
- use `CUDA_VISIBLE_DEVICES` first if you need to pin the visible GPU set

Frontend cannot connect to backend

- confirm the backend is running on `8009`; if not, point the frontend at it
  with `BACKEND_HOST=<host> BACKEND_PORT=<port> npm run dev`
- confirm `http://localhost:8009/healthz` responds before starting the frontend
- confirm `http://localhost:8009/readyz` returns `200` before clicking Generate
- use `apps/dreamverse/scripts/smoke_local.sh` for a repeatable local startup
  check

Mock backend fails during startup

- install FFmpeg or set `FASTVIDEO_FFMPEG_BIN` to an FFmpeg binary
- for non-mock local GPU streaming performance, use the native FFmpeg installer
  described above

## Notes

Dreamverse owns its backend app under `apps/dreamverse/dreamverse/`. It expects
`dreamverse-server`, not `fastvideo serve`.
