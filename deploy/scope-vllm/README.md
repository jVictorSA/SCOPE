# Isolated local vLLM stack for SCOPE benchmarks

This directory contains a small Docker Compose stack for running SCOPE's
EvalScope benchmark evolution against a local OpenAI-compatible vLLM endpoint.

It is intentionally separate from any other vLLM deployment on the host:

- project name `scope-vllm`
- container name `scope-vllm`
- host port `127.0.0.1:8080`
- its own Hugging Face and vLLM cache directories
- no embedder service
- no image build step

The stack uses the prebuilt NVIDIA image
`nvcr.io/nvidia/vllm:26.08-py3`, which is already available on the DGX Spark.

## Safety constraints

Do not stop, restart, recreate, or modify other vLLM projects or containers on
the host. This stack is designed to run alongside them using a different
project name, container name, port, and cache.

The local API is bound to `127.0.0.1` only. Do not expose it externally.

## Memory constraints

The Compose defaults are deliberately conservative:

```text
--max-model-len 8192
--max-num-seqs 1
--max-num-batched-tokens 1024
--enforce-eager
```

The model files select the GPU memory fraction:

| Model file | Model | GPU memory fraction |
|---|---|---:|
| `.env.3b` | `Qwen/Qwen2.5-3B-Instruct-AWQ` | `0.05` |
| `.env.7b` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `0.07` |

These values assume another vLLM workload is already using most of the DGX
Spark's unified memory. Do not raise them while other workloads are running.

Before starting, check available memory and the current GPU processes:

```bash
free -h
nvidia-smi
```

If available memory is very low, start with the 3B model. Do not attempt 14B or
larger models while other vLLM workloads are active.

## Prerequisites

- Docker with GPU access.
- The NVIDIA vLLM image already present, or permission to pull it.
- Python 3.10 or newer for EvalScope.
- A SCOPE checkout with the benchmark extra installed.

From the root of the SCOPE checkout:

```bash
python3 --version
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[benchmarks]"
```

## Validate the templates without starting anything

From the root of the SCOPE checkout, pass the Compose file explicitly with
`-f`. The repository root has no default `docker-compose.yml`, so Compose
cannot find the stack without it:

```bash
docker compose \
  -f deploy/scope-vllm/docker-compose.yml \
  --env-file deploy/scope-vllm/.env.common.example \
  --env-file deploy/scope-vllm/.env.3b.example \
  config
```

Use `.env.7b.example` instead of `.env.3b.example` to validate the 7B variant.

This only renders and validates the configuration. It does not pull, create, or
start a container.

## Host-only deployment directory

Choose a directory outside the SCOPE repository and export it:

```bash
export SCOPE_VLLM_DIR="/path/to/your/scope-vllm"
```

Create it and copy the templates:

```bash
mkdir -p "$SCOPE_VLLM_DIR"
cp deploy/scope-vllm/docker-compose.yml "$SCOPE_VLLM_DIR/"
cp deploy/scope-vllm/.env.common.example "$SCOPE_VLLM_DIR/.env.common"
cp deploy/scope-vllm/.env.3b.example "$SCOPE_VLLM_DIR/.env.3b"
cp deploy/scope-vllm/.env.7b.example "$SCOPE_VLLM_DIR/.env.7b"
chmod 600 "$SCOPE_VLLM_DIR"/.env.common "$SCOPE_VLLM_DIR"/.env.3b "$SCOPE_VLLM_DIR"/.env.7b
mkdir -p "$SCOPE_VLLM_DIR/data/huggingface" "$SCOPE_VLLM_DIR/data/vllm"
```

Edit `$SCOPE_VLLM_DIR/.env.common` and replace
`VLLM_API_KEY=replace-with-a-strong-key` with a real secret.

`VLLM_API_KEY` is not an OpenAI, DeepSeek, or Hugging Face key. It is an
arbitrary shared secret that protects this local vLLM endpoint. Generate a new
one:

```bash
openssl rand -hex 32
```

Put that value in `.env.common`:

```dotenv
VLLM_API_KEY=<generated-value>
```

The same value is passed to SCOPE as `--api-key`. The vLLM server rejects
requests that do not send `Authorization: Bearer <generated-value>`.

`HF_TOKEN` is separate and only needed for gated Hugging Face models. The
Qwen models used here are public, so it can stay empty.

Never commit `.env.common`, `.env.3b`, `.env.7b`, or `data/`.

## Start the 3B stack

```bash
cd "$SCOPE_VLLM_DIR"
docker compose \
  --env-file .env.common \
  --env-file .env.3b \
  -p scope-vllm \
  up -d --force-recreate
```

Watch startup:

```bash
docker logs -f scope-vllm
```

The first start may download the model. The health check allows up to 15
minutes for startup.

## Validate the endpoint

Load the shared environment:

```bash
cd "$SCOPE_VLLM_DIR"
set -a
. ./.env.common
set +a
```

List models:

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

Send a system-role chat request, because EvalScope uses system prompts:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Reply with OK."}
    ],
    "max_tokens": 16
  }'
```

If the system role fails, stop and investigate the model chat template before
running SCOPE.

## Run the SCOPE smoke test

From the root of the SCOPE checkout:

```bash
python examples/evalscope_evolution.py \
  --suite smoke \
  --provider local \
  --model local-model \
  --api-url http://127.0.0.1:8080/v1 \
  --api-key "$VLLM_API_KEY" \
  --eval-batch-size 1 \
  --train-limit 2 \
  --eval-limit 2 \
  --max-steps-per-round 1 \
  --rounds 1 \
  --max-input-chars 2000 \
  --synthesizer-provider local \
  --synthesizer-model local-model \
  --synthesizer-api-url http://127.0.0.1:8080/v1 \
  --synthesizer-api-key "$VLLM_API_KEY"
```

The suite file does not change between models. Only the backend env file and
the underlying model change.

## Switch to the 7B model

```bash
cd "$SCOPE_VLLM_DIR"
docker compose \
  --env-file .env.common \
  --env-file .env.7b \
  -p scope-vllm \
  up -d --force-recreate
```

Switch back with the `.env.3b` command.

Because `VLLM_SERVED_MODEL=local-model`, SCOPE commands stay the same. Use
`--model-id` if you want run artifacts to record which real model was used.

## Monitor and stop

Monitor host memory and GPU:

```bash
watch -n 5 'free -h; nvidia-smi'
```

Container logs:

```bash
docker logs -f scope-vllm
```

Stop only this stack:

```bash
cd "$SCOPE_VLLM_DIR"
docker compose \
  --env-file .env.common \
  --env-file .env.3b \
  -p scope-vllm \
  down
```

Stop the container immediately if available memory falls below a safe margin,
swap grows significantly, other containers become unhealthy, or vLLM reports
allocation failures:

```bash
docker stop scope-vllm
```

## Synthesis quality

SCOPE uses the same local model for benchmark evaluation and guideline
synthesis. A 3B model is useful for validating the pipeline but may be too weak
for structured guideline generation. A 7B model is better but can still produce
malformed or low-quality guidelines.

If synthesis is unreliable, keep the local model as the benchmark target and
switch only the `--synthesizer-*` options to a stronger hosted model.

## Troubleshooting

- This stack intentionally avoids `--swap-space`; the NVIDIA vLLM 26.08 image
  does not accept it. If another vLLM release rejects a flag, inspect
  `vllm serve --help` for that image and remove the unsupported entry.
- If Compose cannot access the GPU, check how the existing vLLM containers are
  configured and mirror that mechanism. A common fallback is replacing the
  `deploy.resources.reservations.devices` block with `gpus: all`.
- If vLLM reports insufficient memory for KV cache, lower `VLLM_MAX_MODEL_LEN`
  first. Do not raise `VLLM_GPU_MEMORY_UTILIZATION` aggressively while other
  workloads are running.
- If the container exits immediately, inspect `docker logs scope-vllm` before
  retrying. `restart: "no"` is intentional so an allocation failure does not
  create a restart loop.

## No image build

There is no `docker build` step. The stack uses the prebuilt NVIDIA image.

If a future change adds a SCOPE application container, that is a separate
deployment and should not be mixed into this vLLM stack.
