# Benchmark suites

This directory contains backend-agnostic benchmark presets for the optional
[EvalScope](../scope/benchmarks/evalscope_runner.py) integration.

Suites describe **what to run** — datasets and prompt-evolution settings. They
deliberately do **not** contain a provider, model name, API URL, API key, output
directory, or run name. That keeps a suite reusable when the benchmark backend
changes (for example, from a hosted API to a future local LLM server).

No benchmark datasets are stored in this repository. EvalScope downloads and
caches them in its own dataset cache.

## Quick start

Install the optional dependency (EvalScope requires Python 3.10+):

```bash
pip install -e ".[benchmarks]"
```

List the suites in `benchmarks/suites.json`:

```bash
python examples/evalscope_evolution.py --list-suites
```

Run one with an explicit model backend:

```bash
export DEEPSEEK_API_KEY=sk-...
python examples/evalscope_evolution.py \
  --suite smoke \
  --provider deepseek --model deepseek-chat
```

Explicit command-line flags override suite values:

```bash
python examples/evalscope_evolution.py \
  --suite reasoning \
  --provider deepseek --model deepseek-chat \
  --rounds 1 --eval-limit 5 --train-limit 5
```

Use `--suite-file` or the `SCOPE_BENCHMARK_SUITES` environment variable to load
a different manifest. `--dry-run` resolves the configuration and writes it to
the run directory without creating a model client or invoking EvalScope.

```bash
python examples/evalscope_evolution.py \
  --suite smoke \
  --provider local --model dummy \
  --api-url http://127.0.0.1:8000/v1 --api-key EMPTY \
  --dry-run
```

## File format

`suites.json` is plain JSON so it can be reviewed without extra dependencies:

```json
{
  "version": 1,
  "suites": {
    "smoke": {
      "description": "One small GSM8K round for backend connectivity checks.",
      "datasets": ["gsm8k"],
      "train_limit": 5,
      "eval_limit": 5,
      "rounds": 1,
      "max_steps_per_round": 3,
      "dataset_args": {
        "gsm8k": {"few_shot_num": 0}
      }
    }
  }
}
```

Rules:

- Only `version` and `suites` are allowed at the top level, and `version` must
  be `1`.
- Each suite requires a non-empty `datasets` list. Duplicate names are removed.
- Unknown suite keys are rejected so typos do not silently change a run.
- `dataset_args`, `generation_config`, and `extra_task_kwargs` must be JSON
  objects. They are deep-merged with the corresponding CLI JSON arguments,
  with CLI values winning.
- Suite values are validated through `EvolutionConfig`.

Allowed keys:

```text
description
datasets
train_limit
eval_limit
train_seed
eval_seed
rounds
max_steps_per_round
quality_samples
failure_threshold
max_input_chars
base_prompt
dataset_args
dataset_hub
dataset_dir
generation_config
eval_batch_size
ignore_errors
extra_task_kwargs
agent_name_prefix
enable_quality_analysis
quality_analysis_frequency
auto_accept_threshold
patience
min_improvement
```

Model/API/output settings are intentionally not allowed. Pass them on the
command line; they are recorded in the redacted `run_config.json`.

## Local LLM backends

The local inference backend is not selected yet. The recommended contract is an
**OpenAI-compatible `/v1` endpoint**, which covers vLLM, SGLang, TGI,
llama.cpp server, Ollama, LM Studio, LocalAI, and similar servers.

```bash
export LOCAL_BASE_URL=http://localhost:8000/v1
export LOCAL_MODEL=your-local-model

python examples/evalscope_evolution.py \
  --suite smoke \
  --provider local --model "$LOCAL_MODEL" \
  --api-url "$LOCAL_BASE_URL" --api-key EMPTY
```

Small local models may be weak at SCOPE's structured guideline synthesis. You
can benchmark the local model while using a stronger synthesis model:

```bash
python examples/evalscope_evolution.py \
  --suite smoke \
  --provider local --model "$LOCAL_MODEL" \
  --api-url "$LOCAL_BASE_URL" --api-key EMPTY \
  --synthesizer-provider deepseek --synthesizer-model deepseek-chat
```

If the eventual backend is not OpenAI-compatible, it can be supported later
through EvalScope's `litellm` eval type or a new provider case in
`examples/evalscope_evolution.py`. Suite files should not need changes.

## Choosing benchmarks

Start with exact-match tasks that do not require an LLM judge:

- `gsm8k`
- `arc`

Defer judge-based, tool-based, or Docker-based benchmarks until the local
backend and scoring costs are understood.

Add local-LLM integration tests only after the backend is chosen. The existing
`tests/test_evalscope_benchmarks.py` and `tests/test_evalscope_suites.py` run
without a network connection, EvalScope, or model credentials.
