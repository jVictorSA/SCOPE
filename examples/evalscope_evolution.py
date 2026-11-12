#!/usr/bin/env python
"""Evolve a benchmark prompt with SCOPE and EvalScope.

This example runs a small, round-based benchmark loop:

1. Evaluate a held-out sample of each dataset with the benchmark's default
   system prompt (the baseline).
2. Run a training slice, inspect EvalScope's per-sample reviews, and ask SCOPE
   to synthesize guidelines from incorrect answers.
3. Append the learned tactical guidelines to the prompt and re-run the held-out
   evaluation.
4. Repeat for the requested number of rounds.

The benchmark data is downloaded and managed by EvalScope.  SCOPE only reads
EvalScope's public output files, so this integration does not vendor or
duplicate benchmark datasets.

Examples:
    # List the repository's benchmark suites
    python examples/evalscope_evolution.py --list-suites

    # Run a named suite (datasets and evolution settings come from the suite)
    python examples/evalscope_evolution.py \\
        --suite smoke --provider deepseek --model deepseek-chat

    # Quick DeepSeek pilot (requires DEEPSEEK_API_KEY)
    python examples/evalscope_evolution.py \\
        --provider deepseek --model deepseek-chat \\
        --datasets gsm8k arc --train-limit 20 --eval-limit 20 --rounds 2

    # Validate suite wiring without calling a model
    python examples/evalscope_evolution.py \\
        --suite smoke --provider local --model dummy \\
        --api-url http://127.0.0.1:8000/v1 --api-key EMPTY --dry-run

    # Use an OpenAI-compatible local server for evaluation and DeepSeek for
    # SCOPE's synthesis model.
    python examples/evalscope_evolution.py \\
        --provider local --model Qwen/Qwen2.5-7B-Instruct \\
        --api-url http://localhost:8000/v1 --api-key EMPTY \\
        --synthesizer-provider deepseek --synthesizer-model deepseek-chat \\
        --datasets gsm8k --rounds 3

Results are saved under ``examples/results/evalscope_evolution/`` by default.
Use ``--output-dir``, ``--run-name``, or ``--temp`` to change that behavior.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Add parent directory to path for local development.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _output import _redact, add_output_args, run_context
from dotenv import load_dotenv

from scope.benchmarks.suites import (
    SuiteError,
    get_suite,
    list_suites,
    merge_json_dicts,
    suite_to_arg_defaults,
)

load_dotenv()


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-3-5-sonnet-20241022",
    "litellm": "gpt-4o-mini",
    "local": None,
}

DEFAULT_EVAL_TYPES = {
    "openai": "openai_api",
    "deepseek": "openai_api",
    "anthropic": "anthropic_api",
    "litellm": "litellm",
    "local": "openai_api",
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "litellm": "OPENAI_API_KEY",
    "local": None,
}

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
}

DEFAULT_SUITE_FILE = Path(
    os.getenv("SCOPE_BENCHMARK_SUITES", str(REPO_ROOT / "benchmarks" / "suites.json"))
).expanduser()


def _limit_value(value):
    """Parse an EvalScope limit without turning ``30`` into ``30.0``.

    EvalScope interprets an integer as a sample count and a float as a
    fraction of the dataset.  Preserving the distinction matters.
    """
    text = str(value).strip()
    try:
        if any(marker in text.lower() for marker in (".", "e")):
            return float(text)
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "limit must be an integer sample count or a float fraction"
        ) from exc


def _parse_json_object(value, label):
    if value is None or value == "":
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{label} must be a JSON object")
    return parsed


def _split_datasets(values):
    datasets = []
    for value in values:
        datasets.extend(part.strip() for part in value.split(",") if part.strip())
    return datasets


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evolve benchmark prompts with SCOPE and EvalScope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    benchmark = parser.add_argument_group("benchmark")
    benchmark.add_argument(
        "--suite",
        default=None,
        help="Benchmark suite name from the suite file (see --list-suites)",
    )
    benchmark.add_argument(
        "--suite-file",
        default=str(DEFAULT_SUITE_FILE),
        help="JSON suite manifest (default: benchmarks/suites.json or " "SCOPE_BENCHMARK_SUITES)",
    )
    benchmark.add_argument(
        "--list-suites",
        action="store_true",
        help="List available benchmark suites and exit",
    )
    benchmark.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="EvalScope dataset names (space- or comma-separated), e.g. gsm8k arc",
    )
    benchmark.add_argument(
        "--train-limit",
        type=_limit_value,
        default=30,
        help="Samples per subset for training (default: 30; float = fraction)",
    )
    benchmark.add_argument(
        "--eval-limit",
        type=_limit_value,
        default=30,
        help="Samples per subset for held-out evaluation (default: 30)",
    )
    benchmark.add_argument(
        "--train-seed", type=int, default=42, help="Training shuffle seed (default: 42)"
    )
    benchmark.add_argument(
        "--eval-seed", type=int, default=4242, help="Evaluation shuffle seed (default: 4242)"
    )
    benchmark.add_argument(
        "--dataset-hub",
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="Where EvalScope downloads datasets (default: modelscope)",
    )
    benchmark.add_argument("--dataset-dir", default=None, help="EvalScope dataset cache directory")
    benchmark.add_argument(
        "--dataset-args",
        default=None,
        help="JSON object of EvalScope dataset_args overrides",
    )
    benchmark.add_argument(
        "--generation-config",
        default=None,
        help="JSON object passed to EvalScope as generation_config",
    )

    evolution = parser.add_argument_group("evolution")
    evolution.add_argument(
        "--rounds", type=int, default=3, help="Number of train/eval rounds (default: 3)"
    )
    evolution.add_argument(
        "--max-steps-per-round",
        type=int,
        default=20,
        help="Maximum failed samples to learn from per round (default: 20)",
    )
    evolution.add_argument(
        "--quality-samples",
        type=int,
        default=0,
        help="Also learn from this many successful samples per round (default: 0)",
    )
    evolution.add_argument(
        "--failure-threshold",
        type=float,
        default=1.0,
        help="Samples below this score are treated as failures (default: 1.0)",
    )
    evolution.add_argument(
        "--base-prompt",
        default="",
        help="Extra text appended after the benchmark's default system prompt",
    )
    evolution.add_argument(
        "--base-prompt-file",
        default=None,
        help="Read extra base-prompt text from a file",
    )
    evolution.add_argument(
        "--max-input-chars",
        type=int,
        default=6000,
        help="Maximum benchmark input characters passed to SCOPE (default: 6000)",
    )
    evolution.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after N rounds without improvement (0 disables early stopping)",
    )
    evolution.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum score improvement to reset patience (default: 0.0)",
    )
    evolution.add_argument(
        "--agent-name-prefix",
        default="evalscope",
        help="Prefix for the SCOPE agent name (default: evalscope)",
    )
    evolution.add_argument(
        "--no-quality-analysis",
        action="store_true",
        help="Disable SCOPE analysis of successful samples",
    )
    evolution.add_argument(
        "--quality-frequency",
        type=int,
        default=1,
        help="Analyze every N successful samples (default: 1)",
    )
    evolution.add_argument(
        "--auto-accept-threshold",
        choices=["all", "low", "medium", "high"],
        default="medium",
        help="SCOPE auto-accept confidence threshold (default: medium)",
    )
    evolution.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the benchmark configuration without running it",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--provider",
        choices=sorted(DEFAULT_MODELS),
        default="deepseek",
        help="Benchmark model provider (default: deepseek)",
    )
    model.add_argument(
        "--model", default=None, help="Benchmark model name (provider-specific default)"
    )
    model.add_argument(
        "--model-id", default=None, help="Optional EvalScope model_id for output folders"
    )
    model.add_argument("--api-url", default=None, help="OpenAI-compatible API base URL")
    model.add_argument("--api-key", default=None, help="API key (overrides environment variable)")
    model.add_argument(
        "--eval-type",
        default=None,
        help="EvalScope eval_type (inferred from provider by default)",
    )
    model.add_argument(
        "--eval-batch-size", type=int, default=4, help="EvalScope batch size (default: 4)"
    )
    model.add_argument(
        "--no-ignore-errors",
        action="store_true",
        help="Abort EvalScope on an individual sample error",
    )

    synthesis = parser.add_argument_group("synthesis model")
    synthesis.add_argument(
        "--synthesizer-provider",
        choices=sorted(DEFAULT_MODELS),
        default=None,
        help="Provider for SCOPE's guideline model (default: same as --provider)",
    )
    synthesis.add_argument(
        "--synthesizer-model",
        default=None,
        help="Model for SCOPE's guideline synthesis (default: same as --model)",
    )
    synthesis.add_argument(
        "--synthesizer-api-url", default=None, help="Synthesis model API base URL"
    )
    synthesis.add_argument("--synthesizer-api-key", default=None, help="Synthesis model API key")

    benchmark.add_argument(
        "--extra-task-kwargs",
        default=None,
        help="JSON object of additional EvalScope TaskConfig arguments",
    )

    add_output_args(parser)
    return parser


def _parse_suite_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--suite-file", default=str(DEFAULT_SUITE_FILE))
    parser.add_argument("--list-suites", action="store_true")
    known, _ = parser.parse_known_args(argv)
    return known


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    suite_args = _parse_suite_args(argv)
    suite_path = Path(suite_args.suite_file).expanduser()

    if suite_args.list_suites:
        try:
            suites = list_suites(suite_path)
        except SuiteError as exc:
            parser.error(str(exc))
        print(f"Available benchmark suites in {suite_path}:")
        for entry in suites:
            description = entry["description"]
            if description:
                print(f"  {entry['name']:<16} {description}")
            else:
                print(f"  {entry['name']}")
        raise SystemExit(0)

    suite = {}
    if suite_args.suite:
        try:
            suite = get_suite(suite_path, suite_args.suite)
        except SuiteError as exc:
            parser.error(str(exc))
        parser.set_defaults(**suite_to_arg_defaults(suite))

    args = parser.parse_args(argv)
    if not args.datasets:
        parser.error("--datasets is required unless --suite provides datasets")
    args.datasets = _split_datasets(args.datasets)

    try:
        cli_dataset_args = _parse_json_object(args.dataset_args, "--dataset-args")
        cli_generation_config = _parse_json_object(args.generation_config, "--generation-config")
        cli_extra_task_kwargs = _parse_json_object(args.extra_task_kwargs, "--extra-task-kwargs")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    args.dataset_args = merge_json_dicts(suite.get("dataset_args"), cli_dataset_args)
    args.generation_config = merge_json_dicts(suite.get("generation_config"), cli_generation_config)
    args.extra_task_kwargs = merge_json_dicts(suite.get("extra_task_kwargs"), cli_extra_task_kwargs)
    args.suite_config = suite
    args.suite_file_path = str(suite_path)
    return args


def _resolve_model(provider, model):
    resolved = model or DEFAULT_MODELS.get(provider)
    if not resolved:
        raise ValueError(f"--model is required for provider '{provider}'")
    return resolved


def _resolve_api_key(provider, explicit):
    if explicit:
        return explicit
    env_name = API_KEY_ENV.get(provider)
    return os.getenv(env_name) if env_name else None


def _resolve_api_url(provider, explicit):
    if explicit:
        return explicit
    if provider == "deepseek":
        return (
            os.getenv("DEEPSEEK_API_BASE")
            or os.getenv("DEEPSEEK_BASE_URL")
            or DEFAULT_BASE_URLS["deepseek"]
        )
    if provider == "openai":
        return os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    return None


def _create_model(provider, model, api_key, base_url):
    if provider == "openai":
        from scope.models import create_openai_model

        return create_openai_model(model, api_key=api_key, base_url=base_url)
    if provider == "deepseek":
        from scope.models import create_deepseek_model

        return create_deepseek_model(model, api_key=api_key, base_url=base_url)
    if provider == "anthropic":
        from scope.models import create_anthropic_model

        return create_anthropic_model(model, api_key=api_key)
    if provider == "litellm":
        from scope.models import create_litellm_model

        return create_litellm_model(model, api_key=api_key, base_url=base_url)
    if provider == "local":
        from scope.models import create_openai_model

        if not base_url:
            raise ValueError("--api-url is required for provider 'local'")
        return create_openai_model(model, api_key=api_key or "EMPTY", base_url=base_url)
    raise ValueError(f"Unsupported provider: {provider}")


def _read_base_prompt(args):
    parts = []
    if args.base_prompt_file:
        parts.append(Path(args.base_prompt_file).expanduser().read_text(encoding="utf-8"))
    if args.base_prompt:
        parts.append(args.base_prompt)
    return "\n\n".join(part.strip() for part in parts if part.strip())


async def main(args):
    from scope import SCOPEOptimizer
    from scope.benchmarks import EvalScopeEvolutionRunner, EvolutionConfig

    datasets = _split_datasets(args.datasets)
    if not datasets:
        raise ValueError("At least one dataset is required")

    benchmark_model = _resolve_model(args.provider, args.model)
    benchmark_api_key = _resolve_api_key(args.provider, args.api_key)
    benchmark_api_url = _resolve_api_url(args.provider, args.api_url)
    if args.provider == "local" and not benchmark_api_key:
        benchmark_api_key = "EMPTY"

    synth_provider = args.synthesizer_provider or args.provider
    if args.synthesizer_model:
        synth_model_name = args.synthesizer_model
    elif synth_provider == args.provider:
        synth_model_name = benchmark_model
    else:
        synth_model_name = _resolve_model(synth_provider, None)
    synth_api_key = _resolve_api_key(synth_provider, args.synthesizer_api_key)
    synth_api_url = _resolve_api_url(synth_provider, args.synthesizer_api_url)

    config = EvolutionConfig(
        datasets=datasets,
        provider=args.provider,
        model=benchmark_model,
        model_id=args.model_id,
        api_url=benchmark_api_url,
        api_key=benchmark_api_key,
        eval_type=args.eval_type or DEFAULT_EVAL_TYPES[args.provider],
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        rounds=args.rounds,
        max_steps_per_round=args.max_steps_per_round,
        quality_samples=args.quality_samples,
        failure_threshold=args.failure_threshold,
        max_input_chars=args.max_input_chars,
        base_prompt=_read_base_prompt(args),
        dataset_args=args.dataset_args,
        dataset_hub=args.dataset_hub,
        dataset_dir=args.dataset_dir,
        generation_config=args.generation_config,
        eval_batch_size=args.eval_batch_size,
        ignore_errors=not args.no_ignore_errors,
        extra_task_kwargs=args.extra_task_kwargs,
        agent_name_prefix=args.agent_name_prefix,
        enable_quality_analysis=not args.no_quality_analysis,
        quality_analysis_frequency=args.quality_frequency,
        auto_accept_threshold=args.auto_accept_threshold,
        patience=args.patience,
        min_improvement=args.min_improvement,
    )

    with run_context("evalscope_evolution", args) as run:
        print(f"Benchmark model: {args.provider}/{benchmark_model}")
        print(f"Synthesis model: {synth_provider}/{synth_model_name}")
        print(f"Datasets: {', '.join(datasets)}")
        if args.suite_config:
            print(f"Suite: {args.suite} ({args.suite_file_path})")

        suite_artifact = None
        if args.suite_config:
            suite_artifact = {
                "name": args.suite,
                "path": args.suite_file_path,
                "config": args.suite_config,
            }
            run.save_json("benchmark_suite.json", suite_artifact)

        if args.dry_run:
            resolved = {
                "suite": suite_artifact,
                "config": _redact(asdict(config)),
            }
            run.save_json("resolved_config.json", resolved)
            print("\nResolved benchmark configuration:")
            print(json.dumps(resolved, indent=2, default=str))
            run.save_summary({"dry_run": True, "result": resolved})
            print(f"\nResults directory: {run.run_dir}")
            return resolved

        if args.provider in {"openai", "deepseek", "anthropic"} and not benchmark_api_key:
            env_name = API_KEY_ENV.get(args.provider)
            raise ValueError(
                f"{env_name} is not set. Export it or pass --api-key for "
                f"provider '{args.provider}'."
            )

        synthesizer_model = _create_model(
            synth_provider,
            synth_model_name,
            synth_api_key,
            synth_api_url,
        )
        wrapped_synthesizer = run.wrap_model(synthesizer_model)

        def optimizer_factory(dataset, agent_name, exp_path):
            return SCOPEOptimizer(
                synthesizer_model=wrapped_synthesizer,
                exp_path=exp_path,
                enable_quality_analysis=config.enable_quality_analysis,
                quality_analysis_frequency=config.quality_analysis_frequency,
                auto_accept_threshold=config.auto_accept_threshold,
                store_history=True,
            )

        runner = EvalScopeEvolutionRunner(
            config=config,
            run_dir=run.run_dir,
            optimizer_factory=optimizer_factory,
        )
        summary = await runner.run()
        run.save_summary(summary)
        run.save_json("benchmark_summary.json", summary)

        print("\nBenchmark evolution summary")
        for item in summary["datasets"]:
            print(
                f"  {item['dataset']}: baseline={item['baseline_score']} "
                f"best={item['best_score']} (round {item['best_round']}) "
                f"final={item['final_score']}"
            )
        print(f"\nResults directory: {run.run_dir}")
        return summary


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
