"""Evolve a benchmark prompt with SCOPE using EvalScope.

This module is the bridge between SCOPE and `EvalScope
<https://github.com/modelscope/evalscope>`_.  It runs an optional, round-based
loop:

1. Evaluate a held-out slice of a benchmark with the current prompt.
2. Run a training slice with the same prompt and collect per-sample reviews.
3. Feed failures (and optionally successful examples) to
   :meth:`SCOPEOptimizer.on_step_complete`.
4. Add the resulting tactical guidelines to the prompt for the next evaluation.
5. Repeat.

EvalScope is imported only when a run starts.  The rest of SCOPE, including the
command-line example, can therefore be installed and used without EvalScope.
"""

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("scope.benchmarks.evalscope")

DEFAULT_MAX_INPUT_CHARS = 6000

ScoreValue = Union[int, float, bool]


class BenchmarkMismatch(Exception):  # noqa: N818 - domain term kept for API clarity
    """Error raised for one failed benchmark sample.

    SCOPE treats the most recent error as useful context when synthesizing a
    guideline.  A dedicated exception makes it clear that the model answered a
    benchmark question incorrectly, without losing the expected and actual
    answers in a generic error message.
    """

    def __init__(
        self,
        message: str,
        sample_id: Optional[Union[int, str]] = None,
        dataset: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
    ):
        super().__init__(message)
        self.sample_id = sample_id
        self.dataset = dataset
        self.expected = expected
        self.actual = actual


@dataclass
class BenchmarkSample:
    """One reviewed benchmark sample."""

    sample_id: Union[int, str]
    input_text: str
    target: List[str]
    prediction: str
    score: float
    subset: str = "main"
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_prediction: Dict[str, Any] = field(default_factory=dict)
    raw_review: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Result of one EvalScope evaluation phase."""

    phase: str
    score: Optional[float]
    report: Dict[str, Any]
    work_dir: Path
    samples: List[BenchmarkSample] = field(default_factory=list)


@dataclass
class DatasetEvolutionResult:
    """Summary of prompt evolution for one dataset."""

    dataset: str
    agent_name: str
    baseline_score: Optional[float]
    final_score: Optional[float]
    best_score: Optional[float]
    best_round: int
    prompt_paths: List[str] = field(default_factory=list)
    rounds: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "agent_name": self.agent_name,
            "baseline_score": self.baseline_score,
            "final_score": self.final_score,
            "best_score": self.best_score,
            "best_round": self.best_round,
            "prompt_paths": self.prompt_paths,
            "rounds": self.rounds,
        }


@dataclass
class EvolutionConfig:
    """Configuration for :class:`EvalScopeEvolutionRunner`.

    The fields mirror the useful EvalScope ``TaskConfig`` options plus the
    SCOPE-specific controls.  Unknown EvalScope arguments can be supplied via
    ``extra_task_kwargs``.
    """

    datasets: List[str]
    provider: str = "openai"
    model: Optional[str] = None
    model_id: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    eval_type: Optional[str] = None

    train_limit: Optional[Union[int, float]] = 30
    eval_limit: Optional[Union[int, float]] = 30
    train_seed: int = 42
    eval_seed: int = 4242

    rounds: int = 3
    max_steps_per_round: int = 20
    quality_samples: int = 0
    failure_threshold: float = 1.0
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    base_prompt: str = ""

    dataset_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dataset_hub: str = "modelscope"
    dataset_dir: Optional[str] = None
    generation_config: Dict[str, Any] = field(default_factory=dict)
    eval_batch_size: int = 4
    ignore_errors: bool = True
    extra_task_kwargs: Dict[str, Any] = field(default_factory=dict)

    agent_name_prefix: str = "evalscope"
    enable_quality_analysis: bool = True
    quality_analysis_frequency: int = 1
    auto_accept_threshold: str = "medium"
    patience: int = 0
    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        self.datasets = [str(name).strip() for name in self.datasets if str(name).strip()]
        if not self.datasets:
            raise ValueError("At least one benchmark dataset is required")

        if self.rounds < 0:
            raise ValueError("rounds must be >= 0")
        if self.max_steps_per_round < 0:
            raise ValueError("max_steps_per_round must be >= 0")
        if self.quality_samples < 0:
            raise ValueError("quality_samples must be >= 0")
        if self.quality_analysis_frequency < 1:
            raise ValueError("quality_analysis_frequency must be >= 1")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be >= 0")
        if self.max_input_chars < 0:
            raise ValueError("max_input_chars must be >= 0")
        for label, limit in (("train_limit", self.train_limit), ("eval_limit", self.eval_limit)):
            if limit is not None and float(limit) < 0:
                raise ValueError(f"{label} must be >= 0")
        if not 0.0 <= float(self.failure_threshold) <= 1.0:
            raise ValueError("failure_threshold must be between 0 and 1")


def _safe_name(value: str) -> str:
    """Return a filesystem-safe single path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    return cleaned or "run"


def _content_to_text(content: Any) -> str:
    """Convert an OpenAI/EvalScope message content value to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return _content_to_text(content[key])
        return json.dumps(content, ensure_ascii=False, default=str)
    if isinstance(content, (list, tuple)):
        parts = [_content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _messages_from(prediction: Dict[str, Any], review: Dict[str, Any]) -> List[Any]:
    for source in (prediction, review):
        messages = source.get("messages")
        if isinstance(messages, list) and messages:
            return messages
    return []


def _first_user_text(messages: Sequence[Any]) -> str:
    for message in messages:
        if _message_role(message) == "user":
            return _content_to_text(_message_content(message))
    if messages:
        return _content_to_text(_message_content(messages[0]))
    return ""


def _last_assistant_text(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        if _message_role(message) == "assistant":
            text = _content_to_text(_message_content(message))
            if text:
                return text
    return ""


def _prediction_text(prediction: Dict[str, Any], messages: Sequence[Any]) -> str:
    """Extract the model output from an EvalScope prediction row."""
    model_output = prediction.get("model_output")
    if isinstance(model_output, str):
        return model_output
    if isinstance(model_output, dict):
        completion = model_output.get("completion")
        if isinstance(completion, str):
            return completion

        choices = model_output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    text = _content_to_text(message.get("content"))
                    if text:
                        return text
                text = _content_to_text(first.get("text"))
                if text:
                    return text

        message = model_output.get("message")
        if isinstance(message, dict):
            text = _content_to_text(message.get("content"))
            if text:
                return text

    for key in ("completion", "prediction", "response"):
        value = prediction.get(key)
        if isinstance(value, str) and value:
            return value

    return _last_assistant_text(messages)


def _target_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _extract_review_score(review: Dict[str, Any]) -> Optional[float]:
    """Read the primary score from one EvalScope review row.

    EvalScope has used both ``{"sample_score": {"score": {"value": {...}}}}``
    and flatter review records over time.  This helper accepts the documented
    schema and a few harmless legacy variants.
    """
    sample_score = review.get("sample_score")
    score_obj: Any = None
    if isinstance(sample_score, dict):
        score_obj = sample_score.get("score")

    if score_obj is None:
        score_obj = review.get("score")

    value: Any = None
    main_name: Optional[str] = None

    if isinstance(score_obj, dict):
        value = score_obj.get("value")
        main_name = score_obj.get("main_score_name")
    else:
        value = score_obj

    if isinstance(value, dict):
        if main_name and main_name in value:
            value = value[main_name]
        else:
            numeric_values: List[ScoreValue] = []
            for candidate in value.values():
                if isinstance(candidate, (int, float, bool)):
                    numeric_values.append(candidate)
            if not numeric_values:
                return None
            value = numeric_values[0]

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of pydantic models and paths to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _report_to_dict(report: Any) -> Dict[str, Any]:
    """Convert a report/return value to a plain dictionary."""
    converted = _jsonable(report)
    if isinstance(converted, dict):
        return converted
    if isinstance(converted, list):
        return {"reports": converted}
    return {"value": converted}


def _score_from_metric(metric: Any) -> Optional[float]:
    if not isinstance(metric, dict):
        return None
    value = metric.get("score")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)

    # Older reports only expose subset-level scores.
    subsets = metric.get("subsets") or metric.get("categories")
    if isinstance(subsets, list):
        scores: List[float] = []
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            subset_score = subset.get("score")
            if isinstance(subset_score, (int, float)):
                scores.append(float(subset_score))
        if scores:
            return sum(scores) / len(scores)
    return None


def report_score(report: Any) -> Optional[float]:
    """Extract the headline score from an EvalScope report.

    ``run_task`` versions return either a report object, a report dictionary,
    a list of reports, or a mapping from dataset name to reports.  The headline
    score is always the first/primary metric score.
    """
    if isinstance(report, (list, tuple)):
        for item in report:
            score = report_score(item)
            if score is not None:
                return score
        return None

    data = _report_to_dict(report)
    if not isinstance(data, dict):
        return None

    score = data.get("score")
    if isinstance(score, bool):
        return float(score)
    if isinstance(score, (int, float)):
        return float(score)

    metrics = data.get("metrics")
    if isinstance(metrics, list) and metrics:
        primary = data.get("primary_metric_identity")
        primary_name = None
        if isinstance(primary, dict):
            primary_name = primary.get("name") or primary.get("metric_name")
        elif isinstance(primary, str):
            primary_name = primary

        if primary_name:
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                identity = metric.get("identity")
                name = metric.get("name") or metric.get("metric_name")
                if isinstance(identity, dict):
                    name = identity.get("name") or identity.get("metric_name") or name
                if name == primary_name:
                    score = _score_from_metric(metric)
                    if score is not None:
                        return score

        for metric in metrics:
            score = _score_from_metric(metric)
            if score is not None:
                return score

    # Current EvalScope returns {dataset_name: Report}; older versions may
    # return a list of reports.  Walk containers until a report is found.
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (dict, list)):
            nested = report_score(value)
            if nested is not None:
                return nested

    return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _subset_name(path: Path, dataset: str) -> str:
    stem = path.stem
    prefix = f"{dataset}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem or "main"


def _review_files(phase_dir: Path, dataset: str) -> List[Path]:
    root = phase_dir / "reviews"
    if not root.exists():
        return []
    files = sorted(root.glob(f"**/{dataset}_*.jsonl"))
    if files:
        return files
    return sorted(root.glob("**/*.jsonl"))


def _prediction_for_review(review_path: Path, phase_dir: Path) -> Optional[Path]:
    reviews_root = phase_dir / "reviews"
    try:
        relative = review_path.relative_to(reviews_root)
    except ValueError:
        relative = Path(review_path.name)

    direct = phase_dir / "predictions" / relative
    if direct.exists():
        return direct

    matches = sorted((phase_dir / "predictions").glob(f"**/{review_path.name}"))
    return matches[0] if matches else None


def load_phase_samples(phase_dir: Union[str, Path], dataset: str) -> List[BenchmarkSample]:
    """Load and join EvalScope predictions and reviews for one phase.

    EvalScope writes predictions and reviews into separate directories.  The
    two rows are joined by their sample ``index`` (falling back to positional
    order when an index is absent).  Prediction text and the conversation are
    recovered from the prediction row when possible, so SCOPE sees the actual
    prompt and answer used by the benchmark.
    """
    phase_path = Path(phase_dir)
    samples: List[BenchmarkSample] = []

    for review_path in _review_files(phase_path, dataset):
        predictions: Dict[Any, Dict[str, Any]] = {}
        prediction_path = _prediction_for_review(review_path, phase_path)
        if prediction_path is not None:
            for position, row in enumerate(_load_jsonl(prediction_path)):
                index = row.get("index", position)
                predictions[index] = row

        reviews = _load_jsonl(review_path)
        subset = _subset_name(review_path, dataset)

        for position, review in enumerate(reviews):
            index = review.get("index", position)
            prediction = predictions.get(index)
            if prediction is None and len(predictions) == len(reviews):
                try:
                    prediction = list(predictions.values())[position]
                except IndexError:
                    prediction = None
            prediction = prediction or {}

            score = _extract_review_score(review)
            if score is None:
                logger.debug(
                    "Skipping %s/%s sample %s because it has no numeric score",
                    dataset,
                    subset,
                    index,
                )
                continue

            messages = _messages_from(prediction, review)
            input_text = _first_user_text(messages)
            if not input_text:
                input_text = _content_to_text(prediction.get("input") or review.get("input"))
            prediction_text = _prediction_text(prediction, messages)
            if not prediction_text:
                sample_score = review.get("sample_score")
                score_obj = sample_score.get("score") if isinstance(sample_score, dict) else None
                if isinstance(score_obj, dict):
                    prediction_text = str(
                        score_obj.get("prediction") or score_obj.get("extracted_prediction") or ""
                    )

            metadata = {}
            sample_score = review.get("sample_score")
            if isinstance(sample_score, dict):
                metadata.update(sample_score.get("sample_metadata") or {})
            metadata.update(review.get("metadata") or {})

            samples.append(
                BenchmarkSample(
                    sample_id=review.get("sample_id", index),
                    input_text=input_text,
                    target=_target_list(review.get("target")),
                    prediction=prediction_text,
                    score=score,
                    subset=subset,
                    metadata=_jsonable(metadata) or {},
                    raw_prediction=prediction,
                    raw_review=review,
                )
            )

    return samples


def _load_evalscope() -> Tuple[Any, Any]:
    """Import EvalScope lazily with a helpful error message."""
    try:
        from evalscope import TaskConfig, run_task  # type: ignore

        return TaskConfig, run_task
    except ImportError:
        try:
            from evalscope.config import TaskConfig  # type: ignore
            from evalscope.run import run_task  # type: ignore

            return TaskConfig, run_task
        except ImportError as exc:
            raise ImportError(
                "EvalScope is required for benchmark evolution. "
                "Install it with: pip install 'scope-optimizer[benchmarks]'. "
                "EvalScope currently requires Python 3.10 or newer."
            ) from exc


def get_default_system_prompt(dataset: str) -> str:
    """Return the benchmark's default system prompt, if available."""
    try:
        from evalscope.api.registry import get_benchmark  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency and optional attr
        logger.debug("Could not import the EvalScope benchmark registry: %s", exc)
        return ""

    try:
        benchmark = get_benchmark(dataset)
    except TypeError:
        # Some EvalScope versions require a TaskConfig argument even though
        # they do not use it for reading static benchmark metadata.
        benchmark = get_benchmark(dataset, None)
    except Exception as exc:  # noqa: BLE001 - benchmark may require downloads
        logger.debug("Could not read default prompt for %s: %s", dataset, exc)
        return ""

    return getattr(benchmark, "system_prompt", None) or ""


def _combine_prompts(base: str, extra: str) -> str:
    base = (base or "").strip()
    extra = (extra or "").strip()
    if base and extra:
        return f"{base}\n\n{extra}"
    return base or extra


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


class EvalScopeEvolutionRunner:
    """Run round-based prompt evolution against one or more benchmarks.

    Args:
        config: Evolution settings.
        run_dir: Directory that receives prompts, EvalScope outputs, and
            phase summaries.
        optimizer_factory: Callable ``(dataset, agent_name, exp_path)`` that
            returns an initialized :class:`scope.SCOPEOptimizer`.
        run_task_func: Optional replacement for ``evalscope.run_task``.  It is
            primarily useful for tests.
        task_config_factory: Optional replacement for ``evalscope.TaskConfig``.
            When supplied with ``run_task_func``, EvalScope itself is not
            imported.
        default_prompt_func: Optional replacement for
            :func:`get_default_system_prompt`.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        run_dir: Union[str, Path],
        optimizer_factory: Callable[[str, str, str], Any],
        run_task_func: Optional[Callable[[Any], Any]] = None,
        task_config_factory: Optional[Callable[..., Any]] = None,
        default_prompt_func: Optional[Callable[[str], str]] = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.optimizer_factory = optimizer_factory
        self._run_task_func = run_task_func
        self._task_config_factory = task_config_factory
        self._default_prompt_func = default_prompt_func or get_default_system_prompt
        self._run_task = None

    def _ensure_eval_scope(self) -> None:
        if self._run_task_func is not None and self._task_config_factory is not None:
            self._run_task = self._run_task_func
            return

        task_config_cls, run_task = _load_evalscope()
        if self._task_config_factory is None:
            self._task_config_factory = task_config_cls
        if self._run_task_func is None:
            self._run_task_func = run_task
        self._run_task = self._run_task_func

    def _build_task_config(
        self,
        dataset: str,
        prompt: str,
        limit: Optional[Union[int, float]],
        seed: int,
        work_dir: Path,
    ) -> Any:
        assert self._task_config_factory is not None

        dataset_args = copy.deepcopy(self.config.dataset_args.get(dataset, {}))
        if prompt:
            dataset_args["system_prompt"] = prompt
        dataset_args.setdefault("shuffle", True)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "datasets": [dataset],
            "dataset_args": {dataset: dataset_args},
            "limit": limit,
            "seed": seed,
            "work_dir": str(work_dir),
            "no_timestamp": True,
            "eval_batch_size": self.config.eval_batch_size,
            "ignore_errors": self.config.ignore_errors,
        }
        if self.config.model_id:
            kwargs["model_id"] = self.config.model_id
        if self.config.api_url:
            kwargs["api_url"] = self.config.api_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.eval_type:
            kwargs["eval_type"] = self.config.eval_type
        if self.config.dataset_hub:
            kwargs["dataset_hub"] = self.config.dataset_hub
        if self.config.dataset_dir:
            kwargs["dataset_dir"] = self.config.dataset_dir
        if self.config.generation_config:
            kwargs["generation_config"] = copy.deepcopy(self.config.generation_config)
        kwargs.update(copy.deepcopy(self.config.extra_task_kwargs))

        return self._task_config_factory(**kwargs)

    async def _run_phase(
        self,
        dataset: str,
        phase: str,
        prompt: str,
        limit: Optional[Union[int, float]],
        seed: int,
        work_dir: Path,
    ) -> PhaseResult:
        if self._run_task is None:
            raise RuntimeError("EvalScope backend has not been initialized")

        work_dir.mkdir(parents=True, exist_ok=True)
        task_config = self._build_task_config(dataset, prompt, limit, seed, work_dir)
        logger.info(
            "Running EvalScope phase %s for %s (limit=%s, seed=%s)",
            phase,
            dataset,
            limit,
            seed,
        )
        report = self._run_task(task_config)
        report_dict = _report_to_dict(report)
        score = report_score(report)
        samples = load_phase_samples(work_dir, dataset)

        _write_json(
            work_dir / "phase_result.json",
            {
                "dataset": dataset,
                "phase": phase,
                "score": score,
                "sample_count": len(samples),
                "report": report_dict,
            },
        )
        return PhaseResult(
            phase=phase,
            score=score,
            report=report_dict,
            work_dir=work_dir,
            samples=samples,
        )

    def _get_base_prompt(self, dataset: str) -> str:
        default_prompt = ""
        try:
            default_prompt = self._default_prompt_func(dataset) or ""
        except Exception as exc:  # noqa: BLE001 - a benchmark may be unavailable
            logger.warning("Could not obtain default prompt for %s: %s", dataset, exc)
        return _combine_prompts(default_prompt, self.config.base_prompt)

    def _select_samples(
        self, samples: Sequence[BenchmarkSample]
    ) -> List[Tuple[BenchmarkSample, bool]]:
        failures = [
            sample for sample in samples if sample.score < float(self.config.failure_threshold)
        ]
        successes = [
            sample for sample in samples if sample.score >= float(self.config.failure_threshold)
        ]

        failures.sort(key=lambda sample: (sample.score, str(sample.sample_id)))
        selected: List[Tuple[BenchmarkSample, bool]] = [
            (sample, True) for sample in failures[: self.config.max_steps_per_round]
        ]

        remaining = max(0, self.config.quality_samples)
        if remaining:
            successes.sort(key=lambda sample: (-sample.score, str(sample.sample_id)))
            selected.extend((sample, False) for sample in successes[:remaining])

        return selected

    def _task_text(self, dataset: str, sample: BenchmarkSample) -> str:
        input_text = sample.input_text or "(no input text was captured)"
        max_chars = self.config.max_input_chars
        if max_chars and len(input_text) > max_chars:
            input_text = input_text[:max_chars] + "\n...[truncated]"

        header = f"Benchmark: {dataset}"
        if sample.subset and sample.subset != "main":
            header += f" (subset: {sample.subset})"
        return f"{header}\n\n{input_text}"

    def _sample_error(self, dataset: str, sample: BenchmarkSample) -> Optional[BenchmarkMismatch]:
        if sample.score >= float(self.config.failure_threshold):
            return None

        expected = ", ".join(sample.target) if sample.target else "(unknown)"
        actual = sample.prediction or "(empty response)"
        return BenchmarkMismatch(
            (
                f"Benchmark sample {sample.sample_id} in {dataset} was incorrect. "
                f"Expected: {expected}. Model answered: {actual}. "
                f"EvalScope score: {sample.score:.3f}."
            ),
            sample_id=sample.sample_id,
            dataset=dataset,
            expected=sample.target,
            actual=sample.prediction,
        )

    async def _learn(
        self,
        optimizer: Any,
        dataset: str,
        agent_name: str,
        round_number: int,
        current_prompt: str,
        sample: BenchmarkSample,
        is_failure: bool,
    ) -> Optional[str]:
        expected = ", ".join(sample.target) if sample.target else "(unknown)"
        observations = (
            f"Reference answer(s): {expected}\n"
            f"EvalScope score: {sample.score:.3f}\n"
            f"Benchmark subset: {sample.subset}"
        )
        try:
            result = await optimizer.on_step_complete(
                agent_name=agent_name,
                agent_role=f"Benchmark solver for {dataset}",
                task=self._task_text(dataset, sample),
                model_output=sample.prediction,
                observations=observations,
                error=self._sample_error(dataset, sample) if is_failure else None,
                current_system_prompt=current_prompt,
                task_id=f"{dataset}:round:{round_number}",
                truncate_context=False,
            )
        except Exception as exc:  # noqa: BLE001 - keep one bad sample from aborting a run
            logger.exception(
                "SCOPE synthesis failed for %s sample %s: %s",
                dataset,
                sample.sample_id,
                exc,
            )
            return None

        if not result:
            return None
        if isinstance(result, str):
            return result
        if isinstance(result, tuple) and result:
            return str(result[0])
        return str(result)

    async def _run_dataset(self, dataset: str) -> DatasetEvolutionResult:
        agent_name = f"{self.config.agent_name_prefix}-{_safe_name(dataset)}"
        dataset_dir = self.run_dir / "evalscope" / _safe_name(dataset)
        prompt_dir = self.run_dir / "prompts" / _safe_name(dataset)
        exp_path = self.run_dir / "artifacts" / "experiments" / _safe_name(agent_name)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        exp_path.mkdir(parents=True, exist_ok=True)

        optimizer = self.optimizer_factory(dataset, agent_name, str(exp_path))
        base_prompt = self._get_base_prompt(dataset)
        prompt_paths: List[str] = []
        rounds: List[Dict[str, Any]] = []

        baseline_prompt_path = prompt_dir / "round-00-baseline.txt"
        baseline_prompt_path.write_text(base_prompt, encoding="utf-8")
        prompt_paths.append(str(baseline_prompt_path))

        baseline = await self._run_phase(
            dataset=dataset,
            phase="baseline",
            prompt=base_prompt,
            limit=self.config.eval_limit,
            seed=self.config.eval_seed,
            work_dir=dataset_dir / "baseline",
        )
        rounds.append(
            {
                "round": 0,
                "phase": "baseline",
                "score": baseline.score,
                "sample_count": len(baseline.samples),
                "work_dir": str(baseline.work_dir),
            }
        )

        best_score = baseline.score
        best_round = 0
        stale_rounds = 0
        final_score = baseline.score

        for round_number in range(1, self.config.rounds + 1):
            strategic = ""
            try:
                strategic = optimizer.get_strategic_rules_for_agent(agent_name)
            except Exception as exc:  # noqa: BLE001 - optional optimizer feature
                logger.warning("Could not load strategic rules for %s: %s", agent_name, exc)

            round_prompt = _combine_prompts(base_prompt, strategic)
            train_prompt_path = prompt_dir / f"round-{round_number:02d}-train.txt"
            train_prompt_path.write_text(round_prompt, encoding="utf-8")
            prompt_paths.append(str(train_prompt_path))

            train = await self._run_phase(
                dataset=dataset,
                phase=f"round-{round_number:02d}-train",
                prompt=round_prompt,
                limit=self.config.train_limit,
                seed=self.config.train_seed,
                work_dir=dataset_dir / f"round-{round_number:02d}-train",
            )

            selected = self._select_samples(train.samples)
            learned: List[str] = []
            for sample, is_failure in selected:
                update = await self._learn(
                    optimizer=optimizer,
                    dataset=dataset,
                    agent_name=agent_name,
                    round_number=round_number,
                    current_prompt=round_prompt,
                    sample=sample,
                    is_failure=is_failure,
                )
                if update:
                    update = update.strip()
                    if update and update not in round_prompt:
                        round_prompt = _combine_prompts(
                            round_prompt, f"## Learned Guideline:\n{update}"
                        )
                        learned.append(update)

            eval_prompt_path = prompt_dir / f"round-{round_number:02d}-eval.txt"
            eval_prompt_path.write_text(round_prompt, encoding="utf-8")
            prompt_paths.append(str(eval_prompt_path))

            evaluation = await self._run_phase(
                dataset=dataset,
                phase=f"round-{round_number:02d}-eval",
                prompt=round_prompt,
                limit=self.config.eval_limit,
                seed=self.config.eval_seed,
                work_dir=dataset_dir / f"round-{round_number:02d}-eval",
            )
            final_score = evaluation.score

            rounds.append(
                {
                    "round": round_number,
                    "phase": "train",
                    "score": train.score,
                    "sample_count": len(train.samples),
                    "selected_samples": len(selected),
                    "failures": sum(1 for _, failure in selected if failure),
                    "learned_guidelines": learned,
                    "work_dir": str(train.work_dir),
                }
            )
            rounds.append(
                {
                    "round": round_number,
                    "phase": "eval",
                    "score": evaluation.score,
                    "sample_count": len(evaluation.samples),
                    "work_dir": str(evaluation.work_dir),
                }
            )

            if evaluation.score is not None:
                improved = best_score is None or evaluation.score > best_score + float(
                    self.config.min_improvement
                )
                if improved:
                    best_score = evaluation.score
                    best_round = round_number
                    stale_rounds = 0
                else:
                    stale_rounds += 1
                if self.config.patience and stale_rounds >= self.config.patience:
                    logger.info("Stopping %s after %s stale rounds", dataset, stale_rounds)
                    break

        result = DatasetEvolutionResult(
            dataset=dataset,
            agent_name=agent_name,
            baseline_score=baseline.score,
            final_score=final_score,
            best_score=best_score,
            best_round=best_round,
            prompt_paths=prompt_paths,
            rounds=rounds,
        )
        _write_json(dataset_dir / "evolution_summary.json", result.to_dict())
        return result

    async def run(self) -> Dict[str, Any]:
        """Run the complete evolution loop and return a JSON-serializable summary."""
        self._ensure_eval_scope()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for dataset in self.config.datasets:
            logger.info("Starting benchmark evolution for %s", dataset)
            results.append(await self._run_dataset(dataset))

        summary = {
            "provider": self.config.provider,
            "model": self.config.model,
            "model_id": self.config.model_id,
            "datasets": [result.to_dict() for result in results],
            "rounds": self.config.rounds,
            "train_limit": self.config.train_limit,
            "eval_limit": self.config.eval_limit,
            "train_seed": self.config.train_seed,
            "eval_seed": self.config.eval_seed,
        }
        _write_json(self.run_dir / "benchmark_summary.json", summary)
        return summary
