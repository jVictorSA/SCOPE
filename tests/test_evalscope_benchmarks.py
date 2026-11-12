"""Tests for the optional EvalScope benchmark integration.

The runner is tested with fake EvalScope functions so these tests do not
download datasets, call a model API, or require EvalScope to be installed.
"""

import asyncio
import json
from pathlib import Path

import pytest

from scope.benchmarks import (
    BenchmarkMismatch,
    EvalScopeEvolutionRunner,
    EvolutionConfig,
)
from scope.benchmarks.evalscope_runner import (
    _extract_review_score,
    _safe_name,
    load_phase_samples,
    report_score,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _prediction(index: int, prompt: str = "system", answer: str = "A"):
    return {
        "index": index,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": answer},
        ],
        "model_output": {
            "choices": [{"message": {"content": answer}}],
        },
    }


def _review(index: int, score: float, target="A"):
    return {
        "index": index,
        "target": [target],
        "sample_score": {
            "score": {
                "value": {"accuracy": score},
                "main_score_name": "accuracy",
            },
            "sample_metadata": {"source": "unit-test"},
        },
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": "A"},
        ],
    }


class TestScoreExtraction:
    def test_extracts_nested_review_score(self):
        assert _extract_review_score(_review(0, 1.0)) == 1.0
        assert _extract_review_score(_review(0, 0.0)) == 0.0

    def test_uses_main_score_name(self):
        review = {
            "sample_score": {
                "score": {
                    "value": {"accuracy": 0.25, "f1": 1.0},
                    "main_score_name": "accuracy",
                }
            }
        }
        assert _extract_review_score(review) == 0.25

    def test_returns_none_without_numeric_score(self):
        assert _extract_review_score({"sample_score": {"score": {"value": {}}}}) is None


class TestReportScore:
    def test_report_dict_score(self):
        assert report_score({"score": 0.75}) == 0.75

    def test_primary_metric_from_metrics(self):
        report = {
            "metrics": [
                {"identity": {"name": "f1"}, "score": 0.2},
                {"identity": {"name": "accuracy"}, "score": 0.8},
            ],
            "primary_metric_identity": {"name": "accuracy"},
        }
        assert report_score(report) == 0.8

    def test_dataset_mapping_and_list(self):
        assert report_score({"gsm8k": {"score": 0.4}}) == 0.4
        assert report_score([{"metrics": [{"score": 0.6}]}]) == 0.6

    def test_missing_score(self):
        assert report_score({"metrics": []}) is None


class TestLoadPhaseSamples:
    def test_joins_predictions_and_reviews(self, tmp_path):
        phase = tmp_path / "phase"
        _write_jsonl(
            phase / "predictions" / "fake-model" / "gsm8k_main.jsonl",
            [_prediction(0, answer="8"), _prediction(1, answer="9")],
        )
        _write_jsonl(
            phase / "reviews" / "fake-model" / "gsm8k_main.jsonl",
            [_review(0, 1.0, target="8"), _review(1, 0.0, target="10")],
        )

        samples = load_phase_samples(phase, "gsm8k")

        assert len(samples) == 2
        assert samples[0].sample_id == 0
        assert samples[0].subset == "main"
        assert samples[0].prediction == "8"
        assert samples[0].target == ["8"]
        assert samples[0].score == 1.0
        assert samples[0].input_text == "Question 0"
        assert samples[0].metadata == {"source": "unit-test"}
        assert samples[1].score == 0.0
        assert samples[1].prediction == "9"

    def test_accepts_dataset_only_filenames_and_review_only(self, tmp_path):
        phase = tmp_path / "phase"
        _write_jsonl(
            phase / "reviews" / "fake-model" / "arc.jsonl",
            [_review(0, 1.0)],
        )

        samples = load_phase_samples(phase, "arc")

        assert len(samples) == 1
        assert samples[0].subset == "arc"
        assert samples[0].prediction == "A"
        assert samples[0].input_text == "Question 0"


def _fake_eval_scope(task_config):
    """Write a small deterministic set of EvalScope artifacts."""
    work_dir = Path(task_config["work_dir"])
    dataset = task_config["datasets"][0]
    prompt = task_config["dataset_args"][dataset].get("system_prompt", "")
    phase = work_dir.name

    if phase == "baseline":
        scores = [0.0]
    elif phase.endswith("-train"):
        scores = [0.0, 1.0]
    else:
        scores = [1.0]

    predictions = []
    reviews = []
    for index, score in enumerate(scores):
        predictions.append(_prediction(index, prompt=prompt, answer="A"))
        reviews.append(_review(index, score, target="A"))

    _write_jsonl(
        work_dir / "predictions" / "fake-model" / f"{dataset}_main.jsonl",
        predictions,
    )
    _write_jsonl(
        work_dir / "reviews" / "fake-model" / f"{dataset}_main.jsonl",
        reviews,
    )
    report = {
        "model_name": "fake-model",
        "dataset_name": dataset,
        "score": sum(scores) / len(scores),
        "metrics": [{"identity": {"name": "accuracy"}, "score": sum(scores) / len(scores)}],
        "primary_metric_identity": {"name": "accuracy"},
    }
    report_path = work_dir / "reports" / "fake-model" / f"{dataset}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {dataset: report}


class FakeOptimizer:
    def __init__(self):
        self.calls = []
        self.strategic_calls = 0

    def get_strategic_rules_for_agent(self, agent_name):
        self.strategic_calls += 1
        return "STRATEGIC RULE"

    async def on_step_complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["error"] is not None:
            return ("TACTICAL RULE", "tactical")
        return None


class TestEvolutionRunner:
    def test_round_loop_writes_prompts_and_summary(self, tmp_path):
        optimizers = []

        def optimizer_factory(dataset, agent_name, exp_path):
            optimizer = FakeOptimizer()
            optimizers.append((dataset, agent_name, exp_path, optimizer))
            return optimizer

        task_configs = []

        def task_config_factory(**kwargs):
            task_configs.append(kwargs)
            return kwargs

        config = EvolutionConfig(
            datasets=["gsm8k"],
            model="fake-model",
            provider="openai",
            eval_type="openai_api",
            rounds=2,
            train_limit=2,
            eval_limit=1,
            max_steps_per_round=1,
            quality_samples=0,
            base_prompt="EXTRA PROMPT",
            agent_name_prefix="test",
        )
        runner = EvalScopeEvolutionRunner(
            config=config,
            run_dir=tmp_path / "run",
            optimizer_factory=optimizer_factory,
            run_task_func=_fake_eval_scope,
            task_config_factory=task_config_factory,
            default_prompt_func=lambda dataset: "BENCHMARK SYSTEM",
        )

        summary = asyncio.run(runner.run())

        assert summary["datasets"][0]["baseline_score"] == 0.0
        assert summary["datasets"][0]["best_score"] == 1.0
        assert summary["datasets"][0]["best_round"] == 1
        assert summary["datasets"][0]["final_score"] == 1.0

        assert len(optimizers) == 1
        dataset, agent_name, exp_path, optimizer = optimizers[0]
        assert dataset == "gsm8k"
        assert agent_name == "test-gsm8k"
        assert Path(exp_path).exists()
        assert len(optimizer.calls) == 2
        assert all(isinstance(call["error"], BenchmarkMismatch) for call in optimizer.calls)
        assert optimizer.strategic_calls == 2

        run_dir = tmp_path / "run"
        assert (run_dir / "benchmark_summary.json").exists()
        assert (run_dir / "evalscope" / "gsm8k" / "baseline" / "phase_result.json").exists()
        assert (run_dir / "prompts" / "gsm8k" / "round-00-baseline.txt").read_text() == (
            "BENCHMARK SYSTEM\n\nEXTRA PROMPT"
        )

        round_one_eval = (run_dir / "prompts" / "gsm8k" / "round-01-eval.txt").read_text()
        assert "BENCHMARK SYSTEM" in round_one_eval
        assert "EXTRA PROMPT" in round_one_eval
        assert "STRATEGIC RULE" in round_one_eval
        assert "TACTICAL RULE" in round_one_eval

        # The first evaluation uses only the base prompt.  Later evaluations
        # include the learned tactical guideline.
        eval_configs = [cfg for cfg in task_configs if Path(cfg["work_dir"]).name.endswith("-eval")]
        assert len(eval_configs) == 2
        assert "BENCHMARK SYSTEM" in eval_configs[0]["dataset_args"]["gsm8k"]["system_prompt"]
        assert "EXTRA PROMPT" in eval_configs[0]["dataset_args"]["gsm8k"]["system_prompt"]
        assert "STRATEGIC RULE" in eval_configs[0]["dataset_args"]["gsm8k"]["system_prompt"]
        assert "TACTICAL RULE" in eval_configs[1]["dataset_args"]["gsm8k"]["system_prompt"]

    def test_zero_rounds_still_runs_baseline(self, tmp_path):
        config = EvolutionConfig(
            datasets=["gsm8k"],
            model="fake-model",
            rounds=0,
            train_limit=2,
            eval_limit=1,
        )
        runner = EvalScopeEvolutionRunner(
            config=config,
            run_dir=tmp_path / "run",
            optimizer_factory=lambda *args: FakeOptimizer(),
            run_task_func=_fake_eval_scope,
            task_config_factory=lambda **kwargs: kwargs,
            default_prompt_func=lambda dataset: "SYSTEM",
        )

        summary = asyncio.run(runner.run())

        assert summary["datasets"][0]["baseline_score"] == 0.0
        assert summary["datasets"][0]["rounds"][0]["phase"] == "baseline"
        assert len(summary["datasets"][0]["rounds"]) == 1


def test_config_validation():
    with pytest.raises(ValueError):
        EvolutionConfig(datasets=[])
    with pytest.raises(ValueError):
        EvolutionConfig(datasets=["gsm8k"], rounds=-1)
    with pytest.raises(ValueError):
        EvolutionConfig(datasets=["gsm8k"], failure_threshold=2.0)
    with pytest.raises(ValueError):
        EvolutionConfig(datasets=["gsm8k"], quality_analysis_frequency=0)
    with pytest.raises(ValueError):
        EvolutionConfig(datasets=["gsm8k"], eval_limit=-1)


def test_safe_name():
    assert _safe_name("gsm8k/arc") == "gsm8k-arc"
    assert _safe_name("") == "run"
