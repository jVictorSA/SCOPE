"""Tests for repository-level EvalScope benchmark suite files.

These tests are dependency-free: they do not import EvalScope, download
datasets, or call a model API.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scope.benchmarks.suites import (
    SUITE_ALLOWED_KEYS,
    SuiteError,
    get_suite,
    list_suites,
    load_suite_file,
    merge_json_dicts,
    suite_to_arg_defaults,
)


def _write_suite_file(tmp_path: Path, data) -> Path:
    path = tmp_path / "suites.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_file(tmp_path: Path, suites=None) -> Path:
    return _write_suite_file(
        tmp_path,
        {
            "version": 1,
            "suites": suites
            or {
                "smoke": {
                    "description": "A smoke suite",
                    "datasets": ["gsm8k"],
                    "train_limit": 5,
                    "eval_limit": 5,
                    "rounds": 1,
                    "dataset_args": {"gsm8k": {"few_shot_num": 0}},
                },
                "reasoning": {
                    "datasets": ["gsm8k", "arc"],
                    "dataset_args": {"arc": {"few_shot_num": 0}},
                },
            },
        },
    )


def test_load_valid_suite_file(tmp_path):
    path = _valid_file(tmp_path)
    data = load_suite_file(path)
    assert data["version"] == 1
    assert set(data["suites"]) == {"smoke", "reasoning"}
    assert data["suites"]["smoke"]["train_limit"] == 5


def test_repository_suite_file_is_valid():
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "suites.json"
    data = load_suite_file(path)
    assert "smoke" in data["suites"]
    assert data["suites"]["smoke"]["datasets"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"suites": {"smoke": {"datasets": ["gsm8k"]}}}, "version"),
        ({"version": 2, "suites": {"smoke": {"datasets": ["gsm8k"]}}}, "version"),
        ({"version": 1}, "suites"),
        ({"version": 1, "suites": {}}, "suites"),
        ({"version": 1, "extra": True, "suites": {"a": {"datasets": ["x"]}}}, "top-level"),
    ],
)
def test_invalid_top_level_structure(tmp_path, payload, message):
    path = _write_suite_file(tmp_path, payload)
    with pytest.raises(SuiteError, match=message):
        load_suite_file(path)


def test_missing_suite_file(tmp_path):
    with pytest.raises(SuiteError, match="does not exist"):
        load_suite_file(tmp_path / "missing.json")


def test_invalid_json(tmp_path):
    path = tmp_path / "suites.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SuiteError, match="invalid JSON"):
        load_suite_file(path)


@pytest.mark.parametrize(
    "suite, message",
    [
        ({}, "datasets"),
        ({"datasets": []}, "datasets"),
        ({"datasets": "gsm8k"}, "datasets"),
        ({"datasets": [""]}, "datasets"),
        ({"datasets": ["gsm8k"], "unknown": 1}, "unknown keys"),
        ({"datasets": ["gsm8k"], "dataset_args": []}, "dataset_args"),
        ({"datasets": ["gsm8k"], "ignore_errors": "yes"}, "ignore_errors"),
        ({"datasets": ["gsm8k"], "dataset_hub": "s3"}, "dataset_hub"),
        ({"datasets": ["gsm8k"], "auto_accept_threshold": "maybe"}, "auto_accept_threshold"),
        ({"datasets": ["gsm8k"], "rounds": -1}, "rounds"),
    ],
)
def test_invalid_suite(tmp_path, suite, message):
    path = _valid_file(tmp_path, {"broken": suite})
    with pytest.raises(SuiteError, match=message):
        load_suite_file(path)


def test_model_api_and_output_keys_are_rejected(tmp_path):
    for key, value in [
        ("provider", "deepseek"),
        ("model", "deepseek-chat"),
        ("api_url", "https://api.deepseek.com/v1"),
        ("api_key", "secret"),
        ("output_dir", "/tmp/results"),
        ("run_name", "run"),
    ]:
        path = _valid_file(tmp_path, {"because": {"datasets": ["gsm8k"], key: value}})
        with pytest.raises(SuiteError, match="unknown keys"):
            load_suite_file(path)


def test_duplicate_datasets_are_removed(tmp_path):
    path = _valid_file(tmp_path, {"dupes": {"datasets": ["gsm8k", "gsm8k", "arc"]}})
    data = load_suite_file(path)
    assert data["suites"]["dupes"]["datasets"] == ["gsm8k", "arc"]


def test_unknown_suite_name_lists_available(tmp_path):
    path = _valid_file(tmp_path)
    with pytest.raises(SuiteError, match="reasoning, smoke"):
        get_suite(path, "missing")


def test_get_suite_returns_a_copy(tmp_path):
    path = _valid_file(tmp_path)
    suite = get_suite(path, "smoke")
    suite["datasets"].append("arc")
    assert get_suite(path, "smoke")["datasets"] == ["gsm8k"]


def test_list_suites_sorted_with_descriptions(tmp_path):
    path = _valid_file(tmp_path)
    entries = list_suites(path)
    assert [entry["name"] for entry in entries] == ["reasoning", "smoke"]
    assert entries[1]["description"] == "A smoke suite"
    assert entries[0]["description"] == ""


def test_suite_to_arg_defaults_inverts_boolean_flags():
    suite = {
        "datasets": ["gsm8k"],
        "train_limit": 5,
        "ignore_errors": True,
        "enable_quality_analysis": False,
        "quality_analysis_frequency": 2,
        "dataset_args": {"gsm8k": {}},
    }
    defaults = suite_to_arg_defaults(suite)
    assert defaults["datasets"] == ["gsm8k"]
    assert defaults["train_limit"] == 5
    assert defaults["no_ignore_errors"] is False
    assert defaults["no_quality_analysis"] is True
    assert defaults["quality_frequency"] == 2
    assert "dataset_args" not in defaults


def test_suite_to_arg_defaults_empty():
    assert suite_to_arg_defaults({}) == {}


def test_merge_json_dicts_nested_override():
    base = {
        "gsm8k": {"few_shot_num": 0, "limit": 5},
        "generation": {"temperature": 0.0, "max_tokens": 512},
    }
    override = {
        "gsm8k": {"limit": 10},
        "generation": {"temperature": 0.7},
        "new": {"enabled": True},
    }
    merged = merge_json_dicts(base, override)
    assert merged == {
        "gsm8k": {"few_shot_num": 0, "limit": 10},
        "generation": {"temperature": 0.7, "max_tokens": 512},
        "new": {"enabled": True},
    }


def test_merge_json_dicts_replaces_non_dict_values():
    assert merge_json_dicts({"a": 1, "b": {"c": 2}}, {"a": "text", "b": 3}) == {
        "a": "text",
        "b": 3,
    }


def test_merge_json_dicts_copies_inputs():
    base = {"a": {"b": 1}}
    merged = merge_json_dicts(base, {"a": {"c": 2}})
    merged["a"]["b"] = 99
    assert base == {"a": {"b": 1}}


def test_allowed_keys_cover_plan_fields():
    expected = {
        "datasets",
        "train_limit",
        "eval_limit",
        "rounds",
        "dataset_args",
        "generation_config",
        "extra_task_kwargs",
    }
    assert expected <= SUITE_ALLOWED_KEYS
    assert not ({"provider", "model", "api_key", "output_dir"} & SUITE_ALLOWED_KEYS)


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "examples" / "evalscope_evolution.py"


def _run_example(*args, cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_list_suites():
    result = _run_example("--list-suites")
    assert result.returncode == 0, result.stderr
    assert "smoke" in result.stdout
    assert "reasoning" in result.stdout


def test_cli_unknown_suite_fails():
    result = _run_example(
        "--suite",
        "does-not-exist",
        "--provider",
        "local",
        "--model",
        "dummy",
        "--api-url",
        "http://127.0.0.1:8000/v1",
        "--dry-run",
    )
    assert result.returncode != 0
    assert "unknown suite" in result.stderr


def test_cli_dry_run_suite_and_overrides(tmp_path):
    result = _run_example(
        "--suite",
        "smoke",
        "--provider",
        "local",
        "--model",
        "dummy",
        "--api-url",
        "http://127.0.0.1:8000/v1",
        "--api-key",
        "SECRET123",
        "--rounds",
        "9",
        "--train-limit",
        "7",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    run_dirs = list((tmp_path / "evalscope_evolution").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    suite_artifact = json.loads((run_dir / "benchmark_suite.json").read_text())
    assert suite_artifact["name"] == "smoke"
    assert suite_artifact["config"]["datasets"] == ["gsm8k"]

    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    config = resolved["config"]
    assert config["datasets"] == ["gsm8k"]
    assert config["dataset_args"] == {"gsm8k": {"few_shot_num": 0}}
    assert config["rounds"] == 9
    assert config["train_limit"] == 7
    assert config["api_key"] == "***"
    assert "SECRET123" not in (run_dir / "resolved_config.json").read_text()
