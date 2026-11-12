"""Tests for the shared example output helper."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scope.models import BaseModelAdapter, Message, ModelResponse


def _load_output_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "_output.py"
    spec = importlib.util.spec_from_file_location("example_output", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


output = _load_output_module()


def _args(**overrides):
    values = {
        "temp": False,
        "output_dir": None,
        "run_name": None,
        "api_key": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_output_dir_argument_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv(output.OUTPUT_DIR_ENV, str(tmp_path / "env"))
    args = _args(output_dir=str(tmp_path / "cli"), run_name="first")

    with output.run_context("demo", args) as run:
        assert run.run_dir == tmp_path / "cli" / "demo" / "first"
        assert Path(run.exp_path) == run.run_dir
        assert (run.run_dir / "run_config.json").exists()
        assert (run.logs_dir / "run.log").exists()
        assert (run.artifacts_dir).is_dir()


def test_environment_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(output.OUTPUT_DIR_ENV, str(tmp_path / "env"))
    args = _args(run_name="env-run")

    with output.run_context("demo", args) as run:
        assert run.run_dir == tmp_path / "env" / "demo" / "env-run"


def test_default_output_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(output.OUTPUT_DIR_ENV, raising=False)
    monkeypatch.setattr(output, "DEFAULT_RESULTS_ROOT", tmp_path / "default")
    args = _args(run_name="default-run")

    with output.run_context("demo", args) as run:
        assert run.run_dir == tmp_path / "default" / "demo" / "default-run"


def test_run_name_collision_gets_suffix(tmp_path):
    args = _args(output_dir=str(tmp_path), run_name="same")

    with output.run_context("demo", args) as first:
        first_path = first.run_dir
    with output.run_context("demo", args) as second:
        second_path = second.run_dir

    assert first_path != second_path
    assert second_path.name == "same-2"


def test_api_keys_are_redacted(tmp_path):
    args = _args(
        output_dir=str(tmp_path),
        run_name="redacted",
        api_key="super-secret-key",
    )

    with output.run_context("demo", args) as run:
        config = json.loads((run.run_dir / "run_config.json").read_text())

    assert config["args"]["api_key"] == "***"
    assert "super-secret-key" not in json.dumps(config)


def test_argv_credentials_are_redacted(tmp_path, monkeypatch):
    args = _args(output_dir=str(tmp_path), run_name="argv")
    monkeypatch.setattr(
        output.sys,
        "argv",
        ["example.py", "--api-key", "super-secret-key", "--model=deepseek-chat"],
    )

    with output.run_context("demo", args) as run:
        config = json.loads((run.run_dir / "run_config.json").read_text())

    assert config["argv"] == [
        "example.py",
        "--api-key",
        "***",
        "--model=deepseek-chat",
    ]


def test_temp_run_is_cleaned_up(tmp_path):
    args = _args(temp=True)

    with output.run_context("demo", args) as run:
        run_dir = run.run_dir
        assert run_dir.is_dir()

    assert not run_dir.exists()


@pytest.mark.asyncio
async def test_logging_model_adapter_writes_jsonl(tmp_path):
    class FakeModel(BaseModelAdapter):
        async def generate(self, messages):
            return ModelResponse(content="reply")

    log_path = tmp_path / "logs" / "model_calls.jsonl"
    adapter = output.LoggingModelAdapter(FakeModel(), log_path)

    response = await adapter.generate([Message.user("hello")])

    assert response.content == "reply"
    entry = json.loads(log_path.read_text().strip())
    assert entry["messages"][0]["content"] == "hello"
    assert entry["response"] == "reply"
