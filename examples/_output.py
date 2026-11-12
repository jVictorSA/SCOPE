"""
Shared output handling for SCOPE examples.

Every example writes its results to a timestamped run directory so that
strategic memory, guideline history, logs, and model prompts/responses are
preserved for later inspection.

Default location:
    examples/results/<example-name>/<timestamp>/

Override the base directory with ``--output-dir`` or the
``SCOPE_EXAMPLE_OUTPUT_DIR`` environment variable. Pass ``--temp`` to restore
the old temporary-directory behavior.
"""
import contextlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scope.models import BaseModelAdapter, Message, ModelResponse

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_DIR_ENV = "SCOPE_EXAMPLE_OUTPUT_DIR"


def _safe_name(value: str) -> str:
    """Turn arbitrary user input into a safe single path component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or "run"


def _timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _jsonable(value: Any) -> Any:
    """Return a JSON-serializable version of ``value``."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _redact(value: Any) -> Any:
    """Redact values that look like credentials from config output."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            lowered = str(key).lower()
            sensitive = (
                lowered in {"api_key", "apikey", "token", "secret", "password"}
                or lowered.endswith(
                    ("_api_key", "_token", "_secret", "_password")
                )
            )
            if sensitive:
                redacted[key] = "***" if item else None
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return _jsonable(value)


def _redact_argv(argv: List[str]) -> List[str]:
    """Redact credential-looking command line arguments."""
    redacted = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue

        flag = arg.split("=", 1)[0].lower()
        sensitive = (
            flag in {"-k", "--api-key", "--apikey", "--token", "--password"}
            or flag.endswith(("-api-key", "_api_key", "-token", "_token"))
        )
        if sensitive:
            if "=" in arg:
                redacted.append(arg.split("=", 1)[0] + "=***")
            else:
                redacted.append(arg)
                redact_next = True
        else:
            redacted.append(arg)
    return redacted


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


class _Tee:
    """Write to the original stream and a log file at the same time."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


class LoggingModelAdapter(BaseModelAdapter):
    """Model adapter that records every prompt and response as JSONL."""

    def __init__(self, model, log_path):
        self.model = model
        self.log_path = Path(log_path)

    async def generate(self, messages: List[Message]) -> ModelResponse:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {
                    "role": msg.role,
                    "content": _jsonable(msg.content),
                    "tool_calls": _jsonable(msg.tool_calls),
                }
                for msg in messages
            ],
        }

        try:
            response = await self.model.generate(messages)
        except Exception as exc:
            entry["error"] = repr(exc)
            self._append(entry)
            raise

        entry["response"] = _jsonable(response.content)
        self._append(entry)
        return response

    def _append(self, entry: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str))
            handle.write("\n")


@dataclass
class RunContext:
    """Paths and helpers for one example run."""

    example: str
    run_dir: Path
    started_at: datetime
    _started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def exp_path(self) -> str:
        """Path passed to ``SCOPEOptimizer(exp_path=...)``."""
        return str(self.run_dir)

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    def experiment_dir(self, name: str) -> str:
        """Create and return an isolated exp_path for a sub-example."""
        path = self.artifacts_dir / "experiments" / _safe_name(name)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @contextlib.contextmanager
    def experiment(self, name: str):
        """Context manager yielding an isolated exp_path for a sub-example."""
        yield self.experiment_dir(name)

    def wrap_model(self, model):
        """Wrap a model so all prompts/responses are logged."""
        return LoggingModelAdapter(model, self.logs_dir / "model_calls.jsonl")

    def save_text(self, name: str, text: str) -> Path:
        """Save a text artifact in the run directory."""
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def save_json(self, name: str, data: Any) -> Path:
        """Save a JSON artifact in the run directory."""
        path = self.run_dir / name
        _write_json(path, data)
        return path

    def save_summary(self, data: Any, status: str = "ok") -> Path:
        """Write summary.json with run metadata and caller-provided data."""
        finished_at = datetime.now()
        summary = {
            "example": self.example,
            "status": status,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(
                time.monotonic() - self._started_monotonic, 3
            ),
            "result": data,
        }
        return self.save_json("summary.json", summary)


def _unique_dir(base: Path, name: str) -> Path:
    """Create a unique directory, adding a numeric suffix on collision."""
    path = base / name
    counter = 2
    while path.exists():
        path = base / f"{name}-{counter}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_run_config(run: RunContext, args, output_root: Optional[Path]) -> None:
    config = {
        "example": run.example,
        "run_dir": str(run.run_dir),
        "output_root": str(output_root) if output_root else None,
        "started_at": run.started_at.isoformat(),
        "argv": _redact_argv(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "args": _redact(vars(args)) if hasattr(args, "__dict__") else None,
    }
    _write_json(run.run_dir / "run_config.json", config)


def add_output_args(parser) -> None:
    """Add result-output arguments to an argparse parser."""
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=(
            "Base directory for example results "
            "(default: examples/results or $%s)" % OUTPUT_DIR_ENV
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run folder (default: timestamp)",
    )
    parser.add_argument(
        "--temp",
        action="store_true",
        help="Use a temporary directory instead of saving results",
    )


@contextlib.contextmanager
def run_context(example_name: str, args):
    """Create a results directory and capture console output for one run."""
    temp_dir = None
    output_root = None

    if getattr(args, "temp", False):
        temp_dir = tempfile.TemporaryDirectory(prefix=f"scope-{example_name}-")
        run_dir = Path(temp_dir.name)
    else:
        root = (
            getattr(args, "output_dir", None)
            or os.getenv(OUTPUT_DIR_ENV)
            or str(DEFAULT_RESULTS_ROOT)
        )
        output_root = Path(root).expanduser().resolve()
        base = output_root / _safe_name(example_name)
        base.mkdir(parents=True, exist_ok=True)
        run_name = getattr(args, "run_name", None) or _timestamp()
        run_dir = _unique_dir(base, _safe_name(run_name))

    run = RunContext(
        example=example_name,
        run_dir=run_dir,
        started_at=datetime.now(),
    )
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    run.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        _write_run_config(run, args, output_root)

    log_path = run.logs_dir / "run.log"
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            stdout = _Tee(sys.stdout, log_file)
            stderr = _Tee(sys.stderr, log_file)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                print(f"Results directory: {run.run_dir}")
                try:
                    yield run
                except BaseException as exc:
                    if temp_dir is None:
                        run.save_summary(
                            {"error": repr(exc), "type": type(exc).__name__},
                            status="error",
                        )
                    raise
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def save_run_summary(run: RunContext, data: Any, status: str = "ok") -> Path:
    """Convenience wrapper around ``RunContext.save_summary``."""
    return run.save_summary(data, status=status)
