"""Load and validate repository-level EvalScope benchmark suites.

Suite files are plain JSON so they can be reviewed without adding a YAML/TOML
dependency.  They contain benchmark and evolution settings only; model,
provider, API URL, API key, and output settings belong to the command line.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .evalscope_runner import EvolutionConfig

SUITE_FILE_VERSION = 1

SUITE_ALLOWED_KEYS = {
    "description",
    "datasets",
    "train_limit",
    "eval_limit",
    "train_seed",
    "eval_seed",
    "rounds",
    "max_steps_per_round",
    "quality_samples",
    "failure_threshold",
    "max_input_chars",
    "base_prompt",
    "dataset_args",
    "dataset_hub",
    "dataset_dir",
    "generation_config",
    "eval_batch_size",
    "ignore_errors",
    "extra_task_kwargs",
    "agent_name_prefix",
    "enable_quality_analysis",
    "quality_analysis_frequency",
    "auto_accept_threshold",
    "patience",
    "min_improvement",
}

DIRECT_ARG_DEFAULTS = {
    "datasets",
    "train_limit",
    "eval_limit",
    "train_seed",
    "eval_seed",
    "rounds",
    "max_steps_per_round",
    "quality_samples",
    "failure_threshold",
    "max_input_chars",
    "base_prompt",
    "dataset_hub",
    "dataset_dir",
    "eval_batch_size",
    "agent_name_prefix",
    "auto_accept_threshold",
    "patience",
    "min_improvement",
}

JSON_OBJECT_FIELDS = {
    "dataset_args",
    "generation_config",
    "extra_task_kwargs",
}


class SuiteError(ValueError):
    """Raised for malformed suite files or unknown suite names."""


def _error(path: Union[str, Path], message: str) -> SuiteError:
    return SuiteError(f"{path}: {message}")


def _validate_suite(path: Union[str, Path], name: str, suite: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(suite, dict):
        raise _error(path, f"suite '{name}' must be a JSON object")

    unknown = set(suite) - SUITE_ALLOWED_KEYS
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise _error(path, f"suite '{name}' has unknown keys: {joined}")

    description = suite.get("description")
    if description is not None and not isinstance(description, str):
        raise _error(path, f"suite '{name}' field 'description' must be a string")

    datasets = suite.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise _error(path, f"suite '{name}' field 'datasets' must be a non-empty list")
    normalized_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, str) or not dataset.strip():
            raise _error(
                path,
                f"suite '{name}' field 'datasets' must contain non-empty strings",
            )
        normalized = dataset.strip()
        if normalized not in normalized_datasets:
            normalized_datasets.append(normalized)

    for field_name in JSON_OBJECT_FIELDS:
        value = suite.get(field_name)
        if value is not None and not isinstance(value, dict):
            raise _error(path, f"suite '{name}' field '{field_name}' must be an object")

    for field_name in ("ignore_errors", "enable_quality_analysis"):
        value = suite.get(field_name)
        if value is not None and not isinstance(value, bool):
            raise _error(path, f"suite '{name}' field '{field_name}' must be a boolean")

    for field_name in ("base_prompt", "dataset_hub", "dataset_dir", "agent_name_prefix"):
        value = suite.get(field_name)
        if value is not None and not isinstance(value, str):
            raise _error(path, f"suite '{name}' field '{field_name}' must be a string")

    dataset_hub = suite.get("dataset_hub")
    if dataset_hub not in (None, "modelscope", "huggingface"):
        raise _error(
            path,
            f"suite '{name}' field 'dataset_hub' must be 'modelscope' or 'huggingface'",
        )

    threshold = suite.get("auto_accept_threshold")
    if threshold not in (None, "all", "low", "medium", "high"):
        raise _error(
            path,
            f"suite '{name}' field 'auto_accept_threshold' must be one of "
            "all, low, medium, high",
        )

    config_values = {
        key: copy.deepcopy(value)
        for key, value in suite.items()
        if key not in {"description", "datasets"} | JSON_OBJECT_FIELDS
    }
    try:
        EvolutionConfig(datasets=normalized_datasets, **config_values)
    except (TypeError, ValueError) as exc:
        raise _error(path, f"suite '{name}' is invalid: {exc}") from exc

    validated = copy.deepcopy(suite)
    validated["datasets"] = normalized_datasets
    return validated


def load_suite_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a suite manifest from JSON."""
    suite_path = Path(path).expanduser()
    try:
        with open(suite_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise _error(suite_path, "suite file does not exist") from exc
    except OSError as exc:
        raise _error(suite_path, f"could not read suite file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _error(suite_path, f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise _error(suite_path, "suite file root must be a JSON object")

    unknown_top_level = set(data) - {"version", "suites"}
    if unknown_top_level:
        joined = ", ".join(sorted(unknown_top_level))
        raise _error(suite_path, f"unknown top-level keys: {joined}")

    version = data.get("version")
    if type(version) is not int or version != SUITE_FILE_VERSION:
        raise _error(
            suite_path,
            f"version must be {SUITE_FILE_VERSION}",
        )

    suites = data.get("suites")
    if not isinstance(suites, dict) or not suites:
        raise _error(suite_path, "field 'suites' must be a non-empty object")

    validated_suites: Dict[str, Any] = {}
    for name, suite in suites.items():
        if not isinstance(name, str) or not name.strip():
            raise _error(suite_path, "suite names must be non-empty strings")
        normalized_name = name.strip()
        if normalized_name != name:
            raise _error(suite_path, f"suite name '{name}' must not have surrounding spaces")
        validated_suites[name] = _validate_suite(suite_path, name, suite)

    return {"version": version, "suites": validated_suites}


def list_suites(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Return suite names and descriptions in sorted order."""
    data = load_suite_file(path)
    entries = []
    for name in sorted(data["suites"]):
        suite = data["suites"][name]
        entries.append(
            {
                "name": name,
                "description": suite.get("description", ""),
            }
        )
    return entries


def get_suite(path: Union[str, Path], name: str) -> Dict[str, Any]:
    """Return one validated suite by name."""
    data = load_suite_file(path)
    suite = data["suites"].get(name)
    if suite is None:
        available = ", ".join(sorted(data["suites"]))
        raise SuiteError(f"unknown suite '{name}' in {path}. Available suites: {available}")
    return copy.deepcopy(suite)


def suite_to_arg_defaults(suite: Dict[str, Any]) -> Dict[str, Any]:
    """Map suite fields to argparse destination names and values."""
    defaults: Dict[str, Any] = {}
    for field_name in DIRECT_ARG_DEFAULTS:
        if field_name in suite:
            defaults[field_name] = copy.deepcopy(suite[field_name])

    if "quality_analysis_frequency" in suite:
        defaults["quality_frequency"] = suite["quality_analysis_frequency"]

    if "enable_quality_analysis" in suite:
        defaults["no_quality_analysis"] = not suite["enable_quality_analysis"]

    if "ignore_errors" in suite:
        defaults["no_ignore_errors"] = not suite["ignore_errors"]

    return defaults


def merge_json_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge JSON-style dictionaries; override values win."""
    result = copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return result

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_json_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


__all__ = [
    "SUITE_ALLOWED_KEYS",
    "SUITE_FILE_VERSION",
    "SuiteError",
    "get_suite",
    "list_suites",
    "load_suite_file",
    "merge_json_dicts",
    "suite_to_arg_defaults",
]
