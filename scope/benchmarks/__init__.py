"""Optional benchmark integrations for SCOPE.

The benchmark helpers are intentionally kept out of the core package imports so
that installing SCOPE does not require a benchmark framework.  EvalScope is
imported lazily by :mod:`scope.benchmarks.evalscope_runner`.
"""

from .evalscope_runner import (
    BenchmarkMismatch,
    BenchmarkSample,
    DatasetEvolutionResult,
    EvalScopeEvolutionRunner,
    EvolutionConfig,
    PhaseResult,
    get_default_system_prompt,
    load_phase_samples,
    report_score,
)
from .suites import (
    SuiteError,
    get_suite,
    list_suites,
    load_suite_file,
    merge_json_dicts,
    suite_to_arg_defaults,
)

__all__ = [
    "BenchmarkMismatch",
    "BenchmarkSample",
    "DatasetEvolutionResult",
    "EvalScopeEvolutionRunner",
    "EvolutionConfig",
    "PhaseResult",
    "SuiteError",
    "get_default_system_prompt",
    "get_suite",
    "list_suites",
    "load_phase_samples",
    "load_suite_file",
    "merge_json_dicts",
    "report_score",
    "suite_to_arg_defaults",
]
