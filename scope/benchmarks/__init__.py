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

__all__ = [
    "BenchmarkMismatch",
    "BenchmarkSample",
    "DatasetEvolutionResult",
    "EvalScopeEvolutionRunner",
    "EvolutionConfig",
    "PhaseResult",
    "get_default_system_prompt",
    "load_phase_samples",
    "report_score",
]
