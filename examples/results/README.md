# Example results

Running any script in `examples/` writes its results here by default:

```text
examples/results/<example-name>/<timestamp>/
```

Each run contains the generated SCOPE data, console log, model prompts and
responses, run configuration, and a summary. The EvalScope benchmark example
additionally stores per-phase predictions, reviews, reports, prompts, and
`benchmark_summary.json` under its run directory. See the "Saving Example
Results" and "Benchmark Evolution with EvalScope" sections in the main README
for details.

Use `--output-dir` or `SCOPE_EXAMPLE_OUTPUT_DIR` to write somewhere else, or
pass `--temp` to discard the results after the run. Generated run folders are
ignored by git.
