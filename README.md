# Run the tool-output experiments

Requires Python 3.12 and [Ollama](https://ollama.com/) running locally. The frozen input is `cases_scaled.json`: 48 scenarios with eight candidates each.

## Setup

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3.5:9b
```

If Ollama is not already running, start it with `ollama serve` in another terminal. Start Phoenix separately:

```sh
source .venv/bin/activate
phoenix serve
```

Phoenix opens at <http://127.0.0.1:6006>. Ollama uses <http://127.0.0.1:11434>.

## Run

From the repository root with the virtual environment activated:

```sh
# Validate all conditions without model calls.
python experiment_phoenix.py --cases 48 --dry-run

# Run 48 scenarios × 3 methods × 3 orders × 3 noise levels: 1,296 calls.
python experiment_phoenix.py --cases 48 --output scaled_results.json

# Resume an interrupted execution with the same dataset and model.
python experiment_phoenix.py --cases 48 --output scaled_results.json --resume

# Audit the saved decisions and compute derived statistics without inference.
python scripts/analyze_results.py
```

The three methods are RAW, CANONICAL (sort by ID), and FILTER (remove irrelevant metadata and sort). Model settings, the system prompt, and the output schema are fixed in the dataset. The runner records the installed model digest and checks it on resume.

The run writes `scaled_results.json`; the offline analysis writes `analysis_results.json`. Phoenix receives parent decision spans and child Ollama input/output spans in project `tool-output-robustness-scaled`.

Use a fresh output filename for each new execution. The analysis script expects `scaled_results.json`; the run command above will overwrite that file unless `--resume` is supplied. Reports, saved results, and local environments are excluded from Git.

## Run the Phoenix diagnostic

First finish the scaled benchmark and analysis above. Export its trace joins from the same Phoenix server:

```sh
python scripts/export_phoenix_traces.py

# Six diagnostic examples × four variants: 24 additional model calls.
sh scripts/reproduce_phoenix_workflow.sh all --out phoenix_workflow/my_diagnostic

# Recheck the saved evaluations without model calls or a server.
python scripts/phoenix_workflow.py replay --out phoenix_workflow/my_diagnostic
```

Use a unique `--out` directory for each execution. The diagnostic crosses metadata removal with sorting, adding FILTER_NO_SORT to the original three strategies. It creates a versioned Phoenix dataset, native experiment runs, and three CODE evaluations: exact-match correctness, hard-constraint compliance, and ranking among eligible candidates. The small dataset is for diagnosis, not representative accuracy measurement.

## Run the position follow-up

After the scaled run, analysis, and trace export, use the same Phoenix instance:

```sh
# Case 012: five background arrangements × eight positions, plus six controls.
# This makes 46 additional model calls.
python analysis/position_sensitivity_case012/runner.py all \
  --out analysis/my_position_run --dataset-name my-position-run

# Rebuild descriptive tables and charts from that execution. No inference.
python -m pip install matplotlib numpy
python analysis/position_sensitivity_case012/analyze.py --out analysis/my_position_run
```

Use a unique output directory and dataset name. The runner retrieves control prompts from the current benchmark's Phoenix traces, verifies the model digest, and preserves the exact inputs and execution schedule. Controls compare against the new benchmark's actual choices; reruns need not reproduce the historical article's outcomes. Existing study directories cannot be overwritten.

The diagnostic and position runners accept `--phoenix-url` and `--ollama-url`. The trace exporter accepts `--base-url`. Their `verify` commands inspect the existing server and refresh local evidence without inference:

```sh
python scripts/phoenix_workflow.py verify --out phoenix_workflow/my_diagnostic
python analysis/position_sensitivity_case012/runner.py verify --out analysis/my_position_run
```

Git includes only these instructions, the frozen scenario input, dependencies, and reproduction code. Generated HTML, datasets/results from new runs, trace exports, screenshots, historical evidence, and local environments remain ignored.
# tool_outputs
