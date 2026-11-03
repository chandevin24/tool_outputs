#!/usr/bin/env python3
"""Audit the frozen experiment without making model calls. Standard library only."""

import collections
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runner", ROOT / "experiment_phoenix.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def prompt_for(dataset, case, method, order, noise):
    response = runner.prepare(
        case, dataset["tool_responses"][case["id"]][noise],
        method, order, noise, dataset["seed"],
    )
    content = json.dumps(
        dict(request=case["request"], public_policy=dict(
            constraints=case["constraints"], ranking=case["ranking"],
            constraint_semantics=dataset["constraint_semantics"],
        ), tool_response=response),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return [{"role": "system", "content": dataset["system_prompt"]},
            {"role": "user", "content": content}]


def classify(case, selected):
    records = {record["id"]: record for record in case["records"]}
    if selected not in records:
        return "invalid_selection"
    record = records[selected]
    if not all(runner.matches(record[f], op, target)
               for f, op, target in case["constraints"]):
        return "hard_constraint_violation"
    if selected != runner.oracle(case):
        return "eligible_wrong_ranking_or_tiebreak"
    return "correct"


def reference_oracle(case):
    """Separately replay the declared policy to cross-check the runner oracle."""
    operators = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
                 "==": lambda a, b: a == b, "contains": lambda a, b: b in a}
    eligible = [r for r in case["records"] if all(
        operators[op](r[field], target) for field, op, target in case["constraints"])]
    assert len(eligible) >= 2
    def key(record):
        priorities = tuple(record[field] * (1 if direction == "asc" else -1)
                           for field, direction in case["ranking"])
        return priorities, record["id"]
    return min(eligible, key=key)["id"]


def analyze():
    dataset = json.loads((ROOT / "cases_scaled.json").read_text())
    results = json.loads((ROOT / "scaled_results.json").read_text())
    cases, observations = results["cases"], results["observations"]
    assert cases == dataset["cases"], "Result cases differ from the frozen dataset"
    expected = {"/".join((case["id"], *condition)) for case in cases
                for condition in itertools.product(runner.METHODS, runner.ORDERS, runner.NOISES)}
    assert set(observations) == expected, "Missing or extra experiment conditions"
    assert results["status"] == "complete"

    # Recreate the runner's signature to verify the assumed local model setting.
    settings = dict(model="qwen3.5:9b", options=dataset["options"], cases=cases,
                    system_prompt=dataset["system_prompt"], schema=dataset["schema"],
                    tool_responses={c["id"]: dataset["tool_responses"][c["id"]] for c in cases})
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    assert signature == results["signature"], "Run signature mismatch"
    metrics = runner.summary(cases, observations)
    assert metrics == results["metrics"], "Saved aggregate metrics disagree with observations"

    by_id = {case["id"]: case for case in cases}
    for case in cases:
        assert reference_oracle(case) == runner.oracle(case), case["id"]
    taxonomy, breakdown, stable, prompt_counts = {}, {}, {}, {}
    for method in runner.METHODS:
        rows = [r for r in observations.values() if r["method"] == method]
        taxonomy[method] = dict(collections.Counter(
            classify(by_id[r["case"]], r["selected"]) for r in rows))
        breakdown[method] = {}
        for dimension in ("difficulty", "domain"):
            breakdown[method][dimension] = {}
            for label in sorted({c[dimension] for c in cases}):
                subset = [r for r in rows if by_id[r["case"]][dimension] == label]
                breakdown[method][dimension][label] = dict(
                    correct=sum(classify(by_id[r["case"]], r["selected"]) == "correct" for r in subset),
                    total=len(subset), scenarios=sum(c[dimension] == label for c in cases),
                    errors=dict(collections.Counter(classify(by_id[r["case"]], r["selected"])
                                                    for r in subset)),
                )
        stability = collections.Counter()
        unique = []
        for case in cases:
            subset = [r for r in rows if r["case"] == case["id"]]
            picks = {r["selected"] for r in subset}
            if len(picks) == 1:
                stability["stable_correct" if next(iter(picks)) == runner.oracle(case)
                          else "stable_wrong"] += 1
            else:
                stability["varying"] += 1
            unique.append(len({json.dumps(prompt_for(dataset, case, method, order, noise), sort_keys=True)
                               for order, noise in itertools.product(runner.ORDERS, runner.NOISES)}))
        stable[method] = dict(stability)
        prompt_counts[method] = dict(total=sum(unique), per_scenario=sorted(set(unique)))

    for row in observations.values():
        case = by_id[row["case"]]
        assert row["oracle"] == runner.oracle(case)
        assert row["correct"] == (row["selected"] == runner.oracle(case))
        assert row["domain"] == case["domain"]

    # Match the runner exactly: paired scenario means, 2,000 resamples,
    # one seeded RNG shared by the two method comparisons, indices 50 and 1949.
    rng, intervals = random.Random(2026), {}
    for method in runner.METHODS[1:]:
        deltas = []
        for case in cases:
            def accuracy(m):
                return sum(r["correct"] for r in observations.values()
                           if r["case"] == case["id"] and r["method"] == m) / 9
            deltas.append(accuracy(method) - accuracy("RAW"))
        boot = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
        intervals[method] = dict(difference_pp=100 * sum(deltas) / len(deltas),
                                 lower_pp=100 * boot[50], upper_pp=100 * boot[1949],
                                 scenarios_better=sum(d > 0 for d in deltas),
                                 scenarios_equal=sum(d == 0 for d in deltas),
                                 scenarios_worse=sum(d < 0 for d in deltas))

    case_studies = {}
    for cid in ("case-012", "case-032", "case-036"):
        case = by_id[cid]
        case_studies[cid] = dict(case=case, oracle=runner.oracle(case), selections={
            method: dict(collections.Counter(r["selected"] for r in observations.values()
                                             if r["case"] == cid and r["method"] == method))
            for method in runner.METHODS
        })

    return dict(
        provenance="Recomputed from frozen dataset and complete scaled results; no model calls.",
        input_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ("experiment_phoenix.py", "cases_scaled.json", "scaled_results.json")},
        signature=signature, model_digest=results["model_digest"], options=dataset["options"],
        cases=len(cases), observations=len(observations), metrics=metrics,
        bootstrap=dict(seed=2026, resamples=2000, unit="scenario", intervals=intervals),
        error_taxonomy=taxonomy, breakdown=breakdown, scenario_stability=stable,
        unique_prompts=prompt_counts, case_studies=case_studies,
    )


if __name__ == "__main__":
    output = ROOT / "analysis_results.json"
    data = analyze()
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: data[k] for k in ("metrics", "bootstrap", "error_taxonomy",
                                          "scenario_stability", "unique_prompts")}, indent=2))
    print(f"Saved {output.name}; all completeness, signature, oracle and metric checks passed.")
