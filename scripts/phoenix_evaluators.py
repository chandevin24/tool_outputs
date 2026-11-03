"""Pure, deterministic Phoenix evaluators. No model, network, or Phoenix dependency.

The experiment adapter passes input/outputs/expected from the Phoenix dataset.
Ranking is intentionally unscored when selection is invalid or ineligible.
"""
from functools import cmp_to_key
import json

VERSION = "tool-policy-evaluators-v1"


def selected_id(output):
    value = output.get("selected_id") if isinstance(output, dict) else None
    return value if isinstance(value, str) else None


def holds(value, operator, target):
    if operator == "<=":
        return value <= target
    if operator == ">=":
        return value >= target
    if operator == "==":
        return value == target
    if operator == "contains":
        return target in value
    raise ValueError(f"Unsupported constraint operator: {operator}")


def violations(case, record):
    return [f"{field}={json.dumps(record[field])} violates {field} {op} {json.dumps(target)}"
            for field, op, target in case["constraints"]
            if not holds(record[field], op, target)]


def eligible_records(case):
    return [r for r in case["records"] if not violations(case, r)]


def ranked_eligible(case):
    def compare(a, b):
        for field, direction in case["ranking"]:
            if direction not in ("asc", "desc"):
                raise ValueError(f"Unsupported ranking direction: {direction}")
            comparison = (a[field] > b[field]) - (a[field] < b[field])
            if comparison:
                return comparison if direction == "asc" else -comparison
        return (a["id"] > b["id"]) - (a["id"] < b["id"])
    return sorted(eligible_records(case), key=cmp_to_key(compare))


def result(score, label, explanation):
    return dict(score=score, label=label, explanation=explanation,
                metadata={"evaluator_version": VERSION})


def exact_match_correctness(input, output, expected):
    """Exact candidate-ID match to the dataset's frozen reference answer."""
    selected, oracle = selected_id(output), expected["selected_id"]
    correct = selected == oracle
    return result(float(correct), "pass" if correct else "fail",
                  f"Selected {selected!r}; frozen oracle is {oracle!r}. "
                  + ("Exact match." if correct else "Candidate IDs differ."))


def hard_constraint_compliance(input, output, expected):
    """Check every declared hard constraint against trusted candidate records."""
    case, selected = input["case"], selected_id(output)
    record = next((r for r in case["records"] if r["id"] == selected), None)
    if record is None:
        return result(0.0, "fail", f"Selection {selected!r} is not a valid candidate ID.")
    failed = violations(case, record)
    if failed:
        return result(0.0, "fail", f"{selected}: " + "; ".join(failed))
    return result(1.0, "pass", f"{selected} satisfies all {len(case['constraints'])} hard constraints.")


def ranking_correctness(input, output, expected):
    """Check lexicographic priorities and ID tie-break, conditional on eligibility."""
    case, selected = input["case"], selected_id(output)
    ranked = ranked_eligible(case)
    if not ranked:
        raise ValueError("Policy has no eligible candidates; define an abstention contract first.")
    chosen = next((r for r in ranked if r["id"] == selected), None)
    if chosen is None:
        return result(None, "not_applicable",
                      f"Selection {selected!r} is invalid or ineligible; evaluate compliance first. "
                      "Excluded from the ranking-score denominator.")
    best = ranked[0]
    if selected == best["id"]:
        return result(1.0, "pass", f"{selected} ranks first among {len(ranked)} eligible candidates, including the ID tie-break.")
    for field, direction in case["ranking"]:
        if chosen[field] != best[field]:
            reason = f"First differing priority: {field} {direction}; selected={chosen[field]}, best={best[field]}."
            break
    else:
        reason = f"All ranking values tie; lexicographically smaller ID {best['id']} wins."
    return result(0.0, "fail", f"Selected eligible {selected}, but {best['id']} ranks first. {reason}")


EVALUATORS = {f.__name__: f for f in
              (exact_match_correctness, hard_constraint_compliance, ranking_correctness)}


def evaluate_all(input, output, expected):
    return {name: evaluator(input, output, expected) for name, evaluator in EVALUATORS.items()}
