#!/usr/bin/env python3
"""Export allowlisted synthetic trace evidence using the installed Phoenix client.

Run with .venv/bin/python. No Phoenix writes, model calls, or credentials exported.
"""

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path

from analyze_results import prompt_for

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "tool-output-robustness-scaled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    args = parser.parse_args()
    from phoenix.client import Client
    client = Client(base_url=args.base_url)
    frame = client.spans.get_spans_dataframe(
        project_identifier=PROJECT, limit=10000, timeout=30,
    )
    if len(frame) >= 10000:
        raise RuntimeError("Export reached limit; increase it before claiming complete coverage")
    dataset = json.loads((ROOT / "cases_scaled.json").read_text())
    results = json.loads((ROOT / "scaled_results.json").read_text())
    cases = {c["id"]: c for c in results["cases"]}
    candidates, children = defaultdict(list), defaultdict(list)
    for _, row in frame.iterrows():
        if row["name"] == "tool_output_decision":
            exp = row["attributes.experiment"]
            key = "/".join(exp[k] for k in ("case_id", "method", "order", "noise"))
            candidates[key].append(row)
        else:
            children[row["parent_id"]].append(row)

    exported, join_index = [], []
    for key, observation in results["observations"].items():
        matches = sorted(candidates[key], key=lambda r: r["start_time"], reverse=True)
        if not matches:
            raise RuntimeError(f"No Phoenix parent for {key}")
        # Pilot runs share keys for the first four cases. Select the latest
        # parent, then verify it against the complete run rather than guessing.
        parent = matches[0]
        exp = parent["attributes.experiment"]
        assert parent["attributes.output.value"] == observation["selected"], key
        assert exp["oracle"] == observation["oracle"], key
        evaluation = parent["attributes.evaluation"]
        assert evaluation["correct"] == observation["correct"], key
        prompt = json.loads(parent["attributes.input.value"])
        assert prompt == prompt_for(dataset, cases[observation["case"]],
                                    observation["method"], observation["order"], observation["noise"]), key
        child_rows = [r for r in children[parent["context.span_id"]] if r["span_kind"] == "LLM"]
        assert len(child_rows) == 1, key
        child = child_rows[0]
        output = child["attributes.llm.output_messages"][0]["message.content"]
        assert json.loads(output) == {"selected_id": observation["selected"]}, key
        child_messages = child["attributes.llm.input_messages"]
        assert [{"role": m["message.role"], "content": m["message.content"]}
                for m in child_messages] == prompt, key
        join_index.append(dict(key=key, trace_id=parent["context.trace_id"],
                               parent_span_id=parent["context.span_id"],
                               child_span_id=child["context.span_id"],
                               start_time=parent["start_time"].isoformat(),
                               selected=observation["selected"], correct=observation["correct"]))
        if observation["case"] not in ("case-012", "case-032", "case-036"):
            continue
        exported.append(dict(
            **join_index[-1], experiment={k: exp[k] for k in (
                "model", "oracle", "difficulty", "duration_s", "order",
                "domain", "noise", "method", "case_id")},
            messages=prompt, output=output,
            parent_name=parent["name"], child_name=child["name"],
            evaluation=evaluation, embedded_evaluations=parent["attributes.evaluations"],
            child_start_time=child["start_time"].isoformat(),
            child_end_time=child["end_time"].isoformat(),
        ))

    # In this installation CODE evaluations are embedded span attributes.
    # Query the annotations endpoint separately; do not imply it contains rows.
    annotations = client.spans.get_span_annotations_dataframe(
        span_ids=[r["parent_span_id"] for r in exported], project_identifier=PROJECT,
    )
    data = dict(
        project=PROJECT,
        client_version=importlib.metadata.version("arize-phoenix-client"),
        phoenix_version=importlib.metadata.version("arize-phoenix"),
        source_span_count=len(frame), source_names=dict(Counter(frame["name"])),
        matched_decisions=len(join_index), excluded_older_parent_spans=sum(len(v) for v in candidates.values()) - len(join_index),
        separate_annotation_rows=len(annotations),
        sanitization="Allowlisted experiment fields, synthetic prompts/outputs, span IDs and timestamps only; no headers, secrets or arbitrary attributes.",
        join_rule="Latest parent per case/method/order/noise; exact prompt, oracle, selection, correctness and child input/output validated against the complete run.",
        join_index=join_index, case_study_spans=sorted(exported, key=lambda r: r["key"]),
    )
    path = ROOT / "assets" / "phoenix_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {path.relative_to(ROOT)}: {len(join_index)} joined decisions, "
          f"{len(exported)} case-study parent/child pairs, {len(annotations)} separate annotation rows.")


if __name__ == "__main__":
    main()
