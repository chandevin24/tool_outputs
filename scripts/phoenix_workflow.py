#!/usr/bin/env python3
"""Publish and verify a small Phoenix-native regression study, separately from the original.

Commands: prepare (local checks), publish (annotations + dataset + empty experiments),
run (at most 24 Ollama calls), verify (read-only Phoenix read-back), replay (offline).
Native create/log_run enables strict serial interleaving across experiments; native
evaluate_experiment runs the reusable CODE evaluators. No inference retries.
"""
import argparse
from collections import Counter
import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_phoenix import prepare as old_prepare, oracle
from analyze_results import prompt_for
from phoenix_evaluators import EVALUATORS, evaluate_all, ranked_eligible

METHODS = ("RAW", "CANONICAL", "FILTER", "FILTER_NO_SORT")
CASES = ("case-012", "case-032", "case-036")
PROJECT = "tool-output-robustness-scaled"
SEED = 20261102


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    def encode(value):
        if dataclasses.is_dataclass(value): return dataclasses.asdict(value)
        if isinstance(value,datetime): return value.isoformat()
        raise TypeError(f"Cannot serialize {type(value)}")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=encode) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc)


def transformed(case, response, method, order, seed):
    # Use the original implementation unchanged for the three historical arms.
    if method in METHODS[:3]:
        return old_prepare(case, response, method, order, "moderate", seed)
    if method != "FILTER_NO_SORT":
        raise ValueError(method)
    ordered = old_prepare(case, response, "RAW", order, "moderate", seed)
    return {"records": [{field: r[field] for field in case["fields"]}
                        for r in ordered["records"]]}


def messages(dataset, input, method):
    case = input["case"]
    response = transformed(case, dataset["tool_responses"][case["id"]][input["noise"]],
                           method, input["order"], dataset["seed"])
    content = json.dumps(dict(request=case["request"], public_policy=dict(
        constraints=case["constraints"], ranking=case["ranking"],
        constraint_semantics=dataset["constraint_semantics"]), tool_response=response),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return [{"role": "system", "content": dataset["system_prompt"]},
            {"role": "user", "content": content}]


def original_rows():
    dataset, results, evidence = (load(ROOT / f) for f in
                                ("cases_scaled.json", "scaled_results.json", "assets/phoenix_evidence.json"))
    cases = {c["id"]: c for c in dataset["cases"]}
    joins = {j["key"]: j for j in evidence["join_index"]}
    assert len(joins) == len(results["observations"]) == 1296
    rows = []
    for key, row in results["observations"].items():
        case = cases[row["case"]]
        expected = {"selected_id": row["oracle"]}
        assert ranked_eligible(case)[0]["id"] == oracle(case) == row["oracle"]
        input = {"case": case, "order": row["order"], "noise": row["noise"]}
        output = {"selected_id": row["selected"]}
        evals = evaluate_all(input, output, expected)
        assert evals["exact_match_correctness"]["score"] == float(row["correct"])
        rows.append(dict(key=key, input=input, output=output, expected=expected,
                         evaluations=evals, **{k: joins[key][k] for k in
                         ("trace_id", "parent_span_id", "child_span_id")}))
    return dataset, rows


def prepare(args):
    dataset, rows = original_rows()
    source = {r["key"]: r for r in rows}
    examples = []
    reasons = {"case-012": "FILTER ranking error; RAW successful",
               "case-032": "RAW and FILTER hard-constraint violations",
               "case-036": "RAW ranking error; FILTER successful"}
    for order in ("original", "reversed"):
        for cid in CASES:
            row = source[f"{cid}/RAW/{order}/moderate"]
            examples.append(dict(input=row["input"], output=row["expected"], metadata=dict(
                case_id=cid, reason=reasons[cid], source_span_id=row["parent_span_id"],
                source_trace_id=row["trace_id"], source_key=row["key"])))
    # Guard the new arm: filtering preserves order, sorting preserves task facts.
    for example in examples:
        inp, case = example["input"], example["input"]["case"]
        prompts = {m: json.loads(messages(dataset, inp, m)[1]["content"])["tool_response"] for m in METHODS}
        assert [r["id"] for r in prompts["RAW"]["records"]] == [r["id"] for r in prompts["FILTER_NO_SORT"]["records"]]
        assert sorted(prompts["FILTER_NO_SORT"]["records"], key=lambda r:r["id"]) == prompts["FILTER"]["records"]
        for method in METHODS[:3]:
            assert messages(dataset, inp, method) == prompt_for(dataset, case, method, inp["order"], inp["noise"])
    # Meaningful edge contracts: invalid selection, ineligible ranking, eligible rank error, ties.
    inp = source["case-032/FILTER/original/moderate"]["input"]
    assert evaluate_all(inp, {"selected_id":"R-8327"}, {"selected_id":"R-3309"})["ranking_correctness"]["score"] is None
    assert evaluate_all(inp, {"selected_id":"INVALID"}, {"selected_id":"R-3309"})["hard_constraint_compliance"]["score"] == 0
    assert evaluate_all(inp, {"selected_id":"R-5546"}, {"selected_id":"R-3309"})["ranking_correctness"]["label"] == "fail"
    save(args.out / "examples.json", examples)
    save(args.out / "historical_evaluations.json", [{k:v for k,v in row.items() if k != "input"} for row in rows])
    # New executions must be self-contained after a fresh checkout. Original
    # evidence directories can continue to use their historical audit snapshot.
    audit_path = args.out / "audit_before.json"
    if not audit_path.exists():
        save(audit_path, {"preserved_sha256": {n: digest(ROOT/n) for n in (
            "cases_scaled.json", "scaled_results.json", "analysis_results.json",
            "experiment_phoenix.py", "assets/phoenix_evidence.json")}})
    config = dict(study_name=args.out.name, original_project=PROJECT, model="qwen3.5:9b",
                  options=dataset["options"], system_prompt=dataset["system_prompt"], schema=dataset["schema"],
                  methods=METHODS, cases=CASES, orders=["original","reversed"], noise="moderate",
                  schedule_seed=SEED, repetitions=1, planned_new_inferences=24, inference_retries=0,
                  design="2x2: keep/remove metadata crossed with preserve/sort record order; two presentation orders per case",
                  selection="Purposively selected historical successes and failures; not a random or held-out sample.",
                  source_sha256={n:digest(ROOT/n) for n in ("cases_scaled.json","scaled_results.json","experiment_phoenix.py","scripts/phoenix_evaluators.py","scripts/phoenix_workflow.py")},
                  sdk_versions={n:importlib.metadata.version(n) for n in
                                ("arize-phoenix","arize-phoenix-client","ollama","openinference-instrumentation-ollama")})
    if (args.out/"config.json").exists():
        assert load(args.out/"config.json") == json.loads(json.dumps(config)), "Configuration drift: choose a new output directory"
    else:
        save(args.out/"config.json",config)
    print("Prepared six examples and independently replayed three evaluators over 1,296 frozen decisions; no inference.",flush=True)


def span_annotation(row, name, evaluation, identifier, source_hash):
    return dict(name=name, span_id=row["parent_span_id"], annotator_kind="CODE",
                result={k:evaluation[k] for k in ("score","label","explanation")},
                metadata={"source_sha256":source_hash, **evaluation["metadata"]}, identifier=identifier)


def publish(args, client):
    config, examples = load(args.out/"config.json"), load(args.out/"examples.json")
    dataset, rows = original_rows()
    manifest_path=args.out/"manifest.json"
    manifest=load(manifest_path) if manifest_path.exists() else {"experiments":{},"created_at":now().isoformat()}
    identifier="tool-policy-v1:"+config["source_sha256"]["scripts/phoenix_evaluators.py"][:12]
    manifest["annotation_identifier"]=identifier
    if not manifest.get("historical_annotations_published"):
        frame=client.spans.get_spans_dataframe(project_identifier=PROJECT,limit=10000,timeout=30)
        assert len(frame)<10000
        by_span={r["context.span_id"]:r for _,r in frame.iterrows()}
        for row in rows:
            p,ch=by_span[row["parent_span_id"]],by_span[row["child_span_id"]]
            obs=load_observation_key(row["key"])
            assert p["attributes.output.value"]==obs["selected"]
            assert json.loads(p["attributes.input.value"])==prompt_for(dataset,row["input"]["case"],obs["method"],obs["order"],obs["noise"])
            assert ch["parent_id"]==row["parent_span_id"]
            assert json.loads(ch["attributes.llm.output_messages"][0]["message.content"])==row["output"]
        annotations=[span_annotation(r,n,e,identifier,config["source_sha256"]["scripts/phoenix_evaluators.py"])
                     for r in rows for n,e in r["evaluations"].items()]
        for start in range(0,len(annotations),300):
            client.spans.log_span_annotations(span_annotations=annotations[start:start+300],sync=True)
        manifest["historical_annotations_published"]=len(annotations)
        save(manifest_path,manifest)
    # Real dataset versions: v1 contains three original-order cases; v2 adds reversed-order cases.
    def upload_examples(examples):
        # JSON upload preserves nested candidate records and native source-span links.
        return [{**e, "span_id":e["metadata"]["source_span_id"]} for e in examples]
    if not manifest.get("dataset_id"):
        ds=client.datasets.create_dataset(name=config["study_name"], examples=upload_examples(examples[:3]),
            dataset_description="Diagnostic regression cases from observed ranking failures, constraint violations, and successes. v2 adds reversed record order.")
        manifest.update(dataset_id=ds.id,dataset_v1_id=ds.version_id)
        save(manifest_path,manifest)
    if not manifest.get("dataset_v2_id"):
        ds=client.datasets.add_examples_to_dataset(dataset=manifest["dataset_id"],examples=upload_examples(examples[3:]))
        manifest["dataset_v2_id"]=ds.version_id
        save(manifest_path,manifest)
    ds=client.datasets.get_dataset(dataset=manifest["dataset_id"],version_id=manifest["dataset_v2_id"])
    assert len(ds)==6
    assert all(isinstance(e["input"]["case"],dict) for e in ds.examples)
    save(args.out/"phoenix_dataset.json",ds.to_dict())
    for method in METHODS:
        if method not in manifest["experiments"]:
            exp=client.experiments.create(dataset_id=ds.id,dataset_version_id=ds.version_id,
                experiment_name=config["study_name"]+"-"+method,
                experiment_description="Six paired diagnostic examples; serial interleaved inferences, one repetition. No significance claim.",
                experiment_metadata={**config,"method":method,"execution_api":"experiments.create + log_run (serial interleaving) + evaluate_experiment"})
            manifest["experiments"][method]=exp
            save(manifest_path,manifest)
    manifest["comparison_url"]=client.experiments.get_dataset_experiments_url(ds.id)
    save(manifest_path,manifest)
    print(json.dumps({"dataset_id":ds.id,"version_id":ds.version_id,"experiments":{m:e["id"] for m,e in manifest["experiments"].items()}},indent=2),flush=True)


_OBSERVATIONS = None
def load_observation_key(key):
    global _OBSERVATIONS
    if _OBSERVATIONS is None:
        _OBSERVATIONS=load(ROOT/"scaled_results.json")["observations"]
    return _OBSERVATIONS[key]


def run(args, client):
    import ollama
    from phoenix.otel import register
    from openinference.instrumentation.ollama import OllamaInstrumentor
    config, manifest=load(args.out/"config.json"),load(args.out/"manifest.json")
    ds=client.datasets.get_dataset(dataset=manifest["dataset_id"],version_id=manifest["dataset_v2_id"])
    source=load(ROOT/"cases_scaled.json")
    model=ollama.Client(host=args.ollama_url,timeout=180)
    info=model.show(config["model"]).model_dump()
    assert not info.get("remote_host") and not info.get("remote_model")
    installed=next(m for m in model.list().models if m.model==config["model"])
    if manifest.get("model_digest"):
        assert manifest["model_digest"]==installed.digest
    manifest["model_digest"]=installed.digest
    manifest["think"]=False if "thinking" in (info.get("capabilities") or []) else None
    save(args.out/"manifest.json",manifest)
    rng=random.Random(SEED)
    schedule=[]
    for ex in sorted(ds.examples,key=lambda e:(e["input"]["case"]["id"],e["input"]["order"])):
        methods=list(METHODS);rng.shuffle(methods)
        schedule.extend({"example_id":ex["id"],"case_id":ex["input"]["case"]["id"],"order":ex["input"]["order"],"method":m} for m in methods)
    save(args.out/"schedule.json",schedule)
    ledger_path=args.out/"inferences.json"
    ledger=load(ledger_path) if ledger_path.exists() else []
    by_example={e["id"]:e for e in ds.examples}
    providers={m:register(endpoint=args.phoenix_url+"/v1/traces",protocol="http/protobuf",
        project_name=e["project_name"],batch=False,set_global_tracer_provider=False,auto_instrument=False,verbose=False)
        for m,e in manifest["experiments"].items()}
    instrumentor=OllamaInstrumentor()
    for index,item in enumerate(schedule):
        previous=next((r for r in ledger if r["sequence"]==index),None)
        if previous and previous.get("status")=="started":
            raise RuntimeError("Uncertain previous inference; inspect ledger and Phoenix before resuming. No automatic retry.")
        if previous:
            continue
        example=by_example[item["example_id"]]
        prompt=messages(source,example["input"],item["method"])
        provider=providers[item["method"]]
        instrumentor.instrument(tracer_provider=provider)
        tracer=provider.get_tracer("tool-policy-diagnostic")
        started=now()
        row={**item,"sequence":index,"status":"started","start_time":started.isoformat()}
        ledger.append(row);save(ledger_path,ledger)
        with tracer.start_as_current_span("tool_output_diagnostic_decision") as span:
            ctx=span.get_span_context()
            row.update(trace_id=f"{ctx.trace_id:032x}",parent_span_id=f"{ctx.span_id:016x}")
            span.set_attribute("openinference.span.kind","CHAIN")
            span.set_attribute("input.value",json.dumps(prompt))
            for k,v in item.items(): span.set_attribute("diagnostic."+k,v)
            span.set_attribute("experiment.id",manifest["experiments"][item["method"]]["id"])
            raw,selected,error=None,None,None
            try:
                reply=model.chat(model=config["model"],messages=prompt,format=config["schema"],
                    options=config["options"],think=manifest["think"],stream=False,keep_alive="10m")
                raw=reply.message.content
                value=json.loads(raw)
                if not isinstance(value,dict) or set(value)!={"selected_id"} or not isinstance(value["selected_id"],str):
                    raise ValueError("Output violates selected_id schema")
                selected=value["selected_id"]
                row.update(prompt_tokens=reply.prompt_eval_count,completion_tokens=reply.eval_count)
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}";span.record_exception(exc)
            output={"selected_id":selected,"raw_response":raw,"error":error}
            span.set_attribute("output.value",json.dumps(output))
            row.update(output=output,status="inferred",end_time=now().isoformat(),
                       prompt_sha256=hashlib.sha256(json.dumps(prompt,sort_keys=True).encode()).hexdigest())
        instrumentor.uninstrument()
        provider.force_flush()
        save(ledger_path,ledger)
        print(f"{index+1}/24 {item['case_id']} {item['order']} {item['method']}: {selected} {error or ''}",flush=True)
    # Upload persisted outputs separately, so a network failure cannot trigger another inference.
    for row in ledger:
        if not row.get("experiment_run_id"):
            exp=manifest["experiments"][row["method"]]
            run=client.experiments.log_run(experiment_id=exp["id"],dataset_example_id=row["example_id"],
                output=row["output"],start_time=datetime.fromisoformat(row["start_time"]),
                end_time=datetime.fromisoformat(row["end_time"]),trace_id=row["trace_id"])
            row["experiment_run_id"]=run["id"];save(ledger_path,ledger)
    for method,exp in manifest["experiments"].items():
        ran=client.experiments.get_experiment(experiment_id=exp["id"])
        client.experiments.evaluate_experiment(experiment=ran,evaluators=EVALUATORS,print_summary=False,retries=0)
    for row in ledger:
        example=by_example[row["example_id"]]
        evaluations=evaluate_all(example["input"],row["output"],example["output"])
        annotations=[span_annotation(row,n,e,manifest["annotation_identifier"],
                     config["source_sha256"]["scripts/phoenix_evaluators.py"])
                     for n,e in evaluations.items()]
        client.spans.log_span_annotations(span_annotations=annotations,sync=True)
    for provider in providers.values(): provider.shutdown()
    print("24 planned inference attempts recorded; native experiment evaluations uploaded.",flush=True)


def verify(args,client):
    config,manifest=load(args.out/"config.json"),load(args.out/"manifest.json")
    ds=client.datasets.get_dataset(dataset=manifest["dataset_id"],version_id=manifest["dataset_v2_id"])
    examples={e["id"]:e for e in ds.examples}
    ledger=load(args.out/"inferences.json")
    assert len(ledger)==24 and all(r["status"]=="inferred" for r in ledger)
    identifier=manifest["annotation_identifier"]
    _,historical=original_rows()
    annotations=client.spans.get_span_annotations(span_ids=[r["parent_span_id"] for r in historical],
        project_identifier=PROJECT,limit=10000,timeout=30)
    annotations=[a for a in annotations if a.get("identifier")==identifier]
    lookup={(a["span_id"],a["name"]):a for a in annotations}
    assert len(lookup)==len(historical)*3==3888
    for row in historical:
        for name,ev in row["evaluations"].items():
            got=lookup[row["parent_span_id"],name]
            assert got["annotator_kind"]=="CODE"
            assert all(got["result"].get(k)==ev[k] for k in ("score","label","explanation"))
    compact=[];counts={};native=[]
    for method,exp in manifest["experiments"].items():
        ran=client.experiments.get_experiment(experiment_id=exp["id"])
        ran["evaluation_runs"]=[dataclasses.asdict(e) if dataclasses.is_dataclass(e) else e for e in ran["evaluation_runs"]]
        native.append(ran)
        assert len(ran["task_runs"])==6
        assert len(ran["evaluation_runs"])==18
        eval_lookup={(e["experiment_run_id"],e["name"]):e for e in ran["evaluation_runs"]}
        frame=client.spans.get_spans_dataframe(project_identifier=exp["project_name"],limit=1000,timeout=30)
        assert len(frame)<1000
        span_lookup={r["context.span_id"]:r for _,r in frame.iterrows()}
        for task in ran["task_runs"]:
            row=next(r for r in ledger if r["experiment_run_id"]==task["id"])
            example=examples[task["dataset_example_id"]]
            assert task["output"]==row["output"] and task["trace_id"]==row["trace_id"]
            p=span_lookup[row["parent_span_id"]]
            children=[r for _,r in frame.iterrows() if r["parent_id"]==row["parent_span_id"] and r["span_kind"]=="LLM"]
            assert len(children)==1
            child=children[0]
            assert child["attributes.llm.output_messages"][0]["message.content"]==row["output"]["raw_response"]
            prompt=messages(load(ROOT/"cases_scaled.json"),example["input"],method)
            assert json.loads(p["attributes.input.value"])==prompt
            assert [{"role":m["message.role"],"content":m["message.content"]} for m in child["attributes.llm.input_messages"]]==prompt
            expected=evaluate_all(example["input"],row["output"],example["output"])
            for name,value in expected.items():
                ev=eval_lookup[task["id"],name]
                assert not ev.get("error"),ev
                assert ev["annotator_kind"]=="CODE"
                assert all((ev.get("result") or {}).get(k)==value[k] for k in ("score","label","explanation")),ev
            span_evals=[span_annotation(row,n,e,identifier,config["source_sha256"]["scripts/phoenix_evaluators.py"]) for n,e in expected.items()]
            # Experiment evaluations and trace annotations are distinct Phoenix resources.
            retrieved=client.spans.get_span_annotations(span_ids=[row["parent_span_id"]],project_identifier=exp["project_name"],limit=100)
            retrieved=[a for a in retrieved if a.get("identifier")==identifier]
            assert len(retrieved)==3
            assert all(a["result"]==next(a2["result"] for a2 in span_evals if a2["name"]==a["name"]) for a in retrieved)
            compact.append(dict(method=method,case_id=row["case_id"],order=row["order"],example_id=task["dataset_example_id"],
                experiment_id=exp["id"],experiment_run_id=task["id"],trace_id=row["trace_id"],parent_span_id=row["parent_span_id"],
                child_span_id=child["context.span_id"],output=row["output"],oracle=example["output"]["selected_id"],
                evaluations=expected,span_annotation_ids={a["name"]:a["id"] for a in retrieved}))
        counts[method]={n:dict(pass_count=sum(r["evaluations"][n]["score"]==1 for r in compact if r["method"]==method),
                         scored=sum(r["evaluations"][n]["score"] is not None for r in compact if r["method"]==method),
                         labels=dict(Counter(r["evaluations"][n]["label"] for r in compact if r["method"]==method))) for n in EVALUATORS}
    save(args.out/"native_experiments.json",native)
    audit_path=args.out/"audit_before.json"
    audit=load(audit_path if audit_path.exists() else ROOT/"phoenix_workflow/audit_before.json")
    # Editorial files were retired during repository cleanup. Keep the original
    # audit intact, but check the frozen study inputs/results rather than reports.
    preserved={n:digest(ROOT/n)==h for n,h in audit["preserved_sha256"].items()
               if n.endswith(".json") or n == "experiment_phoenix.py"}
    assert all(preserved.values()),preserved
    save(args.out/"evidence.json",dict(verified_at=now().isoformat(),verifier_sha256=digest(__file__),dataset_id=ds.id,dataset_version_id=ds.version_id,
        dataset_versions=client.datasets.get_dataset_versions(dataset=ds.id),new_inference_attempts=len(ledger),
        inference_errors=sum(bool(r["output"].get("error")) for r in ledger),
        historical_span_annotations_verified=len(annotations),experiment_evaluations_verified=72,
        new_span_annotations_verified=72,preserved_files=preserved,summary=counts,decisions=compact))
    print(json.dumps(counts,indent=2),flush=True)


def replay(args):
    original_rows()
    evidence=load(args.out/"evidence.json")
    examples={e["id"]:e for e in load(args.out/"phoenix_dataset.json")["examples"]}
    for row in evidence["decisions"]:
        example=examples[row["example_id"]]
        assert evaluate_all(example["input"],row["output"],example["output"])==row["evaluations"]
    print("Offline replay passed: 1,296 original decisions and 24 diagnostic decisions; zero model calls.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","publish","run","verify","replay","all"])
    parser.add_argument("--out",type=Path,default=ROOT/"phoenix_workflow/tool-policy-diagnostic-20261102")
    parser.add_argument("--phoenix-url",default="http://127.0.0.1:6006")
    parser.add_argument("--ollama-url",default="http://127.0.0.1:11434")
    args=parser.parse_args()
    if args.command in ("prepare","all"):prepare(args)
    if args.command=="replay":replay(args)
    if args.command not in ("prepare","replay"):
        from phoenix.client import Client
        client=Client(base_url=args.phoenix_url)
        for command in (["publish","run","verify"] if args.command=="all" else [args.command]):
            globals()[command](args,client)


if __name__=="__main__":
    main()
