#!/usr/bin/env python3
"""Position-controlled case-012 study. prepare/publish/run/verify; no inference retries.

Run from the repository with .venv/bin/python analysis/position_sensitivity_case012/runner.py
all. Resumes finished calls; refuses to repeat an ambiguous interrupted call.
"""
import argparse
from collections import Counter
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from experiment_phoenix import oracle
from analyze_results import prompt_for
from evaluators import EVALUATORS, evaluate_all, ranked_eligible

BACKGROUND_SEEDS = (101, 203, 307, 409, 503)
SCHEDULE_SEED = 20261103
DATASET_NAME = 'tool-position-sensitivity-case012'
MODEL = 'qwen3.5:9b'
TARGET = 'R-9732'
CASE_ID = 'case-012'
ORIGINAL_PROJECT = 'tool-output-robustness-scaled'
PROTECTED = ('cases_scaled.json', 'scaled_results.json',
             'analysis_results.json', 'experiment_phoenix.py', 'scripts/phoenix_evaluators.py',
             'assets/phoenix_evidence.json')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    def encode(value):
        if dataclasses.is_dataclass(value): return dataclasses.asdict(value)
        if isinstance(value, datetime): return value.isoformat()
        raise TypeError(type(value))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=encode) + '\n')
    temp.replace(path)


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
def prompt_hash(messages): return sha_bytes(canonical(messages).encode('utf-8'))
def utcnow(): return datetime.now(timezone.utc)


def build_messages(raw_messages, records):
    payload = json.loads(raw_messages[1]['content'])
    payload['tool_response']['records'] = records
    return [dict(raw_messages[0]), {'role': 'user', 'content': canonical(payload)}]


def validate_examples(examples, source_case, raw_messages, canon_messages, control_payload):
    baseline = json.loads(raw_messages[1]['content'])
    baseline_records = baseline['tool_response']['records']
    record_multiset = sorted(canonical(x) for x in baseline_records)
    assert len(baseline_records) == 8 and len({x['id'] for x in baseline_records}) == 8
    assert oracle(source_case) == ranked_eligible(source_case)[0]['id'] == TARGET
    assert len(examples) == 46
    matches = {'RAW':0, 'CANONICAL':0}
    bg = {}
    for ex in examples:
        meta, inp = ex['metadata'], ex['input']
        messages = inp['messages']
        assert inp['case'] == source_case and ex['output'] == {'selected_id':TARGET}
        assert len(messages) == 2 and messages[0] == raw_messages[0]
        assert prompt_hash(messages) == meta['prompt_sha256']
        payload = json.loads(messages[1]['content'])
        records = payload['tool_response']['records']
        assert len(records) == 8 and sorted(canonical(r) for r in records) == record_multiset
        assert payload['request'] == baseline['request'] == source_case['request']
        assert payload['public_policy'] == baseline['public_policy']
        assert payload['public_policy']['constraints'] == source_case['constraints']
        assert payload['public_policy']['ranking'] == source_case['ranking']
        assert {k:v for k,v in payload['tool_response'].items() if k != 'records'} == control_payload
        assert payload.keys() == baseline.keys()
        assert [r['id'] for r in records].index(TARGET) + 1 == meta['target_position']
        # The template and every model-visible field apart from record order are exactly the original.
        without_records = dict(payload); without_records['tool_response'] = dict(payload['tool_response'])
        without_records['tool_response'].pop('records')
        baseline_without = dict(baseline); baseline_without['tool_response'] = dict(baseline['tool_response'])
        baseline_without['tool_response'].pop('records')
        assert canonical(without_records) == canonical(baseline_without)
        if meta['kind'] == 'position':
            assert meta['background_seed'] in BACKGROUND_SEEDS
            assert 1 <= meta['target_position'] <= 8
            bg.setdefault(meta['background_seed'], []).append((meta['target_position'],
                         [r['id'] for r in records if r['id'] != TARGET]))
        else:
            control = meta['control']
            assert messages == (raw_messages if control == 'RAW' else canon_messages)
            matches[control] += 1
    assert matches == {'RAW':3, 'CANONICAL':3}
    assert set(bg) == set(BACKGROUND_SEEDS)
    assert all(sorted(position for position, _ in rows) == list(range(1,9)) for rows in bg.values())
    assert all(len({tuple(order) for _,order in rows}) == 1 for rows in bg.values())
    assert len({tuple(rows[0][1]) for rows in bg.values()}) == 5
    return {'passed':True, 'examples':46, 'position_prompts':40, 'controls':matches,
            'candidate_records_and_metadata_identical':True, 'request_policy_tool_context_identical':True,
            'only_record_order_varies':True, 'all_target_positions_verified':True,
            'background_order_preserved_within_seed':True, 'oracle_verified':TARGET,
            'exact_historical_raw_and_canonical_messages_verified':True,
            'distinct_position_prompt_hashes':len({e['metadata']['prompt_sha256'] for e in examples if e['metadata']['kind']=='position'})}


def prepare(args, client):
    if (HERE / 'manifest.json').exists() or (HERE / 'raw_decisions.json').exists():
        raise RuntimeError('This output folder already has a study; do not overwrite it.')
    data, results = load(ROOT/'cases_scaled.json'), load(ROOT/'scaled_results.json')
    history_path = HERE/'mcp_historical_traces.json'
    if history_path.exists():
        history = load(history_path)
    else:
        # Use the current benchmark's verified trace joins rather than requiring
        # committed trace IDs from an older Phoenix instance.
        joins = {r['key']: r for r in load(ROOT/'assets/phoenix_evidence.json')['join_index']}
        history = {method: client.spans.get_spans(project_identifier=ORIGINAL_PROJECT,
            trace_ids=[joins[f'{CASE_ID}/{method}/original/moderate']['trace_id']],limit=10,timeout=30)
            for method in ('RAW','CANONICAL')}
        save(history_path,history)
    case = next(c for c in data['cases'] if c['id'] == CASE_ID)
    model = __import__('ollama').Client(host=args.ollama_url, timeout=30)
    info = model.show(MODEL).model_dump()
    assert not info.get('remote_host') and not info.get('remote_model') and 'thinking' in info.get('capabilities', [])
    installed = next(m for m in model.list().models if m.model == MODEL)
    assert installed.digest == results['model_digest'], 'Installed model digest differs from the original benchmark.'
    assert data['options']['temperature'] == 0 and data['options']['top_k'] == 1 and data['options']['seed'] == 20261025
    historical = {}
    prompts = {}
    for method in ('RAW','CANONICAL'):
        spans = history[method]
        p = next(s for s in spans if s['name'] == 'tool_output_decision')
        child = next(s for s in spans if s['span_kind'] == 'LLM' and s['parent_id'] == p['context']['span_id'])
        live = client.spans.get_spans(project_identifier=ORIGINAL_PROJECT,trace_ids=[p['context']['trace_id']],limit=10,timeout=30)
        liveparent = next(s for s in live if s['context']['span_id'] == p['context']['span_id'])
        livechild = next(s for s in live if s['context']['span_id'] == child['context']['span_id'])
        messages = json.loads(p['attributes']['input.value'])
        expected = prompt_for(data,case,method,'original','moderate')
        assert messages == expected == json.loads(liveparent['attributes']['input.value'])
        for i,m in enumerate(messages):
            assert child['attributes'][f'llm.input_messages.{i}.message.role'] == m['role']
            assert child['attributes'][f'llm.input_messages.{i}.message.content'] == m['content']
            assert livechild['attributes'][f'llm.input_messages.{i}.message.content'] == m['content']
        selected = p['attributes']['output.value']
        assert selected == results['observations'][f'{CASE_ID}/{method}/original/moderate']['selected']
        assert p['attributes']['experiment.oracle'] == TARGET
        prompts[method] = messages
        historical[method] = dict(trace_id=p['context']['trace_id'], parent_span_id=p['context']['span_id'],
                                  child_span_id=child['context']['span_id'], selected_id=selected,
                                  prompt_sha256=prompt_hash(messages))
    raw, can = prompts['RAW'], prompts['CANONICAL']
    raw_payload = json.loads(raw[1]['content'])
    original_records = raw_payload['tool_response']['records']
    assert [r['id'] for r in original_records].index(TARGET)+1 == 4
    assert [r['id'] for r in json.loads(can[1]['content'])['tool_response']['records']].index(TARGET)+1 == 8
    target = next(r for r in original_records if r['id'] == TARGET)
    rest = [r for r in original_records if r['id'] != TARGET]
    examples=[]; backgrounds={}
    def add(kind, identifier, position, records, seed=None, control=None, repetition=None):
        messages = build_messages(raw,records)
        meta = dict(trial_id=identifier,kind=kind,target_position=position,background_seed=seed,
                    control=control,control_repetition=repetition,prompt_sha256=prompt_hash(messages))
        examples.append({'input':{'case':case,'messages':messages},'output':{'selected_id':TARGET},'metadata':meta})
    for seed in BACKGROUND_SEEDS:
        background = rest.copy(); random.Random(seed).shuffle(background)
        backgrounds[str(seed)] = [r['id'] for r in background]
        for position in range(1,9):
            records = background.copy(); records.insert(position-1,target)
            add('position',f'seed-{seed}-position-{position}',position,records,seed=seed)
    canon_records = json.loads(can[1]['content'])['tool_response']['records']
    for method, records, position in [('RAW',original_records,4),('CANONICAL',canon_records,8)]:
        for repetition in range(1,4):
            add('control',f'control-{method}-{repetition}',position,records,control=method,repetition=repetition)
    context={k:v for k,v in raw_payload['tool_response'].items() if k!='records'}
    checks=validate_examples(examples,case,raw,can,context)
    rng=random.Random(SCHEDULE_SEED);schedule=[e['metadata']['trial_id'] for e in examples];rng.shuffle(schedule)
    assert len(set(schedule)) == len(schedule) == 46
    config = dict(dataset_name=args.dataset_name, case_id=CASE_ID, model=MODEL, model_digest=installed.digest,
        options=data['options'],think=False,system_prompt=data['system_prompt'],schema=data['schema'],
        background_seeds=list(BACKGROUND_SEEDS),background_orders=backgrounds,schedule_seed=SCHEDULE_SEED,
        total_planned_model_calls=46,position_calls=40,control_calls=6,execution_retries=0,
        planned_dataset_examples=46,noise='moderate',historical_controls=historical,
        protected_sha256={n:sha_file(ROOT/n) for n in PROTECTED},
        source_sha256={'runner.py':sha_file(__file__),'evaluators.py':sha_file(SOURCE/'evaluators.py')},
        packages={n:importlib.metadata.version(n) for n in ('arize-phoenix','arize-phoenix-client','ollama','openinference-instrumentation-ollama')})
    save(HERE/'frozen_dataset.json',{'examples':examples,'background_orders':backgrounds})
    save(HERE/'configuration.json',config);save(HERE/'schedule.json',schedule)
    save(HERE/'preflight_checks.json',checks)
    (HERE/'runner_at_execution.py').write_bytes(Path(__file__).read_bytes())
    if HERE != SOURCE:
        (HERE/'evaluators.py').write_bytes((SOURCE/'evaluators.py').read_bytes())
    print('Preflight passed: 40 position prompts + 3 RAW + 3 CANONICAL; metadata, context, policy, oracle, historical parent/child inputs, model digest and randomized schedule verified.',flush=True)


def publish(args, client):
    config=load(HERE/'configuration.json'); examples=load(HERE/'frozen_dataset.json')['examples']
    manifest_path=HERE/'manifest.json'
    manifest=load(manifest_path) if manifest_path.exists() else {'created_at':utcnow().isoformat()}
    existing=[d for d in client.datasets.list() if d['name']==config['dataset_name']]
    if 'dataset_id' not in manifest:
        if existing: raise RuntimeError('Dataset name already exists; refuse to reuse or overwrite it.')
        ds=client.datasets.create_dataset(name=config['dataset_name'],examples=examples,
            dataset_description='Case-012 position study: 40 controlled prompts and six exact historical controls. One synthetic scenario.',timeout=60)
        manifest.update(dataset_id=ds.id,dataset_version_id=ds.version_id); save(manifest_path,manifest)
    ds=client.datasets.get_dataset(dataset=manifest['dataset_id'],version_id=manifest['dataset_version_id'],timeout=60)
    actual={e['metadata']['trial_id']:e for e in ds.examples}
    assert len(actual)==46 and set(actual)=={e['metadata']['trial_id'] for e in examples}
    for ex in examples:
        actual_ex=actual[ex['metadata']['trial_id']]
        assert actual_ex['input']==ex['input'] and actual_ex['output']==ex['output'] and actual_ex['metadata']==ex['metadata']
    save(HERE/'phoenix_dataset.json',ds.to_dict())
    if 'experiment' not in manifest:
        manifest['experiment']=client.experiments.create(dataset_id=ds.id,dataset_version_id=ds.version_id,
            experiment_name='case012-position-sensitivity-46-calls',
            experiment_description='40 shuffled-background position prompts and six historical controls; one case; serial randomized execution.',
            experiment_metadata=config,repetitions=1,timeout=60)
        manifest['experiment_url']=client.experiments.get_experiment_url(ds.id,manifest['experiment']['id'])
        manifest['dataset_url']=client.experiments.get_dataset_experiments_url(ds.id)
        save(manifest_path,manifest)
    print('Phoenix dataset:',ds.id,'version:',ds.version_id,'experiment:',manifest['experiment']['id'],flush=True)


def run(args, client):
    import ollama
    from phoenix.otel import register
    from openinference.instrumentation.ollama import OllamaInstrumentor
    config, manifest = load(HERE/'configuration.json'), load(HERE/'manifest.json')
    assert sha_file(__file__)==config['source_sha256']['runner.py'] and sha_file(HERE/'evaluators.py')==config['source_sha256']['evaluators.py']
    model=ollama.Client(host=args.ollama_url,timeout=180)
    info=model.show(MODEL).model_dump(); assert not info.get('remote_host') and not info.get('remote_model')
    assert next(m for m in model.list().models if m.model==MODEL).digest==config['model_digest']
    ds=client.datasets.get_dataset(dataset=manifest['dataset_id'],version_id=manifest['dataset_version_id'],timeout=60)
    by_trial={e['metadata']['trial_id']:e for e in ds.examples}
    local={e['metadata']['trial_id']:e for e in load(HERE/'frozen_dataset.json')['examples']}
    assert set(by_trial)==set(local) and len(by_trial)==46
    for trial_id, ex in by_trial.items():
        assert ex['input']==local[trial_id]['input'] and ex['output']==local[trial_id]['output'] and ex['metadata']==local[trial_id]['metadata']
    schedule=load(HERE/'schedule.json'); ledger_path=HERE/'raw_decisions.json'
    ledger=load(ledger_path) if ledger_path.exists() else []
    assert len(ledger)<=46 and all(ledger[i]['trial_id']==schedule[i] for i in range(len(ledger)))
    if any(r['status']=='started' for r in ledger):
        raise RuntimeError('An inference may have occurred without a saved result; inspect the ledger and Phoenix. No automatic retry.')
    provider=register(endpoint=args.phoenix_url+'/v1/traces',protocol='http/protobuf',
        project_name=manifest['experiment']['project_name'],batch=False,set_global_tracer_provider=False,auto_instrument=False,verbose=False)
    instrumentor=OllamaInstrumentor(); instrumentor.instrument(tracer_provider=provider)
    tracer=provider.get_tracer('case012-position-sensitivity')
    for index in range(len(ledger),46):
        trial_id=schedule[index]; ex=by_trial[trial_id]; meta=ex['metadata']; prompt=ex['input']['messages']
        assert prompt_hash(prompt)==meta['prompt_sha256']
        started=utcnow()
        with tracer.start_as_current_span('case012_position_decision') as span:
            span.set_attribute('openinference.span.kind','CHAIN')
            span.set_attribute('input.value',json.dumps(prompt,ensure_ascii=True))
            span.set_attribute('experiment.id',manifest['experiment']['id'])
            for k in ('trial_id','kind','target_position','prompt_sha256'):
                span.set_attribute('position_study.'+k,meta[k])
            for k in ('background_seed','control','control_repetition'):
                if meta.get(k) is not None: span.set_attribute('position_study.'+k,meta[k])
            context=span.get_span_context()
            row=dict(sequence=index,trial_id=trial_id,dataset_example_id=ex['id'],metadata=meta,
                start_time=started.isoformat(),status='started',trace_id=f'{context.trace_id:032x}',
                parent_span_id=f'{context.span_id:016x}')
            ledger.append(row);save(ledger_path,ledger)
            clock=time.perf_counter(); raw=selected=error=None
            try:
                reply=model.chat(model=MODEL,messages=prompt,format=config['schema'],options=config['options'],
                                 think=False,stream=False,keep_alive='10m')
                raw=reply.message.content; result=json.loads(raw)
                if not isinstance(result,dict) or set(result)!={'selected_id'} or not isinstance(result['selected_id'],str):
                    raise ValueError('Output violates the specified response schema')
                selected=result['selected_id']
                row['prompt_tokens']=reply.prompt_eval_count;row['completion_tokens']=reply.eval_count
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}';span.record_exception(exc)
            output={'selected_id':selected,'raw_response':raw,'error':error}
            span.set_attribute('output.value',json.dumps(output,ensure_ascii=True))
            row.update(output=output,end_time=utcnow().isoformat(),latency_seconds=time.perf_counter()-clock,
                       local_evaluations=evaluate_all(ex['input'],output,ex['output']),status='inferred')
        provider.force_flush();save(ledger_path,ledger)
        print(f"{index+1}/46 {trial_id} -> {selected} {error or ''}",flush=True)
    instrumentor.uninstrument();provider.shutdown()
    # Persist recorded outputs without repeating inference if an upload needs to be resumed.
    experiment_id=manifest['experiment']['id']
    previous=client.experiments.get_experiment(experiment_id=experiment_id)
    existing={r['dataset_example_id']:r for r in previous['task_runs']}
    for row in ledger:
        if row['dataset_example_id'] in existing:
            found=existing[row['dataset_example_id']]
            assert found['output']==row['output'] and found['trace_id']==row['trace_id']
            row['experiment_run_id']=found['id']
        else:
            found=client.experiments.log_run(experiment_id=experiment_id,dataset_example_id=row['dataset_example_id'],
                output=row['output'],start_time=datetime.fromisoformat(row['start_time']),
                end_time=datetime.fromisoformat(row['end_time']),trace_id=row['trace_id'],timeout=60)
            row['experiment_run_id']=found['id']
        save(ledger_path,ledger)
    ran=client.experiments.get_experiment(experiment_id=experiment_id)
    assert len(ran['task_runs']) == 46
    if not ran['evaluation_runs']:
        client.experiments.evaluate_experiment(experiment=ran,evaluators=EVALUATORS,print_summary=False,retries=0,timeout=60)
    elif len(ran['evaluation_runs']) < 138:
        client.experiments.resume_evaluation(experiment_id=experiment_id,evaluators=EVALUATORS,print_summary=False,retries=0,timeout=60)
    identifier='case012-position:'+config['source_sha256']['evaluators.py'][:12]
    annotations=[]
    for row in ledger:
        for name,result in row['local_evaluations'].items():
            annotations.append(dict(name=name,span_id=row['parent_span_id'],annotator_kind='CODE',
                result={k:result[k] for k in ('score','label','explanation')},
                metadata={'source_sha256':config['source_sha256']['evaluators.py'],**result['metadata']},identifier=identifier))
    client.spans.log_span_annotations(span_annotations=annotations,sync=True)
    manifest['span_annotation_identifier']=identifier;save(HERE/'manifest.json',manifest)
    print('Recorded 46 attempted inferences and uploaded 138 CODE span annotations and native experiment evaluations.',flush=True)


def verify(args,client):
    config,manifest=load(HERE/'configuration.json'),load(HERE/'manifest.json')
    frozen=load(HERE/'frozen_dataset.json')['examples']; ledger=load(HERE/'raw_decisions.json')
    assert len(ledger)==46 and all(r['status']=='inferred' for r in ledger)
    assert [r['trial_id'] for r in ledger]==load(HERE/'schedule.json')
    ds=client.datasets.get_dataset(dataset=manifest['dataset_id'],version_id=manifest['dataset_version_id'],timeout=60)
    assert len(ds.examples)==46
    examples={e['id']:e for e in ds.examples}; local={e['metadata']['trial_id']:e for e in frozen}
    for ex in ds.examples:
        original=local[ex['metadata']['trial_id']]
        assert ex['input']==original['input'] and ex['output']==original['output'] and ex['metadata']==original['metadata']
    ran=client.experiments.get_experiment(experiment_id=manifest['experiment']['id'])
    assert len(ran['task_runs'])==46 and len(ran['evaluation_runs'])==138
    runs={r['id']:r for r in ran['task_runs']}
    evals=[dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x for x in ran['evaluation_runs']]
    evmap={(r['experiment_run_id'],r['name']):r for r in evals}
    annotations=client.spans.get_span_annotations(span_ids=[r['parent_span_id'] for r in ledger],
        project_identifier=manifest['experiment']['project_name'],limit=1000,timeout=60)
    annotations=[a for a in annotations if a.get('identifier')==manifest['span_annotation_identifier']]
    assert len(annotations)==138
    annotation_map={(a['span_id'],a['name']):a for a in annotations}
    spans=client.spans.get_spans(project_identifier=manifest['experiment']['project_name'],
                                 trace_ids=[r['trace_id'] for r in ledger],limit=1000,timeout=60)
    spans_by_trace={}
    for span in spans: spans_by_trace.setdefault(span['context']['trace_id'],[]).append(span)
    evidence=[]
    for row in ledger:
        ex=examples[row['dataset_example_id']]; run=runs[row['experiment_run_id']]
        assert run['trace_id']==row['trace_id'] and run['output']==row['output']
        trace=spans_by_trace[row['trace_id']]
        parent=next(s for s in trace if s['context']['span_id']==row['parent_span_id'])
        children=[s for s in trace if s['parent_id']==row['parent_span_id'] and s['span_kind']=='LLM']
        assert len(children)==1
        child=children[0]; messages=ex['input']['messages']
        assert json.loads(parent['attributes']['input.value'])==messages
        assert json.loads(parent['attributes']['output.value'])==row['output']
        for i,msg in enumerate(messages):
            assert child['attributes'][f'llm.input_messages.{i}.message.role']==msg['role']
            assert child['attributes'][f'llm.input_messages.{i}.message.content']==msg['content']
        if row['output']['raw_response'] is not None:
            assert child['attributes']['llm.output_messages.0.message.content']==row['output']['raw_response']
        computed=evaluate_all(ex['input'],row['output'],ex['output'])
        assert computed==row['local_evaluations']
        ids={}
        for name,value in computed.items():
            ev=evmap[(run['id'],name)]; ann=annotation_map[(row['parent_span_id'],name)]
            assert ev['annotator_kind']=='CODE' and not ev.get('error') and ann['annotator_kind']=='CODE'
            assert all((ev.get('result') or {}).get(k)==value[k] and ann['result'].get(k)==value[k]
                       for k in ('score','label','explanation'))
            ids[name]=ann['id']
        evidence.append({**row,'child_span_id':child['context']['span_id'],'evaluations':computed,'span_annotation_ids':ids})
    # Historical configuration still records retired reports. The frozen study
    # data and evaluator code remain protected; editable presentation is excluded.
    protected={n:sha_file(ROOT/n)==h for n,h in config['protected_sha256'].items()
               if not n.endswith('.html')}
    assert all(protected.values()),protected
    save(HERE/'native_experiment.json',{**ran,'evaluation_runs':evals})
    save(HERE/'evidence.json',dict(dataset_id=ds.id,dataset_version_id=ds.version_id,
        experiment_id=manifest['experiment']['id'],experiment_url=manifest['experiment_url'],project_name=manifest['experiment']['project_name'],
        planned_calls=46,inference_attempts=len(ledger),model_response_errors=sum(bool(r['output']['error']) for r in ledger),
        verified_llm_child_spans=len(evidence),native_evaluations_verified=len(evals),span_annotations_verified=len(annotations),
        source_and_original_artifacts_unchanged=protected,decisions=evidence))
    print('Verified 46 dataset examples, experiment runs, parent/LLM trace pairs, 138 native evaluations, 138 span annotations, and unchanged source artifacts.',flush=True)


def main():
    global HERE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','publish','run','verify','all'])
    parser.add_argument('--dataset-name',default=DATASET_NAME)
    parser.add_argument('--out',type=Path,default=SOURCE,
                        help='Output directory for a separate execution; existing studies are protected.')
    parser.add_argument('--phoenix-url',default='http://127.0.0.1:6006')
    parser.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    args=parser.parse_args()
    HERE=args.out.resolve()
    from phoenix.client import Client
    client=Client(base_url=args.phoenix_url)
    for name in (['prepare','publish','run','verify'] if args.command=='all' else [args.command]):
        globals()[name](args,client)


if __name__=='__main__': main()
