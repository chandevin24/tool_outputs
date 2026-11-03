#!/usr/bin/env python3
"""Paired tool-output benchmark: local Ollama + optional Phoenix traces."""
import argparse
import copy
import hashlib
import itertools
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

METHODS = ('RAW', 'CANONICAL', 'FILTER')
ORDERS = ('original', 'reversed', 'shuffle')
NOISES = ('clean', 'moderate', 'heavy')


def matches(value, comparison, target):
    """Apply one of the dataset's supported constraint operators."""
    if comparison == '<=':
        return value <= target
    if comparison == '>=':
        return value >= target
    if comparison == '==':
        return value == target
    if comparison == 'contains':
        return target in value

    raise ValueError(f'Unknown comparison: {comparison}')


def oracle(case):
    """Find the lowest-ranked eligible candidate, breaking ties by ID."""
    ranked = []
    for record in case['records']:
        if all(matches(record[f], op, t) for f, op, t in case['constraints']):
            values = [
                record[field] if direction == 'asc' else -record[field]
                for field, direction in case['ranking']
            ]
            ranked.append((tuple(values), record['id']))
    assert len(ranked) >= 2, f"No meaningful alternatives in {case['id']}"
    return min(ranked)[1]


def prepare(case, saved_response, method, order, noise, seed):
    """Perturb presentation only, and ensure the answer is unchanged."""
    response = copy.deepcopy(saved_response)
    if order == 'reversed':
        response['records'].reverse()
    elif order == 'shuffle':
        random.Random(f'{seed}:order:{case["id"]}').shuffle(response['records'])
    # Keep only task-relevant fields; FILTER also removes response metadata.
    relevant = [
        {field: record[field] for field in case['fields']}
        for record in response['records']
    ]
    baseline = sorted(case['records'], key=lambda r: r['id'])
    assert sorted(relevant, key=lambda r: r['id']) == baseline

    # RAW preserves extra fields and record order; CANONICAL keeps all fields
    # but sorts records; FILTER keeps only relevant fields and sorts records.
    if method == 'FILTER':
        response = {'records': relevant}
    if method != 'RAW':
        response['records'].sort(key=lambda r: r['id'])
    return response


def summary(cases, observations):
    """Use matched observations; repeated conditions are not independent cases."""
    output = {}
    for method in METHODS:
        rows = [r for r in observations.values() if r['method'] == method]
        picks = {
            (row['case'], row['order'], row['noise']): row['selected']
            for row in rows
        }
        correct = sum(r['selected'] == r['oracle'] for r in rows)
        invalid = sum(r['selected'] is None for r in rows)
        # Order flips compare all record orders at fixed noise; noise flips
        # compare each noisy result with its clean counterpart.
        flips_order = total_order = flips_noise = total_noise = 0
        for case in cases:
            cid = case['id']
            for noise in NOISES:
                keys = [(cid, order, noise) for order in ORDERS]
                if all(k in picks for k in keys):
                    total_order += 1
                    flips_order += len({picks[k] for k in keys}) > 1
            for order in ORDERS:
                clean = (cid, order, 'clean')
                for noise in NOISES[1:]:
                    noisy = (cid, order, noise)
                    if clean in picks and noisy in picks:
                        total_noise += 1
                        flips_noise += picks[clean] != picks[noisy]
        output[method] = dict(
            accuracy=[correct, len(rows)],
            invalid=[invalid, len(rows)],
            order_flips=[flips_order, total_order],
            noise_flips=[flips_noise, total_noise],
        )
    return output


def report(results):
    """Print all rates and a paired case-bootstrap interval for each gain."""
    metrics = results['metrics']
    print(
        '\nMethod       Accuracy          Invalid           '
        'Order flips       Noise flips'
    )
    for method in METHODS:
        counts = metrics[method]
        cells = [
            f'{numerator}/{denominator} '
            f'({100 * numerator / denominator:.1f}%)'
            if denominator
            else 'n/a'
            for numerator, denominator in counts.values()
        ]
        print(f'{method:12} ' + '  '.join(f'{s:18}' for s in cells))
    observations = list(results['observations'].values())
    ids = [case['id'] for case in results['cases']]
    by_case = {}
    for cid in ids:
        by_case[cid] = {}
        for method in METHODS:
            rows = [
                row
                for row in observations
                if row['case'] == cid and row['method'] == method
            ]
            if len(rows) == 9:
                by_case[cid][method] = (
                    sum(r['selected'] == r['oracle'] for r in rows) / 9
                )
    if not all(len(v) == 3 for v in by_case.values()):
        print(
            'Paired uncertainty intervals will appear when all 27 '
            'conditions per case finish.'
        )
        return
    rng = random.Random(2026)
    for method in METHODS[1:]:
        deltas = [by_case[cid][method] - by_case[cid]['RAW'] for cid in ids]
        boot = sorted(
            sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
            for _ in range(2000)
        )
        mean_gain = 100 * sum(deltas) / len(deltas)
        lower_bound = 100 * boot[50]
        upper_bound = 100 * boot[1949]
        print(
            f'{method} - RAW paired accuracy gain: {mean_gain:+.1f} pp '
            f'(case-bootstrap 95% interval: '
            f'{lower_bound:+.1f} to {upper_bound:+.1f} pp)'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path(__file__).with_name('cases_scaled.json'),
    )
    parser.add_argument(
        '--cases',
        type=int,
        default=48,
        help='First N frozen cases; use 4 or 12 for a pilot',
    )
    parser.add_argument('--model', default='qwen3.5:9b')
    parser.add_argument('--output', type=Path, default=Path('scaled_results.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-phoenix', action='store_true')
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    cases = dataset['cases'][:args.cases]
    if args.cases < 1 or len(cases) != args.cases or ':cloud' in args.model:
        parser.error('Specify a valid case count and local Ollama model.')
    seed = dataset['seed']
    settings = {
        'model': args.model,
        'options': dataset['options'],
        'cases': cases,
        'system_prompt': dataset['system_prompt'],
        'schema': dataset['schema'],
        'tool_responses': {
            case['id']: dataset['tool_responses'][case['id']]
            for case in cases
        },
    }
    signature = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    results = dict(
        signature=signature,
        cases=cases,
        observations={},
        metrics={},
        status='running',
    )
    if args.resume and args.output.exists():
        results = json.loads(args.output.read_text())
        if results['signature'] != signature:
            parser.error(
                'Dataset/model/options differ from previous run. '
                'Use a new output.'
            )
    if not args.dry_run:
        import ollama
        client = ollama.Client(host='http://127.0.0.1:11434', timeout=180)
        info = client._request(
            dict,
            'POST',
            '/api/show',
            json={'model': args.model},
        )
        if info.get('remote_host') or info.get('remote_model'):
            parser.error('Only locally installed, non-cloud Ollama models are allowed.')
        think = False if 'thinking' in info.get('capabilities', []) else None
        # Save the installation digest to prevent mixing model versions on resume.
        local_models = client._request(dict, 'GET', '/api/tags')['models']
        local_model = next(
            (model for model in local_models if model['model'] == args.model),
            None,
        )
        if local_model is None:
            parser.error('The model must be installed locally.')
        if (
            results.get('model_digest')
            and results['model_digest'] != local_model['digest']
        ):
            parser.error('Installed model changed; select a new output file.')
        results['model_digest'] = local_model['digest']
        if not args.no_phoenix:
            from phoenix.otel import register
            from openinference.instrumentation.ollama import OllamaInstrumentor
            from openinference.instrumentation import (
                Annotation,
                get_evaluation_attributes,
            )
            provider = register(
                project_name='tool-output-robustness-scaled',
                auto_instrument=False,
                batch=False,
            )
            # Ollama instrumentation adds a child LLM span for every chat call.
            OllamaInstrumentor().instrument(tracer_provider=provider)
            tracer = provider.get_tracer(__name__)
    done = len(results['observations'])
    for case in cases:
        answer = oracle(case)
        # Shuffle deterministically to reduce run-order effects while allowing
        # an interrupted run to resume with the same schedule.
        conditions = list(itertools.product(METHODS, ORDERS, NOISES))
        random.Random(f'{seed}:schedule:{case["id"]}').shuffle(conditions)
        valid_ids = {r['id'] for r in case['records']}
        for method, order, noise in conditions:
            key = '/'.join((case['id'], method, order, noise))
            response = prepare(
                case,
                dataset['tool_responses'][case['id']][noise],
                method,
                order,
                noise,
                seed,
            )
            policy = dict(
                constraints=case['constraints'],
                ranking=case['ranking'],
                constraint_semantics=dataset['constraint_semantics'],
            )
            content = json.dumps(
                dict(
                    request=case['request'],
                    public_policy=policy,
                    tool_response=response,
                ),
                sort_keys=True,
                separators=(',', ':'),
                ensure_ascii=True,
            )
            prompt = [
                {'role': 'system', 'content': dataset['system_prompt']},
                {'role': 'user', 'content': content},
            ]
            prompt_size = (
                len(content.encode())
                + len(dataset['system_prompt'].encode())
                + 1216
            )
            assert prompt_size <= dataset['options']['num_ctx']
            if key in results['observations'] or args.dry_run:
                continue
            selected = error = None
            prompt_tokens = output_tokens = None
            started = time.perf_counter()
            # The decision span holds labels and correctness. Ollama adds a
            # child span with details about the model call.
            context = (
                tracer.start_as_current_span('tool_output_decision')
                if not args.no_phoenix
                else nullcontext()
            )
            with context as span:
                if span:
                    attributes = dict(
                        case_id=case['id'],
                        domain=case['domain'],
                        difficulty=case['difficulty'],
                        method=method,
                        order=order,
                        noise=noise,
                        oracle=answer,
                        model=args.model,
                    )
                    for name, val in attributes.items():
                        span.set_attribute(f'experiment.{name}', val)
                    span.set_attribute('openinference.span.kind', 'CHAIN')
                    span.set_attribute('input.value', json.dumps(prompt))
                try:
                    reply = client.chat(
                        model=args.model,
                        messages=prompt,
                        format=dataset['schema'],
                        think=think,
                        options=dataset['options'],
                        stream=False,
                        keep_alive='10m',
                    )
                    value = json.loads(reply.message.content)
                    if (
                        type(value) is not dict
                        or set(value) != {'selected_id'}
                        or value['selected_id'] not in valid_ids
                    ):
                        raise ValueError('Invalid model selection')
                    selected = value['selected_id']
                    prompt_tokens = getattr(reply, 'prompt_eval_count', None)
                    output_tokens = getattr(reply, 'eval_count', None)
                except Exception as exc:
                    error = str(exc)
                    if span:
                        span.record_exception(exc)
                duration = time.perf_counter() - started
                if span:
                    span.set_attribute('output.value', selected or 'INVALID')
                    span.set_attribute('evaluation.correct', selected == answer)
                    span.set_attribute('evaluation.invalid', selected is None)
                    span.set_attributes(
                        get_evaluation_attributes(
                            evaluations=[
                                Annotation(
                                    name='correctness',
                                    score=float(selected == answer),
                                    annotator_kind='CODE',
                                )
                            ]
                        )
                    )
                    span.set_attribute('experiment.duration_s', duration)
                    if isinstance(prompt_tokens, int):
                        span.set_attribute('llm.token_count.prompt', prompt_tokens)
                    if isinstance(output_tokens, int):
                        span.set_attribute('llm.token_count.completion', output_tokens)
            results['observations'][key] = dict(
                case=case['id'],
                domain=case['domain'],
                method=method,
                order=order,
                noise=noise,
                selected=selected,
                oracle=answer,
                correct=selected == answer,
                error=error,
                duration_s=duration,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            done += 1
            results['metrics'] = summary(cases, results['observations'])
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print(
                f'{done}/{len(cases) * 27} {key}: {selected or "INVALID"}',
                flush=True,
            )
    if args.dry_run:
        print(
            f'Dry-run validated {len(cases)} cases and {len(cases) * 27} '
            'conditions; no model inference.'
        )
        return
    results['metrics'] = summary(cases, results['observations'])
    results['status'] = (
        'complete'
        if len(results['observations']) == len(cases) * 27
        else 'partial'
    )
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    report(results)


if __name__ == '__main__':
    main()
