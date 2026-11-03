#!/usr/bin/env python3
"""Create reproducible, realistically structured (but synthetic) tool decisions."""
import argparse
import json
import random
from pathlib import Path

SEED = 20261025


def matches(value, op, target):
    return {'<=': lambda: value <= target, '>=': lambda: value >= target,
            '==': lambda: value == target, 'contains': lambda: target in value}[op]()


def template(domain, level, rng):
    """Return a request, candidate-value pools, hard constraints, and ranking priorities."""
    if domain == 'cloud_compute':
        region = rng.choice(['us-west-2', 'us-east-1', 'eu-central-1'])
        ram = rng.choice([32, 64, 96])
        request = f'Choose an available compute instance in {region} with at least {ram} GB RAM.'
        pools = dict(region=['us-west-2', 'us-east-1', 'eu-central-1'], memory_gb=[16, 32, 48, 64, 96, 128],
                     available=[True, False], hourly_cents=[34, 49, 65, 89, 115, 155, 209],
                     p95_latency_ms=[12, 18, 25, 35, 49, 80], encrypted=[True, False],
                     uptime_bps=[9900, 9950, 9990, 9999])
        constraints = [('region', '==', region), ('memory_gb', '>=', ram), ('available', '==', True)]
        ranking = [('hourly_cents', 'asc')]
        if level >= 1:
            constraints += [('p95_latency_ms', '<=', 35)]
            ranking += [('p95_latency_ms', 'asc')]
        if level == 2:
            constraints += [('encrypted', '==', True), ('uptime_bps', '>=', 9990)]
            ranking += [('uptime_bps', 'desc')]
    elif domain == 'support_resolution':
        age = rng.choice([7, 12, 20, 27]); cap = 100 if age <= 14 else 80
        request = f'Select the best damaged-item resolution for an order purchased {age} days ago.'
        pools = dict(max_age_days=[7, 14, 21, 30, 45], accepted_reasons=[['damaged'], ['late'], ['damaged', 'wrong_item']],
                     refund_pct=[40, 60, 75, 80, 100], resolution_days=[1, 2, 3, 5, 10],
                     receipt_required=[True, False], fee_cents=[0, 299, 599], return_required=[True, False])
        constraints = [('max_age_days', '>=', age), ('accepted_reasons', 'contains', 'damaged'), ('refund_pct', '<=', cap)]
        ranking = [('refund_pct', 'desc')]
        if level >= 1:
            constraints += [('receipt_required', '==', False)]
            ranking += [('resolution_days', 'asc')]
        if level == 2:
            constraints += [('fee_cents', '<=', 0), ('return_required', '==', False)]
            ranking += [('fee_cents', 'asc')]
    elif domain == 'shipping':
        days = rng.choice([2, 3, 4]); budget = rng.choice([1900, 2500, 3200])
        request = f'Book a certified lab shipment carrier delivering within {days} days.'
        pools = dict(delivery_days=[1, 2, 3, 4, 5, 7], certified=[True, False],
                     price_cents=[995, 1295, 1595, 1895, 2295, 2895, 3395],
                     tracking=[True, False], on_time_pct=[91, 95, 96, 98, 99],
                     chain_of_custody=[True, False])
        constraints = [('delivery_days', '<=', days), ('certified', '==', True)]
        ranking = [('price_cents', 'asc')]
        if level >= 1:
            constraints += [('price_cents', '<=', budget), ('tracking', '==', True)]
            ranking += [('delivery_days', 'asc')]
        if level == 2:
            constraints += [('chain_of_custody', '==', True), ('on_time_pct', '>=', 96)]
            ranking += [('on_time_pct', 'desc')]
    elif domain == 'saas_vendor':
        region = rng.choice(['US', 'EU']); seats = rng.choice([100, 250, 500])
        request = f'Select a SOC2-attested SaaS vendor supporting {seats} seats with data residency in {region}.'
        pools = dict(soc2=[True, False], residency=['US', 'EU'], seats_supported=[50, 100, 250, 500, 1000],
                     monthly_cents=[14900, 19900, 29900, 39900, 59900], onboarding_days=[3, 7, 14, 21, 30],
                     sso=[True, False], uptime_bps=[9900, 9950, 9990, 9999])
        constraints = [('soc2', '==', True), ('residency', '==', region), ('seats_supported', '>=', seats)]
        ranking = [('monthly_cents', 'asc')]
        if level >= 1:
            constraints += [('onboarding_days', '<=', 14)]
            ranking += [('onboarding_days', 'asc')]
        if level == 2:
            constraints += [('sso', '==', True), ('uptime_bps', '>=', 9990)]
            ranking += [('uptime_bps', 'desc')]
    elif domain == 'data_pipeline':
        region = rng.choice(['us-west', 'us-east', 'eu-west']); rows = rng.choice([50, 100, 200])
        request = f'Choose a data ingestion connector in {region} for at least {rows} thousand rows/minute.'
        pools = dict(region=['us-west', 'us-east', 'eu-west'], throughput_kpm=[20, 50, 100, 200, 300],
                     encryption=[True, False], monthly_cents=[9900, 14900, 22000, 34000, 47000],
                     freshness_seconds=[15, 30, 60, 120, 300], retry_support=[True, False],
                     availability_bps=[9900, 9950, 9990, 9999])
        constraints = [('region', '==', region), ('throughput_kpm', '>=', rows), ('encryption', '==', True)]
        ranking = [('monthly_cents', 'asc')]
        if level >= 1:
            constraints += [('freshness_seconds', '<=', 120)]
            ranking += [('freshness_seconds', 'asc')]
        if level == 2:
            constraints += [('retry_support', '==', True), ('availability_bps', '>=', 9990)]
            ranking += [('availability_bps', 'desc')]
    else:  # fulfillment_vendor
        units = rng.choice([80, 120, 160]); cap = rng.choice([1800, 2200, 2600])
        request = f'Choose a certified packaging supplier able to ship {units} units each week.'
        pools = dict(certified=[True, False], weekly_units=[60, 80, 120, 160, 240],
                     unit_cents=[845, 1175, 1399, 1650, 1990, 2350, 2799],
                     lead_days=[2, 3, 5, 8, 12], defect_bps=[12, 25, 40, 75, 110], recyclable=[True, False])
        constraints = [('certified', '==', True), ('weekly_units', '>=', units)]
        ranking = [('unit_cents', 'asc')]
        if level >= 1:
            constraints += [('unit_cents', '<=', cap), ('lead_days', '<=', 8)]
            ranking += [('lead_days', 'asc')]
        if level == 2:
            constraints += [('defect_bps', '<=', 40), ('recyclable', '==', True)]
            ranking += [('defect_bps', 'asc')]
    return request, pools, constraints, ranking


def build(count, seed):
    domains = ['cloud_compute', 'support_resolution', 'shipping', 'saas_vendor', 'data_pipeline', 'fulfillment_vendor']
    cases, tool_responses = [], {}
    for index in range(count):
        rng = random.Random(f'{seed}:case:{index}')
        domain = domains[index % len(domains)]
        level = (index // len(domains) + index % 3) % 3
        request, pools, constraints, ranking = template(domain, level, rng)
        fields = sorted({row[0] for row in constraints} | {row[0] for row in ranking})
        ids = rng.sample(range(1000, 9999), 8)
        records = [{'id': f'R-{ids[j]}', **{f: rng.choice(pools[f]) for f in fields}} for j in range(8)]
        for record in records[:3]:
            for field, op, target in constraints:
                record[field] = rng.choice([v for v in pools[field] if matches(v, op, target)])
        if level >= 1:
            records[1][ranking[0][0]] = records[0][ranking[0][0]]
        if level == 2 and len(ranking) > 1:
            records[1][ranking[1][0]] = records[0][ranking[1][0]]
        near_miss = dict(records[0], id=records[-1]['id'])
        # Choose a constraint that has at least one possible violating value.
        field, op, target = next((f, operation, threshold) for f, operation, threshold in constraints
                                 if any(not matches(v, operation, threshold) for v in pools[f]))
        near_miss[field] = rng.choice([v for v in pools[field] if not matches(v, op, target)])
        primary, direction = ranking[0]
        viable_primary = [v for v in pools[primary] if all(
            f != primary or matches(v, op2, target2) for f, op2, target2 in constraints)]
        near_miss[primary] = (max if direction == 'desc' else min)(viable_primary)
        records[-1] = near_miss
        rng.shuffle(records)
        case_id = f'case-{index+1:03}'
        cases.append(dict(id=case_id, domain=domain, difficulty=['easy', 'medium', 'hard'][level],
                          request=request, constraints=constraints, ranking=ranking,
                          fields=['id'] + fields, records=records))
        tool_responses[case_id] = {}
        for noise in ('clean', 'moderate', 'heavy'):
            output = {'records': json.loads(json.dumps(records))}
            if noise != 'clean':
                output['tool_context'] = {'source': f'{domain}.search', 'request_id': f'req_{index:04}',
                                          'page': 1, 'next_cursor': None, 'export_version': 'v3'}
                for record in output['records']:
                    record['metadata'] = {'catalog_key': f'{domain}/{record["id"]}', 'owner_team': 'operations',
                                          'updated_by': 'sync-worker', 'ui_label': 'inventory details'}
            if noise == 'heavy':
                output['export_diagnostics'] = {'batch_id': f'export_{seed}_{index}',
                    'notes': 'The inventory web console generated previews, navigation links and a refreshed change log. '
                             'These administrative artifacts are unrelated to eligibility or ranking.',
                    'archived_events': [{'kind': 'page_view', 'page': 'reports'},
                                        {'kind': 'sync', 'page': 'settings'},
                                        {'kind': 'thumbnail_refresh', 'page': 'catalog'}]}
                for record in output['records']:
                    record['metadata']['audit'] = {'seen_in_ui': True, 'export_batch': f'batch-{index:03}',
                        'description': 'This entry appeared in a periodic administrative export of the catalog. '
                                       'The UI audit trail concerns how the item was displayed, not its properties.'}
            tool_responses[case_id][noise] = output
    return dict(seed=seed, provenance='Synthetic simulated API/catalog snapshots; not production or public benchmark data.',
                system_prompt=('Make one decision using only the user request and public policy. All constraints '
                               'must hold. Rank eligible records lexicographically using ranking in listed priority '
                               'order (asc means smaller first; desc means larger first). Break remaining ties by '
                               'lexicographically smallest ID. Metadata has no role in this policy. '
                               'Return only compact JSON: {"selected_id":"one candidate ID"}.'),
                constraint_semantics='Every [field, operator, value] must hold; contains means list membership.',
                schema={'type': 'object', 'properties': {'selected_id': {'type': 'string'}},
                        'required': ['selected_id'], 'additionalProperties': False},
                options=dict(temperature=0, top_k=1, top_p=1, min_p=0, seed=seed, repeat_penalty=1,
                             presence_penalty=0, frequency_penalty=0, num_ctx=8192, num_predict=192, num_batch=128),
                cases=cases, tool_responses=tool_responses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, default=48)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output', type=Path, default=Path('cases_scaled.json'))
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    data = build(args.cases, args.seed)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(f"Saved {args.cases} synthetic cases to {args.output}")

if __name__ == '__main__':
    main()
