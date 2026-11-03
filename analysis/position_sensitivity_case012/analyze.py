#!/usr/bin/env python3
"""Compute descriptive tables and two static charts from verified Phoenix evidence.

Uses Python with matplotlib; makes no Phoenix writes or model calls.
"""
from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent,
                    help='Directory containing the completed position study.')
HERE = parser.parse_args().out.resolve()
with (HERE/'evidence.json').open() as f: evidence=json.load(f)
with (HERE/'configuration.json').open() as f: config=json.load(f)
rows=evidence['decisions']
assert len(rows)==46 and evidence['inference_attempts']==46
trials=[r for r in rows if r['metadata']['kind']=='position']
controls=[r for r in rows if r['metadata']['kind']=='control']
assert len(trials)==40 and len(controls)==6
position_summary=[]
for pos in range(1,9):
    subset=[r for r in trials if r['metadata']['target_position']==pos]
    assert len(subset)==5
    counts=Counter(r['output']['selected_id'] or 'INVALID' for r in subset)
    position_summary.append(dict(position=pos,total=len(subset),correct=counts['R-9732'],
        accuracy=counts['R-9732']/len(subset),selections=dict(sorted(counts.items())),
        selected_R_6880=counts['R-6880']))
seed_rows=[]
for seed in config['background_seeds']:
    subset=sorted((r for r in trials if r['metadata']['background_seed']==seed),
                  key=lambda r:r['metadata']['target_position'])
    assert [r['metadata']['target_position'] for r in subset]==list(range(1,9))
    picks=[r['output']['selected_id'] or 'INVALID' for r in subset]
    hits=[i+1 for i,p in enumerate(picks) if p=='R-9732']
    seed_rows.append(dict(seed=seed,background_order=config['background_orders'][str(seed)],
        selections=picks,correct_positions=hits,oracle_selection_changes=0<len(hits)<8,
        selected_id_changes=len(set(picks))>1))
control_results={}
for method in ('RAW','CANONICAL'):
    subset=sorted((r for r in controls if r['metadata']['control']==method),
                  key=lambda r:r['metadata']['control_repetition'])
    assert len(subset)==3
    historic=config['historical_controls'][method]['selected_id']
    control_results[method]={'historical_selection':historic,'selected_ids':[r['output']['selected_id'] for r in subset],
        'matches_historical':sum(r['output']['selected_id']==historic for r in subset),
        'prompt_sha256':subset[0]['metadata']['prompt_sha256'],'trace_ids':[r['trace_id'] for r in subset]}
    assert len({r['metadata']['prompt_sha256'] for r in subset})==1
    assert subset[0]['metadata']['prompt_sha256']==config['historical_controls'][method]['prompt_sha256']
by_prompt=defaultdict(list)
for row in rows: by_prompt[row['metadata']['prompt_sha256']].append(row)
repeated=[]
for digest,group in sorted(by_prompt.items()):
    if len(group)<2: continue
    selected=[r['output']['selected_id'] for r in group]
    raw=[r['output']['raw_response'] for r in group]
    repeated.append(dict(prompt_sha256=digest,count=len(group),trial_ids=[r['trial_id'] for r in group],
        selected_ids=selected,distinct_selected_ids=list(sorted({str(x) for x in selected})),
        distinct_raw_responses=list(sorted({str(x) for x in raw})),
        inconsistent_selection=len(set(selected))>1,inconsistent_raw_response=len(set(raw))>1))
first=sum(p['correct'] for p in position_summary[:4]);last=sum(p['correct'] for p in position_summary[4:])
competitor_by_seed={str(row['seed']):row['selections'].count('R-6880') for row in seed_rows}
result=dict(model=config['model'],model_digest=config['model_digest'],case_id='case-012',
    target='R-9732',inference_attempts=len(rows),inference_errors=evidence['model_response_errors'],
    position_results=position_summary,background_results=seed_rows,
    early_positions={'correct':first,'total':20,'accuracy':first/20},
    late_positions={'correct':last,'total':20,'accuracy':last/20},
    controls=control_results,identical_prompt_groups=repeated,
    R_6880_by_background_seed=competitor_by_seed,
    R_6880_total=sum(competitor_by_seed.values()),
    identical_prompt_groups_with_selection_changes=sum(x['inconsistent_selection'] for x in repeated),
    identical_prompt_groups_with_raw_response_changes=sum(x['inconsistent_raw_response'] for x in repeated),
    dataset_id=evidence['dataset_id'],dataset_version_id=evidence['dataset_version_id'],
    experiment_id=evidence['experiment_id'],experiment_url=evidence['experiment_url'])
(HERE/'analysis_results.json').write_text(json.dumps(result,indent=2)+'\n')
with (HERE/'accuracy_by_position.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['correct_candidate_position','correct','total','accuracy','selected_ID_distribution','selected_R_6880'])
    for r in position_summary:w.writerow([r['position'],r['correct'],r['total'],r['accuracy'],json.dumps(r['selections'],sort_keys=True),r['selected_R_6880']])
with (HERE/'selection_heatmap.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['background_seed']+[f'position_{i}' for i in range(1,9)])
    for row in seed_rows:w.writerow([row['seed']]+row['selections'])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.edgecolor':'#bdc9bf','text.color':'#25322c',
                     'axes.labelcolor':'#424b45','xtick.color':'#536157','ytick.color':'#536157'})
paper='#f8f7f3';green='#416e58';muted='#667168'
fig,ax=plt.subplots(figsize=(10,4.6),facecolor=paper);ax.set_facecolor(paper)
vals=[100*r['accuracy'] for r in position_summary]
bars=ax.bar(range(1,9),vals,width=.62,color=green,zorder=3)
ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100],labels=['0%','20%','40%','60%','80%','100%'])
ax.set_xticks(range(1,9));ax.set_xlabel('Position of oracle candidate R-9732',labelpad=12)
ax.set_ylabel('Exact-match accuracy');ax.grid(axis='y',color='#dce0d8',linewidth=.8,zorder=0)
for bar,row in zip(bars,position_summary): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{row['correct']}/5",ha='center',va='bottom',fontsize=11,fontweight='bold')
ax.set_title('Does the oracle candidate’s position change the choice?',loc='left',fontsize=15,fontweight='bold',pad=22)
fig.text(.125,.015,'Five background orderings per position; all 40 trials use the same synthetic scenario.',fontsize=9,color=muted,va='bottom')
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(HERE/'accuracy_by_position.png',dpi=180,facecolor=paper);plt.close(fig)
all_ids=sorted({p for row in seed_rows for p in row['selections']})
other=[x for x in all_ids if x not in ('R-9732','R-6880','INVALID')]
colors={'R-9732':'#416e58','R-6880':'#c28451','INVALID':'#7d817c'}
palette=['#647f8e','#86759a','#be9b4d','#bd6b66','#638c83','#a8778b','#8b8d52']
colors.update({name:palette[i%len(palette)] for i,name in enumerate(other)})
ordered=[x for x in ['R-9732','R-6880']+other+['INVALID'] if x in all_ids]
from matplotlib.colors import ListedColormap
colorindex={name:i for i,name in enumerate(ordered)}
matrix=np.array([[colorindex[x] for x in row['selections']] for row in seed_rows])
fig,ax=plt.subplots(figsize=(12,5.1),facecolor=paper);ax.set_facecolor(paper)
ax.imshow(matrix,cmap=ListedColormap([colors[x] for x in ordered]),vmin=-.5,vmax=len(ordered)-.5,aspect='auto')
ax.set_xticks(range(8),labels=range(1,9));ax.set_yticks(range(5),labels=[str(r['seed']) for r in seed_rows])
ax.set_xlabel('Position of R-9732',labelpad=10);ax.set_ylabel('Fixed background shuffle seed',labelpad=10)
ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
ax.grid(which='minor',color=paper,linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
for i,row in enumerate(seed_rows):
    for j,name in enumerate(row['selections']):ax.text(j,i,name,ha='center',va='center',color='white',fontsize=9,fontweight='bold')
ax.set_title('Selected candidate across positions and fixed background orderings',loc='left',fontsize=15,fontweight='bold',pad=18)
handles=[Patch(facecolor=colors[x],label=x+(' · oracle' if x=='R-9732' else '')) for x in ordered]
ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.22),ncol=min(4,len(handles)),frameon=False,fontsize=9)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(HERE/'selection_heatmap.png',dpi=180,facecolor=paper);plt.close(fig)

# Documentation lives in the root README; retain machine-readable results and charts.
print(json.dumps({'position':position_summary,'early':result['early_positions'],'late':result['late_positions'],
                  'controls':control_results,'identical_selection_changes':result['identical_prompt_groups_with_selection_changes']},indent=2))
