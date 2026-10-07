import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
parser=argparse.ArgumentParser(description='Plot completed autojev noncausal/causal training and resource curves.')
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path)
parser.add_argument('--baseline',type=Path,default=Path('baselines/base-development.json'))
args=parser.parse_args()
root=args.root
results={name:{'summary':json.loads((root/f'runs/{name}/run/summary.json').read_text()),
               'resources':json.loads((root/f'logs/{name}-resources.json').read_text())}
         for name in ['full-noncausal','full-causal']}
folder=args.output or root/'plots/full-public'
folder.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':0.22,'figure.dpi':120,'savefig.dpi':200})
names=['full-noncausal','full-causal']
labels={'full-noncausal':'Noncausal full attention','full-causal':'Causal'}
colors={'full-noncausal':'#2475b4','full-causal':'#dc7335'}
data={}; sources={}
for name in names:
 assert results[name]['summary']['complete'],f'{name}: training is incomplete'
 data[name]={}
 for kind in ['training','evaluations']:
  path=root/f'runs/{name}/run/{kind}.jsonl'
  data[name][kind]=[json.loads(line) for line in path.read_text().splitlines()]
  sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
 assert len(data[name]['training'])==results[name]['summary']['steps']
baseline=json.loads((root/args.baseline).read_text())['overall']
for path in [root/args.baseline]+[root/f'runs/{name}/run/summary.json' for name in names]+[root/f'logs/{name}-resources.json' for name in names]:
 sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
def rolling(values,window=50):
 values=np.asarray(values,dtype=float)
 prefix=np.concatenate(([0.0],np.cumsum(values)))
 ends=np.arange(1,len(values)+1); starts=np.maximum(0,ends-window)
 return (prefix[ends]-prefix[starts])/(ends-starts)
def finish(fig,stem):
 fig.tight_layout(pad=1.6)
 for extension in ['png','svg']:
  fig.savefig(folder/f'{stem}.{extension}',bbox_inches='tight')
 plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13,8))
for name in names:
 train=data[name]['training']; evaluations=data[name]['evaluations']; c=colors[name]; label=labels[name]
 steps=[r['step'] for r in train]
 axes[0,0].plot(steps,[r['loss'] for r in train],color=c,alpha=.15,linewidth=.6)
 axes[0,0].plot(steps,rolling([r['loss'] for r in train]),color=c,label=label,linewidth=1.6)
 es=[r['step'] for r in evaluations]
 axes[0,1].plot(es,[r['fitted']['accuracy']*100 for r in evaluations],'-o',color=c,label=label,markersize=3)
 selected=results[name]['summary']['best_step']
 chosen=next(r for r in evaluations if r['step']==selected)
 axes[0,1].scatter([selected],[chosen['fitted']['accuracy']*100],color=c,marker='*',s=150,zorder=5)
 for ax,key in [(axes[1,0],'ece'),(axes[1,1],'brier')]:
  for state,style in [('raw','--'),('fitted','-')]:
   ax.plot(es,[r[state][key] for r in evaluations],style,color=c,label=label+' / '+('calibrated' if state=='fitted' else 'raw'))
axes[0,0].set_title('Training loss: raw (faint), trailing 50-step mean')
axes[0,1].set_title('Development accuracy; stars mark selected checkpoints')
axes[0,1].axhline(baseline['accuracy']*100,color='#555555',linestyle=':',label='Fixed base reference')
axes[0,1].set_ylabel('Accuracy (%)')
axes[1,0].set_title('Development ECE, 15 bins'); axes[1,1].set_title('Development hard-label Brier')
axes[1,0].axhline(baseline['ece']+.01,color='#555555',linestyle=':',label='Calibration gate')
axes[1,1].axhline(baseline['brier']+.01,color='#555555',linestyle=':',label='Calibration gate')
for ax in axes.flat:
 ax.set_xlabel('Optimizer step'); ax.legend(fontsize=8)
finish(fig,'learning_and_calibration')
fig,axes=plt.subplots(1,3,figsize=(16,4.7))
for name in names:
 rows=data[name]['training']; x=[r['step'] for r in rows]; c=colors[name]
 axes[0].plot(x,[r['step_seconds'] for r in rows],color=c,alpha=.15,linewidth=.6)
 axes[0].plot(x,rolling([r['step_seconds'] for r in rows]),color=c,label=labels[name])
 axes[1].plot(x,[r['gpu_peak_gb'] for r in rows],color=c,label=labels[name])
axes[0].set_title('Step wall time: raw + trailing 50-step mean'); axes[0].set_ylabel('Seconds')
axes[1].set_title('Cumulative peak GPU allocated memory'); axes[1].set_ylabel('GB (decimal)')
for ax in axes[:2]: ax.set_xlabel('Optimizer step'); ax.legend(fontsize=8)
positions=np.arange(len(names)); resources=[results[name]['resources'] for name in names]
axes[2].bar(positions-.18,[r['peak_rss_bytes']/1e9 for r in resources],width=.36,label='Peak RSS',color='#497e9b')
axes[2].bar(positions+.18,[r['rss_budget_bytes']/1e9 for r in resources],width=.36,label='Allowed RSS (80% initial available)',color='#aec4d2')
axes[2].set_xticks(positions,['Noncausal','Causal']); axes[2].set_ylabel('GB (decimal)'); axes[2].set_title('Final process RSS and configured budgets'); axes[2].legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.14))
finish(fig,'time_and_resources')
artifacts=[]
for path in sorted(folder.glob('*')):
 if path.suffix not in ['.png','.svg']: continue
 record={'path':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 if path.suffix=='.png':
  with Image.open(path) as picture:
   picture.load(); record['pixels']=list(picture.size)
   assert min(picture.size)>500
 artifacts.append(record)
assert len(artifacts)==4
(folder/'plot-manifest.json').write_text(json.dumps({'passed':True,'sources':sources,'artifacts':artifacts,'notes':['Trailing means use available preceding steps, up to 50.','GPU curves are cumulative torch.cuda.max_memory_allocated at each training step, not instantaneous or whole-card use.','Dev is used for checkpoint selection; no test inference was performed.']},indent=2))
print(f'Saved PNG/SVG plots and source hashes to {folder}')
