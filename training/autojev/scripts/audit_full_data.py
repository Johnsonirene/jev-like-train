import argparse,collections,hashlib,itertools,json
from pathlib import Path
from autojev.data import state_key
root=Path(__file__).resolve().parents[1]
folder=root/'data/full-public'
folds={name:[json.loads(line) for line in (folder/(name+'.jsonl')).open() if line.strip()] for name in ['train','dev','temperature','test','known','image_test']}
result={'files':{},'overlaps':{}}
keys={}
for name,rows in folds.items():
 ids={row['id'] for row in rows}; states={state_key(row) for row in rows}
 families={(row['source']['dataset'],row['family']) for row in rows}
 keys[name]=(ids,states,families)
 result['files'][name]={'rows':len(rows),'sources':dict(collections.Counter(row['source']['dataset'] for row in rows)),'types':dict(collections.Counter(row['question']['type'] for row in rows)),'sha256':hashlib.sha256((folder/(name+'.jsonl')).read_bytes()).hexdigest(),'families':len(families),'duplicate_ids':len(rows)-len(ids)}
for left,right in itertools.combinations(folds,2):
 result['overlaps'][left+'/'+right]={'id':len(keys[left][0]&keys[right][0]),'literal_state':len(keys[left][1]&keys[right][1]),'family':len(keys[left][2]&keys[right][2])}
checks=[len(folds[name])>0 and result['files'][name]['duplicate_ids']==0 for name in ['train','dev','temperature']]
for left,right in itertools.combinations(['train','dev','temperature'],2):
 checks.append(not any(result['overlaps'][left+'/'+right].values()))
for left in ['dev','temperature']:
 for right in ['test','known','image_test']:
  overlap={kind:len(keys[left][index]&keys[right][index]) for index,kind in enumerate(['id','literal_state','family'])}
  checks.append(not any(overlap.values()))
image_train={row['source']['image_sha256'] for row in folds['train'] if row['source']['dataset']=='generated_geometry'}
image_test={row['source']['image_sha256'] for row in folds['image_test']}
result['image_sha256_overlap']=len(image_train&image_test)
checks.append(not image_train&image_test)
result['passed']=all(checks)
(root/'logs/full-data-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result['passed'],'Full public data checkpoint failed; stop and report.'
