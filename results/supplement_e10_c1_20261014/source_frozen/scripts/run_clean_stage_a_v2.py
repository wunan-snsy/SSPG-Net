"""Isolated clean-data replication using unchanged legacy training algorithms.

Only data and output paths are redirected. Stage A stops after E1 and reliability;
it never chooses hyperparameters or retries based on Test performance.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/clean_revision_v2/stage_a'
AUDIT=ROOT/'reports/clean_revision_v2/audit'
MANIFEST=ROOT/'manifests/clean_revision_v2/decoder_authentic.csv'
WEIGHTS=ROOT/'checkpoints/clean_revision_v2'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def gate():
    approval=json.loads((AUDIT/'training_gate.json').read_text())
    if approval['status']!='PASS_FOR_AUTHENTIC_ONLY_DECODER':raise RuntimeError('Lineage review is unresolved')
    if sha(MANIFEST)!=approval['decoder_manifest_sha256']:raise RuntimeError('Decoder manifest changed')
    if sha(ROOT/'manifests/imd2020_manifest_e5_clean.csv')!=approval['downstream_manifest_sha256']:raise RuntimeError('Downstream manifest changed')
    lineage=list(csv.DictReader((AUDIT/'image_lineage.csv').open()))
    by_path={r['image_path']:r for r in lineage}
    rows=list(csv.DictReader(MANIFEST.open()))
    partitions={s:set() for s in ['train','val']}
    heldout={r['pixel_sha256'] for r in lineage if r['label']=='0' and r['downstream_split'] in ['val','test']}
    for r in rows:
        source=by_path[r['image_path']]
        if r['label']!='0' or source['downstream_split']!='train':raise RuntimeError('Forbidden decoder source')
        if sha(ROOT/'data/imd2020'/r['image_path'])!=source['file_sha256']:raise RuntimeError('Source image changed')
        partitions[r['split']].add(source['pixel_sha256'])
    if not all(partitions.values()) or partitions['train'] & partitions['val']:raise RuntimeError('Internal decoder overlap')
    if (partitions['train']|partitions['val']) & heldout:raise RuntimeError('Held-out overlap')
    return approval

def decoder():
    from training import train_stego_structured_v33 as m
    m.MANIFEST=MANIFEST;m.CHECKPOINT_DIR=WEIGHTS;m.RESULT_DIR=OUT/'decoder'
    m.BEST_CHECKPOINT=WEIGHTS/'decoder_best.pth';m.LAST_CHECKPOINT=WEIGHTS/'decoder_last.pth'
    m.TRAINING_LOG=m.RESULT_DIR/'training_log.csv'
    if m.BEST_CHECKPOINT.exists():raise RuntimeError('Refusing to overwrite existing clean decoder')
    m.main()

def full():
    from training import train_e53r_joint_v1 as m
    m.STEGO_CHECKPOINT=WEIGHTS/'decoder_best.pth';m.BEST_CHECKPOINT=WEIGHTS/'full_best.pth'
    target=OUT/'full'
    for name,filename in [('HISTORY_PATH','history.csv'),('VAL_PREDICTIONS_PATH','val_predictions.csv'),('SUMMARY_PATH','summary.csv'),('PROTOCOL_SNAPSHOT_PATH','legacy_protocol.json')]:
        setattr(m,name,target/filename)
    if m.BEST_CHECKPOINT.exists():raise RuntimeError('Refusing to overwrite existing clean FULL')
    m.main()

def e1():
    from scripts import eval_test_final as m
    m.OUT=OUT/'e1';m.CHECKPOINT=WEIGHTS/'full_best.pth';m.STEGO=WEIGHTS/'decoder_best.pth'
    m.VAL_SUMMARY=OUT/'full/summary.csv'
    if (m.OUT/'detection_metrics.json').exists():raise RuntimeError('E1 already evaluated')
    m.main()
    old=ROOT/'results/test_evaluation'
    old_det=json.loads((old/'detection_metrics.json').read_text());old_loc=json.loads((old/'localization_metrics.json').read_text())
    det=json.loads((m.OUT/'detection_metrics.json').read_text());loc=json.loads((m.OUT/'localization_metrics.json').read_text())
    rows=[{'metric':key,'historical':a[key],'clean':b[key],'difference':b[key]-a[key]} for a,b,keys in [(old_det,det,['roc_auc','f1']),(old_loc,loc,['pixel_f1','pixel_iou'])] for key in keys]
    with (OUT/'e1_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def reliability():
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from training import train_stego_structured_v33 as m
    model=m.StructuredStegoDecoderV33().to('cuda')
    ckpt=torch.load(WEIGHTS/'decoder_best.pth',map_location='cuda',weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']);model.eval()
    rows=list(csv.DictReader((ROOT/'manifests/imd2020_manifest_e5_clean.csv').open()))
    all_results=[];totals={}
    # Held-out reliability is measured only after FULL selection and E1.
    with torch.no_grad():
        for split in ['val','test']:
            selected=[r for r in rows if r['split']==split and r['label']=='0']
            for r in selected:
                with Image.open(ROOT/'data/imd2020'/r['image_path']) as im:
                    x=pil_to_tensor(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).float().unsqueeze(0).to('cuda')/255
                seed=int.from_bytes(hashlib.sha256(('clean-reliability-v2|'+r['image_path']).encode()).digest()[:8],'big')%(2**63-1)
                gen=torch.Generator(device='cuda').manual_seed(seed)
                payload=torch.randint(0,2,(1,128),generator=gen,device='cuda').float()
                stego=m.embed_payload(x,payload);prediction=model(stego)>=0
                correct=int((prediction==payload.bool()).sum());mse=float((stego-x).square().mean())
                all_results.append(dict(split=split,image_path=r['image_path'],correct_bits=correct,total_bits=128,exact_recovery=int(correct==128),psnr_db=float(10*np.log10(1/mse))))
            part=[r for r in all_results if r['split']==split]
            bit=sum(r['correct_bits'] for r in part)/(128*len(part))
            totals[split]={'sources':len(part),'bit_accuracy':bit,'ber':1-bit,'exact_recovery':sum(r['exact_recovery'] for r in part)/len(part),'mean_psnr_db':float(np.mean([r['psnr_db'] for r in part]))}
    write_json(OUT/'decoder/reliability.json',totals)
    with (OUT/'decoder/reliability_per_source.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_results[0]));w.writeheader();w.writerows(all_results)
    print(json.dumps(totals,indent=2))

def pipeline():
    approval=gate();OUT.mkdir(parents=True,exist_ok=True);WEIGHTS.mkdir(parents=True,exist_ok=True)
    if (OUT/'status.json').exists():raise RuntimeError('Existing run found; inspect status before resuming individual stages')
    sources=['training/train_stego_structured_v33.py','training/train_e53r_joint_v1.py','training/e53r_training_protocol_v1.py','scripts/eval_test_final.py','scripts/run_clean_stage_a_v2.py']
    write_json(OUT/'run_provenance.json',{'experiment_id':'CLEAN-A-V2','created':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'pid':os.getpid(),'python':sys.executable,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':{p:sha(ROOT/p) for p in sources},'gate':approval,'method':'Unchanged legacy decoder/FULL algorithms with isolated paths. Legacy IDs in embedded checkpoints identify implementation, not experiment lineage; this sidecar identifies CLEAN-A-V2.','decoder_data_change':'287 unique authentic Train images, deterministic 85/15 internal split; excludes all native manipulated images because folder lineage is insufficient.','decision_rule':'One frozen run; no test-guided tuning or numerical success requirement. All stage-A outcomes reported.'})
    completed=[]
    for stage in ['decoder','full','e1','reliability']:
        write_json(OUT/'status.json',{'state':'RUNNING','stage':stage,'completed':completed,'pid':os.getpid(),'updated':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        with (OUT/f'{stage}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='42',PYTHONIOENCODING='utf-8'))
        if result.returncode:
            write_json(OUT/'status.json',{'state':'FAILED','stage':stage,'completed':completed,'returncode':result.returncode});raise RuntimeError(f'{stage} failed; see stage log')
        completed.append(stage)
    write_json(OUT/'status.json',{'state':'COMPLETE','completed':completed,'next':'Analyze A before preparing B; no automatic architecture changes.'})
    write_json(OUT/'checkpoint_hashes.json',{p.name:sha(p) for p in WEIGHTS.glob('*.pth')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pipeline','decoder','full','e1','reliability','preflight'],default='pipeline');args=parser.parse_args()
    if args.stage=='pipeline':pipeline()
    else:
        import torch
        gate()
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
        torch.set_num_threads(4)
        if args.stage=='preflight':print(json.dumps({'cuda':torch.cuda.get_device_name(0),'gate':'passed','decoder_manifest_sha256':sha(MANIFEST)},indent=2))
        else:globals()[args.stage]()
