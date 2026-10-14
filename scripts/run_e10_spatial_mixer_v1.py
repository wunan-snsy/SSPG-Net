"""Frozen E10-SCM-v1 runner; isolated outputs and sequential CUDA training."""
from pathlib import Path
from dataclasses import asdict, replace
import argparse
import atexit
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/e10_spatial_mixer_v1'
BASE=ROOT/'results/clean_revision_v2'
DEC=ROOT/'checkpoints/clean_revision_v2/decoder_budget142/best.pth'
SEEDS=(20261001,20261002,20261003)
COUNT=1416100
FULL_COUNT=1416490
MODEL_NAME='SPATIAL_MIXER_MATCHED'
CODE_FILES=['models/sspg_spatial_mixer_matched_v2.py','scripts/run_e10_spatial_mixer_v1.py',
            'scripts/summarize_e10_spatial_mixer_v1.py','tests/test_e10_spatial_mixer_v1.py',
            'models/sspg_ablation_model_v2.py','models/sspg_param_matched_no_graph_v2.py',
            'models/sspg_frontend_v2.py','models/vit_backbone_v2.py','models/provenance_heads_v2.py',
            'preprocessing/active_sspg_dataset.py','training/e53r_training_protocol_v1.py',
            'training/train_e53r_joint_v1.py','training/train_e53r_ablation_v1.py',
            'tests/test_e53r_joint_heads_dependency_r1.py']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
    temp.replace(path)


def historical_directory(model,seed):
    return BASE/f'budget142_param_matched_no_graph_e2/{seed}' if model=='matched_no_graph' else BASE/f'budget142_e2/{model}/{seed}'


def verify_historical():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from scripts.run_clean_stage_a_v2 import gate
    gate()  # Train image hashes + held-out metadata only; no Test image loading.
    state=json.loads((BASE/'decoder_budget142/status.json').read_text())
    if state['state']!='COMPLETE' or sha(DEC)!=state['best_checkpoint_sha256']:
        raise RuntimeError(f'BLOCKED: decoder status/hash mismatch: {DEC}')
    hashes={str(DEC.relative_to(ROOT)):sha(DEC)}
    for path in [ROOT/'manifests/imd2020_manifest_e5_clean.csv',ROOT/'manifests/clean_revision_v2/decoder_authentic.csv']:
        hashes[str(path.relative_to(ROOT))]=sha(path)
    rows=[]
    for model in ['full','no_graph','matched_no_graph','no_provenance']:
        for seed in SEEDS:
            directory=historical_directory(model,seed)
            lineage=json.loads((directory/'lineage.json').read_text())
            for filename,key in [('best.pth','checkpoint_sha256'),('metrics.csv','summary_sha256')]:
                path=directory/filename
                if sha(path)!=lineage[key]:
                    raise RuntimeError(f'BLOCKED: historical artifact hash mismatch: {path}')
                hashes[str(path.relative_to(ROOT))]=lineage[key]
            if lineage['test_accessed'] is not False or lineage['decoder_sha256']!=state['best_checkpoint_sha256']:
                raise RuntimeError(f'BLOCKED: historical lineage inconsistent: {directory}')
            metrics=pd.read_csv(directory/'metrics.csv').iloc[0]
            pred=pd.read_csv(directory/'val_predictions.csv')
            if len(pred)!=124 or not pred['detection_label'].value_counts().eq(62).all():
                raise RuntimeError(f'BLOCKED: historical Validation cohort: {directory}')
            tn,fp,fn,tp=confusion_matrix(pred['detection_label'],pred['detection_probability']>=metrics['detection_threshold'],labels=[0,1]).ravel()
            if not np.isclose(roc_auc_score(pred['detection_label'],pred['detection_probability']),metrics['detection_auc'],atol=1e-10):
                raise RuntimeError(f'BLOCKED: historical AUC: {directory}')
            if not np.isclose(2*tp/(2*tp+fp+fn),metrics['detection_f1'],atol=1e-10):
                raise RuntimeError(f'BLOCKED: historical detection F1: {directory}')
            ltp,lfp,lfn=[int(metrics['localization_'+k]) for k in ['tp','fp','fn']]
            if not np.isclose(2*ltp/(2*ltp+lfp+lfn),metrics['localization_pixel_f1'],atol=1e-10):
                raise RuntimeError(f'BLOCKED: historical pixel F1: {directory}')
            rows.append(dict(model=model,seed=seed,pixel_f1=float(metrics['localization_pixel_f1'])))
            for path in [directory/'lineage.json',directory/'val_predictions.csv',
                         directory/('training_protocol.json' if model=='matched_no_graph' else 'protocol.json')]:
                hashes[str(path.relative_to(ROOT))]=sha(path)
    reference={'full':(.7574,.0070),'no_graph':(.7217,.0072),'matched_no_graph':(.6951,.0376),'no_provenance':(.6310,.0011)}
    for model,(mean,sd) in reference.items():
        values=[r['pixel_f1'] for r in rows if r['model']==model]
        if (round(float(np.mean(values)),4),round(float(np.std(values,ddof=1)),4))!=(mean,sd):
            raise RuntimeError(f'BLOCKED: historical table 14 disagrees for {model}')
    return hashes,rows


def configure(seed):
    import torch
    import training.e53r_training_protocol_v1 as proto
    import training.train_e53r_joint_v1 as joint
    old=json.loads((historical_directory('full',seed)/'protocol.json').read_text())
    formal=proto.E53RTrainingProtocolV1(**old)
    if any(old[k]!=asdict(proto.PROTOCOL)[k] for k in old if k not in ['experiment_id','protocol_status','training_seed']):
        raise RuntimeError('BLOCKED: current base protocol differs from completed E2')
    protocol=replace(formal,experiment_id=f'E10-SCM-v1-{seed}',protocol_status='E10_SCM_FROZEN')
    proto.PROTOCOL=protocol
    joint.PROTOCOL=protocol
    joint.STEGO_CHECKPOINT=DEC
    digest=hashlib.sha256(proto.protocol_json().encode()).hexdigest()
    joint.EXPECTED_PROTOCOL_SHA256=digest
    torch.set_num_threads(4)
    return joint,protocol,digest


def install_access_guard(joint,log_path):
    import pandas as pd
    manifest=pd.read_csv(joint.MANIFEST_PATH)
    records=joint.build_source_records(manifest,'train')+joint.build_source_records(manifest,'val')
    allowed={str(Path(r['image_path']).resolve()).casefold() for r in records}
    root_data=str((ROOT/'data/imd2020').resolve()).casefold()+os.sep
    accessed=set()
    forbidden=[]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=str(Path(os.fsdecode(args[0])).resolve()).casefold()
        if path.startswith(root_data):
            if path not in allowed:
                forbidden.append(path)
                raise RuntimeError(f'E10 forbidden image read: {path}')
            accessed.add(path)
        if any(x in path for x in ['e9_coco','data'+os.sep+'casia','data'+os.sep+'coco']):
            forbidden.append(path)
            raise RuntimeError(f'E10 forbidden external data read: {path}')
    sys.addaudithook(guard)
    atexit.register(lambda:write_json(log_path,{'allowed_train_val_source_paths':len(allowed),
        'accessed_source_paths':sorted(accessed),'forbidden_attempts':forbidden,
        'scope':'Raw image reads restricted to E2 Train/Validation authentic source paths; manifests/lineage may contain unused Test metadata.'}))


def preflight():
    import torch
    import pandas as pd
    from models.sspg_spatial_mixer_matched_v2 import SSPGSpatialMixerMatchedV2,EXPANSION
    from preprocessing.active_sspg_dataset import ActiveSSPGDataset,canonical_sample_key
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists():
        verify_frozen()
        print('Existing frozen E10 design verified; preflight was already completed.',flush=True)
        return
    input_hashes,historical=verify_historical()
    if not torch.cuda.is_available():
        raise RuntimeError('BLOCKED: CUDA unavailable')
    joint,protocol,_=configure(SEEDS[0])
    install_access_guard(joint,OUT/'preflight_image_access.json')
    manifest=pd.read_csv(joint.MANIFEST_PATH)
    train=joint.build_source_records(manifest,'train')
    val=joint.build_source_records(manifest,'val')
    joint.audit_train_val_sources(train,val)
    if (len(train),len(val))!=(289,62):
        raise RuntimeError('BLOCKED: source partition changed')
    train_ds=ActiveSSPGDataset(train,base_seed=protocol.train_seed)
    val_ds=ActiveSSPGDataset(val,base_seed=protocol.val_seed)
    assert (len(train_ds),len(val_ds))==(578,124)
    keys={canonical_sample_key(r) for r in val}
    for seed in SEEDS:
        old=pd.read_csv(historical_directory('full',seed)/'val_predictions.csv')
        assert keys==set(old['sample_key'])
    torch.manual_seed(999)  # Synthetic preflight, not a scientific seed run.
    model=SSPGSpatialMixerMatchedV2(str(DEC)).cuda()
    counts={name:sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name,module in model.named_children() if module is not None}
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total==COUNT and abs(total-FULL_COUNT)/FULL_COUNT<=.001
    assert sum(counts.values())==total
    del model
    gc.collect()
    torch.cuda.empty_cache()
    checks=['tests/test_e10_spatial_mixer_v1.py','tests/test_e53r_training_protocol_v1.py',
            'tests/test_e53r_joint_heads_dependency_r1.py','tests/test_e53r_active_dataset_integration.py']
    for test in checks:
        log=OUT/(Path(test).stem+'.log')
        print('Preflight test:',test,flush=True)
        with log.open('w',encoding='utf-8') as f:
            env=dict(os.environ,PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1')
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--test-script',test],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
        if result.returncode:
            raise RuntimeError(f'Preflight test failed: {test}; see {log}')
    candidates=[{'expansion':e,'count':555812+2*(3328+513*e)} for e in range(1,4097)]
    closest=min(candidates,key=lambda r:abs(r['count']-FULL_COUNT))
    assert closest=={'expansion':EXPANSION,'count':COUNT}
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    design={'experiment_id':'E10-SCM-v1','model':MODEL_NAME,'training_seeds':SEEDS,
        'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'python':sys.executable,
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'git_branch':git('branch','--show-current'),'git_commit':git('rev-parse','HEAD'),
        'git_status_before_experiment':git('status','--short'),
        'reference_version':'v1.1.0; no local tag; remote ref lookup failed TLS. Local verified E2 artifacts are authoritative.',
        'architecture':'768->256 projection; two channel-only LayerNorm, 3x3 depthwise, 1x1 256->832->256 GELU residual blocks; no graph or cross-token attention',
        'expansion_selection':'Integer width minimizing absolute trainable-count difference only; no E10 Validation outcome inspected',
        'expansion':EXPANSION,'trainable_parameters':total,'full_trainable_parameters':FULL_COUNT,
        'relative_parameter_difference_percent':100*(total-FULL_COUNT)/FULL_COUNT,'module_parameter_counts':counts,
        'train_sources':289,'val_sources':62,'train_active_samples':578,'val_active_samples':124,
        'test_accessed':False,'casia_accessed':False,'e9_accessed':False,
        'validation_role':'Validation selects checkpoint and Youden J threshold; not a new blind test',
        'historical_metric_audit':historical,'input_hashes':input_hashes,
        'code_hashes':{p:sha(ROOT/p) for p in CODE_FILES},'tests_passed':checks,
        'superseded_test':'Original joint_heads_v2 dependency assertion uses a channel-constant perturbation removed by LayerNorm; preserved failure log. Existing dependency_r1 test corrects this known issue without changing model code.'}
    write_json(OUT/'design.json',design)
    write_json(OUT/'protocol.json',dict(asdict(protocol),experiment_id='E10-SCM-v1',training_seeds=SEEDS,
        model=MODEL_NAME,expansion=EXPANSION,trainable_parameters=COUNT,test_accessed=False,casia_accessed=False,e9_accessed=False))
    write_json(OUT/'status.json',{'state':'PREFLIGHT_PASS','completed_seeds':[]})
    print('PREFLIGHT PASS; design frozen; trainable count=',total,flush=True)


def verify_frozen():
    design=json.loads((OUT/'design.json').read_text())
    for rel,expected in {**design['input_hashes'],**design['code_hashes']}.items():
        path=ROOT/rel
        if not path.is_file() or sha(path)!=expected:
            raise RuntimeError(f'BLOCKED: frozen file missing/changed: {path}; restore the recorded artifact, do not substitute results')
    return design


@__import__('torch').no_grad()
def evaluate(model,loader,joint):
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
    model.eval()
    rows=[]
    for batch in loader:
        received,payload,mask,target=joint.move_batch_to_device(batch)
        output=model(received,payload)
        probability=float(torch.sigmoid(output['detection_logits'])[0])
        loc=torch.sigmoid(model.localization_logits_256(output['localization_grid_logits']))>=.5
        truth=mask>=.5
        tp,fp,fn,tn=[int(x.sum()) for x in [loc&truth,loc&~truth,~loc&truth,~loc&~truth]]
        key=batch['sample_key'][0]
        rows.append(dict(sample_key=key,source_id=key.split('|')[1],source_path=batch['source_path'][0],
            attack=batch['attack'][0],detection_label=int(target[0]),detection_probability=probability,
            localization_tp=tp,localization_fp=fp,localization_fn=fn,localization_tn=tn,
            localization_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            localization_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.))
    labels=np.array([r['detection_label'] for r in rows])
    probabilities=np.array([r['detection_probability'] for r in rows])
    fpr,tpr,thresholds=roc_curve(labels,probabilities)
    finite=np.where(np.isfinite(thresholds))[0]
    threshold=float(thresholds[finite[np.argmax((tpr-fpr)[finite])]])
    tn,fp,fn,tp=map(int,confusion_matrix(labels,probabilities>=threshold,labels=[0,1]).ravel())
    ltp,lfp,lfn,ltn=[sum(r['localization_'+k] for r in rows) for k in ['tp','fp','fn','tn']]
    return dict(detection_auc=float(roc_auc_score(labels,probabilities)),detection_threshold=threshold,
        detection_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,detection_f1=2*tp/(2*tp+fp+fn),
        detection_sensitivity=tp/(tp+fn),detection_specificity=tn/(tn+fp),
        detection_tn=tn,detection_fp=fp,detection_fn=fn,detection_tp=tp,
        localization_threshold=.5,localization_pixel_f1=2*ltp/(2*ltp+lfp+lfn),
        localization_pixel_iou=ltp/(ltp+lfp+lfn),localization_tp=ltp,localization_fp=lfp,
        localization_fn=lfn,localization_tn=ltn,prediction_rows=rows)


def verify_completed(seed):
    folder=OUT/str(seed)
    if not (folder/'lineage.json').exists():
        return False
    lineage=json.loads((folder/'lineage.json').read_text())
    for filename,expected in lineage['artifact_hashes'].items():
        if sha(folder/filename)!=expected:
            raise RuntimeError(f'Completed E10 seed artifact changed: {folder/filename}')
    return True


def run_seed(seed):
    import torch
    import pandas as pd
    from preprocessing.active_sspg_dataset import ActiveSSPGDataset
    from models.sspg_spatial_mixer_matched_v2 import SSPGSpatialMixerMatchedV2
    design=verify_frozen()
    if verify_completed(seed):
        print('Completed seed verified; skipped:',seed,flush=True)
        return
    folder=OUT/str(seed)
    if folder.exists():
        raise RuntimeError(f'Incomplete seed folder exists: {folder}; refusing to overwrite or silently rerun')
    folder.mkdir()
    joint,protocol,digest=configure(seed)
    install_access_guard(joint,folder/'image_access.json')
    joint.set_global_seed(seed)
    manifest=pd.read_csv(joint.MANIFEST_PATH)
    train_records=joint.build_source_records(manifest,'train')
    val_records=joint.build_source_records(manifest,'val')
    joint.audit_train_val_sources(train_records,val_records)
    train_ds=ActiveSSPGDataset(train_records,base_seed=protocol.train_seed)
    val_ds=ActiveSSPGDataset(val_records,base_seed=protocol.val_seed)
    train_loader=joint.build_loader(train_ds,True,seed)
    val_loader=joint.build_loader(val_ds,False,seed)
    joint.BEST_CHECKPOINT=folder/'best.pth'
    joint.HISTORY_PATH=folder/'training_log.csv'
    write_json(folder/'training_protocol.json',dict(asdict(protocol),model=MODEL_NAME,
        expected_trainable_parameters=COUNT,protocol_sha256=digest,test_accessed=False,casia_accessed=False,e9_accessed=False))
    model=SSPGSpatialMixerMatchedV2(str(DEC)).cuda()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==COUNT
    optimizer=joint.build_optimizer(model)
    history=[]
    best_loss=float('inf')
    best_epoch=None
    stale=0
    started=time.time()
    for epoch in range(1,protocol.max_epochs+1):
        print(f'E10 seed {seed}: epoch {epoch:02d}/{protocol.max_epochs}',flush=True)
        train=joint.train_one_epoch(model,train_loader,optimizer,epoch)
        val=joint.validate_one_epoch(model,val_loader)
        improved=val['total_loss']<best_loss-protocol.min_delta
        if improved:
            best_loss=float(val['total_loss'])
            best_epoch=epoch
            stale=0
            joint.save_checkpoint_atomic(model,optimizer,epoch,best_loss,digest)
        else:
            stale+=1
        history.append(dict(epoch=epoch,**{f'train_{k}':v for k,v in train.items()},
            **{f'val_{k}':v for k,v in val.items()},checkpoint_saved=int(improved)))
        joint.save_history(history)
        write_json(folder/'progress.json',dict(seed=seed,epoch=epoch,best_epoch=best_epoch,
            best_validation_joint_loss=best_loss,stale=stale,elapsed_seconds=time.time()-started))
        print(f'  val_joint={val["total_loss"]:.8f}; best_epoch={best_epoch}; stale={stale}',flush=True)
        if stale>=protocol.early_stopping_patience:
            break
    checkpoint=torch.load(folder/'best.pth',map_location='cuda',weights_only=False)
    assert checkpoint['protocol_sha256']==digest
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    result=evaluate(model,val_loader,joint)
    pd.DataFrame(result.pop('prediction_rows')).to_csv(folder/'val_predictions.csv',index=False)
    result.update(experiment_id=protocol.experiment_id,model=MODEL_NAME,seed=seed,
        protocol_sha256=digest,trainable_parameters=COUNT,best_epoch=best_epoch,
        best_val_joint_loss=best_loss,epochs_completed=len(history),training_seconds=time.time()-started,
        train_sources=len(train_records),val_sources=len(val_records),train_active_samples=len(train_ds),
        val_active_samples=len(val_ds),test_accessed=False,casia_accessed=False,e9_accessed=False)
    pd.DataFrame([result]).to_csv(folder/'metrics.csv',index=False)
    write_json(folder/'metrics.json',result)
    files=['best.pth','training_log.csv','training_protocol.json','val_predictions.csv','metrics.csv','metrics.json']
    write_json(folder/'lineage.json',dict(experiment_id=protocol.experiment_id,seed=seed,
        protocol_sha256=digest,decoder_sha256=design['input_hashes'][str(DEC.relative_to(ROOT))],
        artifact_hashes={p:sha(folder/p) for p in files},test_accessed=False,casia_accessed=False,e9_accessed=False))
    print(json.dumps(result,indent=2),flush=True)


def pipeline():
    if not (OUT/'design.json').exists():
        preflight()
    verify_frozen()
    completed=[]
    for seed in SEEDS:
        if verify_completed(seed):
            completed.append(seed)
            continue
        write_json(OUT/'status.json',dict(state='RUNNING',seed=seed,completed_seeds=completed,pid=os.getpid()))
        log=OUT/f'{seed}.log'
        if log.exists():
            raise RuntimeError(f'Existing unfinished log: {log}; inspect instead of overwriting')
        env=dict(os.environ,PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',PYTHONHASHSEED='42')
        with log.open('w',encoding='utf-8') as f:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--seed',str(seed)],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
            write_json(OUT/'status.json',dict(state='RUNNING',seed=seed,completed_seeds=completed,
                pid=os.getpid(),training_pid=process.pid))
            code=process.wait()
        if code or not verify_completed(seed):
            write_json(OUT/'status.json',dict(state='FAILED',seed=seed,completed_seeds=completed,returncode=code))
            raise RuntimeError(f'E10 seed failed: {seed}; see {log}')
        completed.append(seed)
    write_json(OUT/'status.json',dict(state='TRAINING_COMPLETE',completed_seeds=completed))
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_e10_spatial_mixer_v1.py')],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--preflight',action='store_true')
    selection.add_argument('--seed',type=int,choices=SEEDS)
    selection.add_argument('--all',action='store_true')
    selection.add_argument('--resume',action='store_true')
    selection.add_argument('--test-script',choices=['tests/test_e10_spatial_mixer_v1.py',
        'tests/test_e53r_training_protocol_v1.py','tests/test_e53r_joint_heads_dependency_r1.py',
        'tests/test_e53r_active_dataset_integration.py'])
    args=parser.parse_args()
    try:
        if args.test_script:
            import runpy
            import torch
            import training.train_e53r_joint_v1 as joint
            torch.set_num_threads(4)
            install_access_guard(joint,OUT/(Path(args.test_script).stem+'_access.json'))
            test_path=str(ROOT/args.test_script)
            sys.argv=[test_path]
            runpy.run_path(test_path,run_name='__main__')
        elif args.preflight:
            preflight()
        elif args.seed:
            run_seed(args.seed)
        else:
            pipeline()
    except Exception as exc:
        write_json(OUT/'last_failure.json',dict(message=str(exc),traceback=traceback.format_exc(),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        if args.preflight:
            write_json(OUT/'status.json',dict(state='BLOCKED',message=str(exc)))
        raise
