"""Build an additive, byte-preserving public E10/C1 evidence supplement."""
from pathlib import Path
import csv,hashlib,json,shutil,zipfile,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PUBLIC=ROOT/'output/github_e10_c1_20261014'
DEST=PUBLIC/'results/supplement_e10_c1_20261014'
ASSETS=ROOT/'output/github_e10_c1_assets_20261014'
BRIDGE=[]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def copy(source,target):
    s=ROOT/source;d=DEST/target;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d);assert sha(s)==sha(d)
    BRIDGE.append(dict(source=str(s.relative_to(ROOT)).replace('\\','/'),published=str(d.relative_to(PUBLIC)).replace('\\','/'),sha256=sha(s),bytes=s.stat().st_size))
def main():
    from preprocessing.active_sspg_dataset import stable_seed,sample_box,sample_copy_move_geometry,ATTACK_GEOMETRY_SEED_OFFSET
    assert (PUBLIC/'.git').is_dir();DEST.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    e10=ROOT/'results/e10_spatial_mixer_v1'
    for name in ['audit_report.md','design.json','protocol.json','status.json','RUN_COMMANDS.md','environment_freeze.txt','e10_comparison_summary.csv','e10_paired_differences.csv','e10_per_seed_metrics.csv','paper_e10_results.tex','figure_e10_comparison.pdf','figure_e10_comparison.svg','figure_e10_comparison.png','figure_e10_comparison.alignment.json','figure_e10_comparison.collision.json','figure_e10_parameters.pdf','figure_e10_parameters.svg','figure_e10_parameters.png','figure_e10_parameters.alignment.json','figure_e10_parameters.collision.json','figure_qa.log']:
        copy(e10.relative_to(ROOT)/name,Path('e10')/name)
    for seed in [20261001,20261002,20261003]:
        p=e10/str(seed);lineage=json.loads((p/'lineage.json').read_text())
        for name,h in lineage['artifact_hashes'].items():assert sha(p/name)==h
        for name in ['lineage.json','training_protocol.json','training_log.csv','metrics.csv','metrics.json','val_predictions.csv','image_access.json','progress.json']:
            copy(p.relative_to(ROOT)/name,Path('e10')/str(seed)/name)
        for kind in ['full','no_graph','no_provenance']:
            folder=ROOT/f'results/clean_revision_v2/budget142_e2/{kind}/{seed}'
            for name in ['metrics.csv','val_predictions.csv','lineage.json','protocol.json']:
                copy(folder.relative_to(ROOT)/name,Path('e2_comparators')/kind/str(seed)/name)
        folder=ROOT/f'results/clean_revision_v2/budget142_param_matched_no_graph_e2/{seed}'
        for name in ['metrics.csv','val_predictions.csv','lineage.json','training_protocol.json']:
            copy(folder.relative_to(ROOT)/name,Path('e2_comparators/matched_no_graph')/str(seed)/name)
    efficiency=ROOT/'results/e10_inference_efficiency_v1'
    for name in ['efficiency_summary.csv','all_per_image_timings.csv','per_repetition_statistics.csv','inference_efficiency_report.md','paper_inference_efficiency.tex','protocol.json','RUN_COMMANDS.md','status.json']:
        copy(efficiency.relative_to(ROOT)/name,Path('inference_efficiency')/name)
    for model in ['FULL','NO_GRAPH','MATCHED_NO_GRAPH','SPATIAL_MIXER_MATCHED']:
        for name in ['per_image_timings.csv','summary.json','image_access.json']:
            copy(efficiency.relative_to(ROOT)/model/name,Path('inference_efficiency')/model/name)
    c1=ROOT/'results/clean_revision_v2/budget142_stage_c'
    for name in ['protocol.json','status.json']:copy(c1.relative_to(ROOT)/name,Path('c1')/name)
    archive=ASSETS/'SSPG-Net-C1-dense-predictions-20261014.zip';dense_index=[];summary=[];allcounts=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for condition,offset in [('splicing',3000001),('telea_inpainting',3000002),('copy_move_jpeg85',3000003)]:
            p=c1/condition;metrics=json.loads((p/'metrics.json').read_text());pred=pd.read_csv(p/'predictions.csv');assert len(pred)==126 and pred.detection_label.sum()==63
            for name in ['predictions.csv','metrics.json']:copy(p.relative_to(ROOT)/name,Path('c1')/condition/name)
            counts=[]
            for i,r in enumerate(pred.itertuples()):
                key,attack,label=r.sample_key.rsplit('|',2);assert label==str(r.detection_label) and attack==condition
                gt=np.zeros((256,256),dtype=np.uint8)
                if r.detection_label:
                    rng=np.random.default_rng(stable_seed(20260924,key)+ATTACK_GEOMETRY_SEED_OFFSET+offset)
                    geom=sample_copy_move_geometry(rng) if condition=='copy_move_jpeg85' else dict(zip(['top','left','height','width'],sample_box(rng)))
                    if condition=='copy_move_jpeg85':geom.update(top=geom['destination_top'],left=geom['destination_left'],height=geom['box_h'],width=geom['box_w'])
                    gt[geom['top']:geom['top']+geom['height'],geom['left']:geom['left']+geom['width']]=1
                probpath=p/'prediction_masks'/r.prediction_mask;prob=np.load(probpath);assert prob.dtype==np.float32 and prob.shape==(256,256) and np.isfinite(prob).all()
                pm=prob>=.5;truth=gt.astype(bool);tp=int((pm&truth).sum());fp=int((pm&~truth).sum());fn=int((~pm&truth).sum());tn=int((~pm&~truth).sum())
                record=dict(condition=condition,row_index=i,sample_key=r.sample_key,source_path=r.source_path,detection_label=r.detection_label,pixel_tp=tp,pixel_fp=fp,pixel_fn=fn,pixel_tn=tn,gt_area_fraction=float(gt.mean()),prediction_mask=r.prediction_mask,prediction_sha256=sha(probpath),gt_pixel_sha256=hashlib.sha256(gt.tobytes()).hexdigest())
                counts.append(record);allcounts.append(record)
                member=f'c1/{condition}/prediction_masks/{r.prediction_mask}';z.write(probpath,member)
                dense_index.append(dict(path=member,sha256=sha(probpath),bytes=probpath.stat().st_size,kind='archived_float32_probability'))
                gtfile=ASSETS/'derived_gt'/condition/r.prediction_mask;gtfile.parent.mkdir(parents=True,exist_ok=True);np.save(gtfile,gt)
                member=f'c1/{condition}/gt_masks/{r.prediction_mask}';z.write(gtfile,member);dense_index.append(dict(path=member,sha256=sha(gtfile),bytes=gtfile.stat().st_size,kind='deterministically_reconstructed_binary_GT'))
            for k in ['tp','fp','fn','tn']:assert sum(r['pixel_'+k] for r in counts)==metrics['localization'][k],(condition,k)
            pd.DataFrame(counts).to_csv(DEST/'c1'/condition/'per_sample_pixel_counts.csv',index=False)
            summary.append(dict(condition=condition,sources=63,active_samples=126,**{'detection_'+k:v for k,v in metrics['detection'].items()},**{'localization_'+k:v for k,v in metrics['localization'].items()}))
    pd.DataFrame(summary).to_csv(DEST/'c1/C1_summary.csv',index=False);pd.DataFrame(allcounts).to_csv(DEST/'c1/C1_per_sample_pixel_counts.csv',index=False)
    write(DEST/'c1/DENSE_ARRAY_INDEX.json',dense_index)
    sources=pd.read_csv(ROOT/'results/submission_addon/00_audit/audited_source_identity.csv');sources=sources[(sources.dataset=='IMD2020')&(sources.split=='test')];assert len(sources)==63;sources.to_csv(DEST/'c1/source_identity.csv',index=False)
    weight_index=[];weight_zip=ASSETS/'SSPG-Net-E10-Spatial-Mixer-three-seeds-20261014.zip'
    with zipfile.ZipFile(weight_zip,'w',zipfile.ZIP_STORED) as z:
        for seed in [20261001,20261002,20261003]:
            p=e10/str(seed)/'best.pth';relative=str(p.relative_to(ROOT)).replace('\\','/');weight_index.append(dict(path=relative,seed=seed,bytes=p.stat().st_size,sha256=sha(p)));z.write(p,relative)
    write(DEST/'WEIGHT_INDEX.json',weight_index)
    assets=[]
    for p in [archive,weight_zip]:assets.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),download_url='https://github.com/wunan-snsy/SSPG-Net/releases/download/e10-c1-20261014/'+p.name))
    write(DEST/'ASSET_INDEX.json',assets);(ASSETS/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in assets))
    # Preserve the exact executed source separately from repository entry points.
    scripts=['models/sspg_spatial_mixer_matched_v2.py','scripts/run_e10_spatial_mixer_v1.py','scripts/summarize_e10_spatial_mixer_v1.py','scripts/finalize_e10_delivery_v1.py','scripts/benchmark_e10_inference_efficiency_v1.py','scripts/run_budget142_stage_c_v2.py','scripts/eval_test_final.py','scripts/run_clean_stage_a_v2.py','tests/test_e10_spatial_mixer_v1.py','scripts/publish_e10_c1_package_20261014.py']
    for rel in scripts:
        copy(rel,Path('source_frozen')/rel)
        target=PUBLIC/rel
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
        elif sha(target)!=sha(ROOT/rel):print('Existing repository source retained; historical copy archived:',rel)
    write(DEST/'PROVENANCE_BRIDGE.json',BRIDGE)
    print('Package prepared',DEST,'\n',json.dumps(assets,indent=2))
if __name__=='__main__':main()
