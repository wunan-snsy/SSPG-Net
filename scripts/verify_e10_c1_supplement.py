"""Verify published E10/C1 evidence without training, images, PyTorch or CUDA."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics,zipfile
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'results/supplement_e10_c1_20261014'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def obj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-9),(a,b)
def detection(pred,threshold):
    positive=[float(r['detection_probability']) for r in pred if int(r['detection_label'])==1]
    negative=[float(r['detection_probability']) for r in pred if int(r['detection_label'])==0]
    auc=sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
    tp=sum(a>=threshold for a in positive);fn=len(positive)-tp;fp=sum(a>=threshold for a in negative);tn=len(negative)-fp
    return dict(auc=auc,tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/(2*tp+fp+fn),balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets-dir',type=Path);a=parser.parse_args()
    sums=(RELEASE/'SHA256SUMS').read_text().splitlines()
    for line in sums:
        expected,relative=line.split('  ',1);assert sha(RELEASE/relative)==expected,relative
    for r in obj(RELEASE/'PROVENANCE_BRIDGE.json'):assert sha(ROOT/r['published'])==r['sha256'],r['published']
    model_results={}
    for model in ['full','no_graph','no_provenance','matched_no_graph','spatial_mixer']:
        values=[]
        for seed in [20261001,20261002,20261003]:
            folder=RELEASE/'e10'/str(seed) if model=='spatial_mixer' else RELEASE/'e2_comparators'/model/str(seed)
            metric=rows(folder/'metrics.csv')[0];pred=rows(folder/'val_predictions.csv');assert len(pred)==124
            assert len({r['sample_key'] for r in pred})==62
            result=detection(pred,float(metric['detection_threshold']));close(result['auc'],metric['detection_auc']);close(result['f1'],metric['detection_f1']);close(result['balanced_accuracy'],metric['detection_balanced_accuracy'])
            for k in ['tp','fp','fn','tn']:close(result[k],metric['detection_'+k])
            tp,fp,fn,tn=[int(metric['localization_'+k]) for k in ['tp','fp','fn','tn']]
            close(2*tp/(2*tp+fp+fn),metric['localization_pixel_f1']);close(tp/(tp+fp+fn),metric['localization_pixel_iou']);assert tp+fp+fn+tn==124*65536
            if model=='spatial_mixer':
                for k,value in zip(['tp','fp','fn','tn'],[tp,fp,fn,tn]):assert sum(int(r['localization_'+k]) for r in pred)==value
                lineage=obj(folder/'lineage.json')
                for name,h in lineage['artifact_hashes'].items():
                    if name!='best.pth':assert sha(folder/name)==h
                assert metric['test_accessed'].lower()=='false'
            values.append(float(metric['localization_pixel_f1']))
        model_results[model]=values
    summary={r['model']:r for r in rows(RELEASE/'e10/e10_comparison_summary.csv')}
    for model,display in [('full','FULL'),('no_graph','NO_GRAPH'),('matched_no_graph','Matched NO_GRAPH'),('spatial_mixer','SPATIAL_MIXER_MATCHED')]:
        close(statistics.mean(model_results[model]),summary[display]['localization_pixel_f1_mean']);close(statistics.stdev(model_results[model]),summary[display]['localization_pixel_f1_sd'])
    paired=[f-m for f,m in zip(model_results['full'],model_results['spatial_mixer'])]
    for r,v in zip(rows(RELEASE/'e10/e10_paired_differences.csv'),paired):close(r['full_minus_spatial_localization_pixel_f1'],v)
    close(statistics.mean(paired),.049153399572824015)
    for condition in ['splicing','telea_inpainting','copy_move_jpeg85']:
        folder=RELEASE/'c1'/condition;metric=obj(folder/'metrics.json');pred=rows(folder/'predictions.csv');assert len(pred)==126
        result=detection(pred,float(metric['detection']['threshold']))
        close(result['auc'],metric['detection']['roc_auc']);close(result['f1'],metric['detection']['f1']);close(result['balanced_accuracy'],metric['detection']['balanced_accuracy'])
        count=rows(folder/'per_sample_pixel_counts.csv');assert len(count)==126
        for k in ['tp','fp','fn','tn']:assert sum(int(r['pixel_'+k]) for r in count)==metric['localization'][k]
        tp,fp,fn,tn=[metric['localization'][k] for k in ['tp','fp','fn','tn']];assert tp+fp+fn+tn==126*65536
        close(2*tp/(2*tp+fp+fn),metric['localization']['pixel_f1']);close(tp/(tp+fp+fn),metric['localization']['pixel_iou'])
    efficiency=rows(RELEASE/'inference_efficiency/efficiency_summary.csv')
    assert len({r['input_tensor_sha256'] for r in efficiency})==1
    for r in efficiency:
        t=rows(RELEASE/'inference_efficiency'/r['model']/'per_image_timings.csv');assert len(t)==372
        close(statistics.mean(float(x['wall_ms']) for x in t),r['wall_mean_ms']);close(statistics.mean(float(x['gpu_interval_ms']) for x in t),r['gpu_mean_ms'])
        assert int(r['batch_size'])==1 and r['autocast'].lower()=='false'
    if a.assets_dir:
        for asset in obj(RELEASE/'ASSET_INDEX.json'):
            p=a.assets_dir/asset['name'];assert p.stat().st_size==asset['bytes'] and sha(p)==asset['sha256']
            if 'dense-predictions' in p.name:
                with zipfile.ZipFile(p) as z:
                    idx=obj(RELEASE/'c1/DENSE_ARRAY_INDEX.json');assert len(idx)==len(z.namelist())==756
                    for item in idx:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
            else:
                with zipfile.ZipFile(p) as z:
                    for item in obj(RELEASE/'WEIGHT_INDEX.json'):
                        with z.open(item['path']) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    print(f'PASS: {len(sums)} frozen package files; 15 Validation model/seed records; C1 378 prediction rows; 1,488 timing measurements.')
    print(f'FULL minus Spatial Mixer Pixel F1: {statistics.mean(paired):.10f} +/- {statistics.stdev(paired):.10f} sample SD (n=3).')
    if a.assets_dir:print('PASS: both release archives; 378 archived probability maps, 378 reconstructed GT maps, three checkpoint byte hashes.')
if __name__=='__main__':main()
