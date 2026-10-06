"""E5 benign post-processing robustness evaluation."""
from pathlib import Path
from io import BytesIO
import json, sys
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torchvision.transforms.functional import to_pil_image, to_tensor

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import training.train_e53r_joint_v1 as joint
from preprocessing.active_sspg_dataset import ActiveSSPGDataset
from models.sspg_joint_model_v2 import SSPGJointModelV2
from training.e53r_training_protocol_v1 import PROTOCOL
OUT=ROOT/'results'/'robustness'; DEVICE=joint.DEVICE

def process(x,c):
    if c['type']=='jpeg':
        b=BytesIO(); to_pil_image(x.cpu()).save(b,format='JPEG',quality=c['quality']); b.seek(0)
        return to_tensor(Image.open(b).convert('RGB'))
    if c['type']=='resize':
        s=c['scale']; y=F.interpolate(x[None],scale_factor=s,mode='bilinear',align_corners=False,antialias=True)
        return F.interpolate(y,size=(256,256),mode='bilinear',align_corners=False,antialias=True)[0]
    return to_tensor(to_pil_image(x.cpu()).filter(ImageFilter.GaussianBlur(c['sigma'])))

@torch.no_grad()
def run(model,ds,c,det_thr):
    ys=[]; ps=[]; bits=tot=tp=fp=fn=0
    for i in range(len(ds)):
        b=ds[i]; x=process(b['received'],c)[None].to(DEVICE); payload=b['payload'][None].to(DEVICE)
        mask=(b['mask'][None].to(DEVICE)>=.5); out=model(x,payload)
        prob=float(torch.sigmoid(out['detection_logits'])[0]); ys.append(int(b['detection_label'])); ps.append(prob)
        decoded=(torch.sigmoid(out['decoder_logits'])>=.5); truth=(payload>=.5); bits+=int((decoded==truth).sum()); tot+=truth.numel()
        pred=torch.sigmoid(model.localization_logits_256(out['localization_grid_logits']))>=.5
        tp+=int((pred&mask).sum()); fp+=int((pred&~mask).sum()); fn+=int((~pred&mask).sum())
    ys=np.asarray(ys); ps=np.asarray(ps); pred=(ps>=det_thr)
    authentic=ys==0; den=2*tp+fp+fn; ioden=tp+fp+fn
    return {'bit_accuracy':bits/tot,'ber':1-bits/tot,'roc_auc':float(roc_auc_score(ys,ps)),
        'false_positive_rate':float(pred[authentic].mean()),'localization_pixel_f1':2*tp/den if den else 1.,
        'localization_pixel_iou':tp/ioden if ioden else 1.}

def main():
    cfg=json.load(open(ROOT/'configs'/'e5_benign_robustness.json'))
    df=pd.read_csv(joint.MANIFEST_PATH); rec=joint.build_source_records(df,'val'); ds=ActiveSSPGDataset(rec,PROTOCOL.val_seed)
    model=SSPGJointModelV2(str(joint.STEGO_CHECKPOINT),PROTOCOL.adapter_bottleneck,PROTOCOL.fusion_hidden_dim,PROTOCOL.knn_k,PROTOCOL.gnn_hidden_dim,PROTOCOL.gnn_depth).to(DEVICE)
    ck=torch.load(joint.BEST_CHECKPOINT,map_location=DEVICE,weights_only=False); model.load_state_dict(ck['model_state_dict']); model.eval()
    det=float(pd.read_csv(ROOT/'results'/'release_v1.0.0'/'joint'/'e53r_joint_v1_summary.csv').iloc[0].detection_threshold)
    rows=[]
    for c in cfg['conditions']:
        label=f"{c['type']}_{c.get('quality',c.get('scale',c.get('sigma')))}"; print(label,flush=True)
        rows.append({'condition':label,**c,**run(model,ds,c,det)})
    OUT.mkdir(parents=True,exist_ok=True); out=pd.DataFrame(rows); out.to_csv(OUT/'Table_benign_attack.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); print(out.to_string(index=False))
if __name__=='__main__': main()
