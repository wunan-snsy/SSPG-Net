"""E6 embedding-strength sensitivity using the frozen joint model."""
from pathlib import Path
import json, math, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import preprocessing.active_sspg_dataset as active
import training.train_e53r_joint_v1 as joint
from training.train_stego_structured_v33 import payload_to_spatial_map
from models.sspg_joint_model_v2 import SSPGJointModelV2
from training.e53r_training_protocol_v1 import PROTOCOL

OUT=ROOT/'results'/'alpha_sensitivity'; DEVICE=joint.DEVICE

def embed_at_alpha(images,payload,alpha):
    spatial=payload_to_spatial_map(payload)
    residual=(spatial*2.0-1.0).repeat(1,3,1,1)
    return torch.clamp(images+alpha*residual,0.0,1.0)

@torch.no_grad()
def evaluate(model,records,alpha,det_thr):
    original_embed=active.embed_payload
    active.embed_payload=lambda images,payload: embed_at_alpha(images,payload,alpha)
    try:
        ds=active.ActiveSSPGDataset(records,PROTOCOL.val_seed)
        ys=[]; probs=[]; correct=total=tp=fp=fn=0; mse_sum=pixels=0
        for i in range(len(ds)):
            b=ds[i]; x=b['received'][None].to(DEVICE); payload=b['payload'][None].to(DEVICE)
            out=model(x,payload); ys.append(int(b['detection_label'])); probs.append(float(torch.sigmoid(out['detection_logits'])[0]))
            decoded=torch.sigmoid(out['decoder_logits'])>=.5; truth=payload>=.5
            correct+=int((decoded==truth).sum()); total+=truth.numel()
            pred=torch.sigmoid(model.localization_logits_256(out['localization_grid_logits']))>=.5
            mask=b['mask'][None].to(DEVICE)>=.5; tp+=int((pred&mask).sum()); fp+=int((pred&~mask).sum()); fn+=int((~pred&mask).sum())
            source=active.load_rgb_256(records[i//2]['image_path']).to(DEVICE)
            protected=embed_at_alpha(source[None],payload,alpha)[0]
            mse_sum+=float(((protected-source)**2).sum()); pixels+=source.numel()
        ys=np.asarray(ys); probs=np.asarray(probs); predicted=probs>=det_thr; authentic=ys==0
        mse=mse_sum/pixels
        return {'alpha':alpha,'psnr_db':float('inf') if mse==0 else -10*math.log10(mse),
                'bit_accuracy':correct/total,'ber':1-correct/total,'roc_auc':float(roc_auc_score(ys,probs)),
                'false_positive_rate':float(predicted[authentic].mean()),
                'localization_pixel_f1':2*tp/(2*tp+fp+fn),'localization_pixel_iou':tp/(tp+fp+fn)}
    finally:
        active.embed_payload=original_embed

def main():
    cfg=json.loads((ROOT/'configs'/'e6_alpha_sensitivity.json').read_text())
    records=joint.build_source_records(pd.read_csv(joint.MANIFEST_PATH),cfg['split'])
    model=SSPGJointModelV2(str(joint.STEGO_CHECKPOINT),PROTOCOL.adapter_bottleneck,PROTOCOL.fusion_hidden_dim,PROTOCOL.knn_k,PROTOCOL.gnn_hidden_dim,PROTOCOL.gnn_depth).to(DEVICE)
    model.load_state_dict(torch.load(joint.BEST_CHECKPOINT,map_location=DEVICE,weights_only=False)['model_state_dict']); model.eval()
    det=float(pd.read_csv(ROOT/cfg['frozen_detection_threshold_source']).iloc[0].detection_threshold)
    rows=[]
    for alpha in cfg['alpha_values']:
        print(f'alpha={alpha}',flush=True); rows.append(evaluate(model,records,float(alpha),det))
    OUT.mkdir(parents=True,exist_ok=True); table=pd.DataFrame(rows); table.to_csv(OUT/'Table_alpha_sensitivity.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(table.alpha,table.bit_accuracy,'o-',label='Bit accuracy'); ax[0].plot(table.alpha,table.roc_auc,'s-',label='ROC-AUC'); ax[0].set_ylabel('Task score'); ax[0].legend(frameon=False)
    ax[1].plot(table.alpha,table.psnr_db,'o-',color='tab:green'); ax[1].set_ylabel('PSNR (dB)')
    for a in ax: a.set_xlabel('Embedding strength α'); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/'Figure_alpha_sensitivity.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    print(table.to_string(index=False))
if __name__=='__main__': main()
