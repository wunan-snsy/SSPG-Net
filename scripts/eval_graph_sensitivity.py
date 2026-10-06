"""E7 kNN graph-topology sensitivity using frozen model weights."""
from pathlib import Path
import json, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import training.train_e53r_joint_v1 as joint
from preprocessing.active_sspg_dataset import ActiveSSPGDataset
from models.sspg_joint_model_v2 import SSPGJointModelV2
from training.e53r_training_protocol_v1 import PROTOCOL

OUT=ROOT/'results'/'graph_sensitivity'; DEVICE=joint.DEVICE

@torch.no_grad()
def evaluate(model,ds,det_thr):
    ys=[]; probs=[]; tp=fp=fn=0; edge_counts=[]
    for i in range(len(ds)):
        b=ds[i]; out=model(b['received'][None].to(DEVICE),b['payload'][None].to(DEVICE))
        ys.append(int(b['detection_label'])); probs.append(float(torch.sigmoid(out['detection_logits'])[0]))
        pred=torch.sigmoid(model.localization_logits_256(out['localization_grid_logits']))>=.5
        truth=b['mask'][None].to(DEVICE)>=.5; tp+=int((pred&truth).sum()); fp+=int((pred&~truth).sum()); fn+=int((~pred&truth).sum())
        edge_counts.append(int(out['edge_index'][0].shape[1]))
    ys=np.asarray(ys); probs=np.asarray(probs); pred=(probs>=det_thr).astype(int)
    return {'mean_edges':float(np.mean(edge_counts)),'roc_auc':float(roc_auc_score(ys,probs)),
            'balanced_accuracy':float(balanced_accuracy_score(ys,pred)),'detection_f1':float(f1_score(ys,pred)),
            'localization_pixel_f1':2*tp/(2*tp+fp+fn),'localization_pixel_iou':tp/(tp+fp+fn)}

def main():
    cfg=json.loads((ROOT/'configs'/'e7_graph_sensitivity.json').read_text())
    records=joint.build_source_records(pd.read_csv(joint.MANIFEST_PATH),cfg['split']); ds=ActiveSSPGDataset(records,PROTOCOL.val_seed)
    model=SSPGJointModelV2(str(joint.STEGO_CHECKPOINT),PROTOCOL.adapter_bottleneck,PROTOCOL.fusion_hidden_dim,PROTOCOL.knn_k,PROTOCOL.gnn_hidden_dim,PROTOCOL.gnn_depth).to(DEVICE)
    model.load_state_dict(torch.load(joint.BEST_CHECKPOINT,map_location=DEVICE,weights_only=False)['model_state_dict']); model.eval()
    det=float(pd.read_csv(ROOT/'results'/'release_v1.0.0'/'joint'/'e53r_joint_v1_summary.csv').iloc[0].detection_threshold)
    rows=[]
    for k in cfg['knn_k_values']:
        print(f'knn_k={k}',flush=True); model.frontend.graph_builder.knn_k=int(k); rows.append({'knn_k':int(k),'gnn_depth':cfg['gnn_depth'],**evaluate(model,ds,det)})
    OUT.mkdir(parents=True,exist_ok=True); table=pd.DataFrame(rows); table.to_csv(OUT/'Table_graph_sensitivity.csv',index=False); (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(table.knn_k,table.roc_auc,'o-',label='ROC-AUC'); ax.plot(table.knn_k,table.detection_f1,'s-',label='Detection F1'); ax.plot(table.knn_k,table.localization_pixel_f1,'^-',label='Pixel F1'); ax.set_xlabel('kNN neighbours (k)'); ax.set_ylabel('Score'); ax.set_xticks(table.knn_k); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/'Figure_graph_sensitivity.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    print(table.to_string(index=False))
if __name__=='__main__': main()
