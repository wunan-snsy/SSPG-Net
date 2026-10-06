import torch
from models.sspg_net import SSPGNet
from preprocessing.patch import patch_targets
from losses.detection_loss import detection_loss
from losses.localization_loss import localization_loss
from losses.provenance_loss import provenance_loss

def run_smoke_test(cfg):
    torch.manual_seed(cfg.get('seed',42)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SSPGNet(cfg).to(device).train(); B=cfg['batch_size']; S=cfg['image_size']
    x=torch.rand(B,3,S,S,device=device); out=model(x)
    det_t=torch.ones(B,device=device); loc_t=patch_targets(out['mask'],cfg['patch_size'])
    # Synthetic graph labels exist solely to test the provenance-loss/backward path.
    prov_t=torch.zeros_like(out['provenance_logits']); e=out['edge_index']; side=S//cfg['patch_size']; m=loc_t>0.5
    for b in range(B): prov_t[b]=(m[b,e[0]] & m[b,e[1]]).float()
    ld=detection_loss(out['detection_logits'],det_t); ll=localization_loss(out['localization_logits'],loc_t); lp=provenance_loss(out['provenance_logits'],prov_t)
    w=cfg['loss_weights']; loss=w['detection']*ld+w['localization']*ll+w['provenance']*lp; loss.backward()
    print('=== SSPG-Net Reference Implementation V1.0 ==='); print('Device:',device); print('Input:',tuple(x.shape)); print('Stego:',tuple(out['stego'].shape)); print('Residual:',tuple(out['residual'].shape)); print('Patch features:',tuple(out['patch_features'].shape)); print('Graph nodes:',out['patch_features'].shape[1],'edges:',out['edge_index'].shape[1]); print('Detection logits:',tuple(out['detection_logits'].shape)); print('Localization logits:',tuple(out['localization_logits'].shape)); print('Provenance logits:',tuple(out['provenance_logits'].shape)); print(f'Loss: {loss.item():.6f} (det={ld.item():.4f}, loc={ll.item():.4f}, prov={lp.item():.4f})')
    if device.type=='cuda': print(f'GPU allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MiB')
    print('Backward: PASS'); print('SMOKE TEST: PASS'); return True
