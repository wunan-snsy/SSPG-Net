"""Pre-fixed exploratory unknown-attack evaluation of budget142 FULL."""
from io import BytesIO
from pathlib import Path
import json
import sys
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_clean_stage_a_v2 import gate, sha, write_json
import scripts.eval_test_final as ev
from models.sspg_joint_model_v2 import SSPGJointModelV2
from preprocessing.active_sspg_dataset import (
    canonical_sample_key, stable_seed, make_payload_deterministic,
    load_rgb_256, sample_box, apply_controlled_attack,
    ATTACK_GEOMETRY_SEED_OFFSET,
)
from training.train_stego_structured_v33 import embed_payload
from training.e53r_training_protocol_v1 import PROTOCOL
from training.train_e53r_joint_v1 import build_loader

OUT = ROOT / 'results/clean_revision_v2/budget142_stage_c'
CHECKPOINT = ROOT / 'checkpoints/clean_revision_v2/budget142_full_best.pth'
DECODER = ROOT / 'checkpoints/clean_revision_v2/decoder_budget142/best.pth'
VAL_SUMMARY = ROOT / 'results/clean_revision_v2/budget142_full/summary.csv'
ATTACKS = ('splicing', 'telea_inpainting', 'copy_move_jpeg85')
SEED = ev.TEST_SEED
JPEG_QUALITY = 85
DEVICE = ev.DEVICE


def to_uint8(tensor):
    return (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def to_tensor(array):
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0


class UnknownAttackDataset(Dataset):
    def __init__(self, records, attack):
        self.records = records
        self.attack = attack

    def __len__(self):
        return 2 * len(self.records)

    def __getitem__(self, index):
        source_i, manipulated = divmod(index, 2)
        source = self.records[source_i]
        key = canonical_sample_key(source)
        source_seed = stable_seed(SEED, key)
        original = load_rgb_256(source['image_path'])
        payload = make_payload_deterministic(source_seed)
        protected = embed_payload(original.unsqueeze(0), payload.unsqueeze(0)).squeeze(0)
        received = protected
        mask = torch.zeros(1, 256, 256)
        if manipulated:
            offset = {'splicing': 3000001, 'telea_inpainting': 3000002,
                      'copy_move_jpeg85': 3000003}[self.attack]
            rng = np.random.default_rng(source_seed + ATTACK_GEOMETRY_SEED_OFFSET + offset)
            if self.attack == 'copy_move_jpeg85':
                moved, mask = apply_controlled_attack(protected, 'copy_move', rng)
                image = Image.fromarray(to_uint8(moved), 'RGB')
                buffer = BytesIO()
                image.save(buffer, format='JPEG', quality=JPEG_QUALITY, subsampling=0)
                buffer.seek(0)
                received = to_tensor(np.asarray(Image.open(buffer).convert('RGB')))
            else:
                top, left, height, width = sample_box(rng)
                mask[:, top:top + height, left:left + width] = 1.0
                if self.attack == 'splicing':
                    donor_i = (source_i + 1) % len(self.records)
                    donor = load_rgb_256(self.records[donor_i]['image_path'])
                    received = protected.clone()
                    received[:, top:top + height, left:left + width] = donor[:, top:top + height, left:left + width]
                else:
                    cv_mask = (mask[0].numpy() * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(to_uint8(protected), cv2.COLOR_RGB2BGR)
                    restored = cv2.inpaint(bgr, cv_mask, 3, cv2.INPAINT_TELEA)
                    received = to_tensor(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        return {'received': received, 'payload': payload, 'mask': mask,
                'detection_label': torch.tensor(float(manipulated)),
                'attack': self.attack if manipulated else 'authentic',
                'source_path': source['image_path'],
                'sample_key': f'{key}|{self.attack}|{int(manipulated)}'}


def verify():
    gate()
    full_state = json.loads((ROOT / 'results/clean_revision_v2/budget142_full/status.json').read_text())
    dec_state = json.loads((ROOT / 'results/clean_revision_v2/decoder_budget142/status.json').read_text())
    if full_state['state'] != 'COMPLETE' or sha(CHECKPOINT) != full_state['checkpoint_sha256']:
        raise RuntimeError('FULL checkpoint changed')
    if dec_state['state'] != 'COMPLETE' or sha(DECODER) != dec_state['best_checkpoint_sha256']:
        raise RuntimeError('Decoder checkpoint changed')


def main():
    verify()
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite stage C output')
    records = ev.test_records()
    if len(records) != 63 or len({canonical_sample_key(r) for r in records}) != 63:
        raise RuntimeError('Expected 63 distinct Test source records')
    threshold = float(pd.read_csv(VAL_SUMMARY, float_precision='round_trip').iloc[0]['detection_threshold'])
    OUT.mkdir(parents=True)
    write_json(OUT / 'protocol.json', {
        'experiment_id': 'CLEAN-BUDGET142-STAGE-C-EXPLORATORY-V2',
        'checkpoint_sha256': sha(CHECKPOINT), 'decoder_sha256': sha(DECODER),
        'validation_summary_sha256': sha(VAL_SUMMARY),
        'source_manifest_sha256': sha(ev.MANIFEST), 'sources': len(records),
        'test_seed': SEED, 'attacks': ATTACKS,
        'detection_threshold_from_validation': threshold,
        'localization_threshold': float(PROTOCOL.localization_threshold),
        'splicing': 'Next sorted Test authentic source used as raw donor; same-coordinate 15-30% rectangle replaces protected host pixels.',
        'telea_inpainting': 'OpenCV Telea radius 3 on 15-30% rectangle of protected image.',
        'copy_move_jpeg85': 'Frozen copy-move geometry followed by JPEG quality 85, 4:4:4; ground truth is destination rectangle, while compression affects the whole image.',
        'selection': 'No training or threshold change; all Test outcomes exploratory because prior Test results have been seen.'})
    try:
        write_json(OUT / 'status.json', {'state': 'RUNNING', 'stage': 'model_load'})
        torch.set_num_threads(4)
        model = SSPGJointModelV2(stego_checkpoint=str(DECODER),
            adapter_bottleneck=PROTOCOL.adapter_bottleneck,
            fusion_hidden_dim=PROTOCOL.fusion_hidden_dim, knn_k=PROTOCOL.knn_k,
            gnn_hidden_dim=PROTOCOL.gnn_hidden_dim, gnn_depth=PROTOCOL.gnn_depth).to(DEVICE)
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        if ckpt.get('selection_metric') != 'validation_joint_loss' or ckpt.get('test_accessed') is not False:
            raise RuntimeError('FULL checkpoint lacks Validation-only selection certification')
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        for attack in ATTACKS:
            write_json(OUT / 'status.json', {'state': 'RUNNING', 'stage': attack})
            d = OUT / attack
            d.mkdir()
            ev.OUT = d
            dataset = UnknownAttackDataset(records, attack)
            loader = build_loader(dataset, shuffle=False, seed=SEED)
            detection, localization, rows = ev.evaluate(model, loader, threshold, float(PROTOCOL.localization_threshold))
            write_json(d / 'metrics.json', {'detection': detection, 'localization': localization,
                                          'sources': len(records), 'active_samples': len(dataset),
                                          'experiment_id': 'CLEAN-BUDGET142-STAGE-C-EXPLORATORY-V2'})
            pd.DataFrame(rows).to_csv(d / 'predictions.csv', index=False)
            print(f'{attack}: AUC={detection["roc_auc"]:.6f} F1={detection["f1"]:.6f} PixelF1={localization["pixel_f1"]:.6f}', flush=True)
        write_json(OUT / 'status.json', {'state': 'COMPLETE', 'attacks': ATTACKS})
    except Exception:
        write_json(OUT / 'status.json', {'state': 'FAILED', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
