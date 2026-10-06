"""Official EditGuard clean weights with E1 source and attack geometry."""
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_functional
import yaml

from e8_metrics import localization_metrics, pixel_counts
from e8_freeze_protocol import records_for
from eval_test_final import ROOT, TEST_SEED
from preprocessing.active_sspg_dataset import (
    ATTACK_GEOMETRY_SEED_OFFSET, ActiveSSPGDataset, apply_controlled_attack,
    load_rgb_256, stable_seed,
)

EDITGUARD = ROOT / "third_party/EditGuard"
OUT = ROOT / "results/external_baselines/e8"


def load_model():
    # BasicSR 1.4.2 imports the removed torchvision.functional_tensor module.
    sys.modules["torchvision.transforms.functional_tensor"] = tv_functional
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    sys.path.insert(0, str(EDITGUARD / "code"))
    from models.networks import define_G_v2
    from models.modules.common import DWT, IWT
    opt = yaml.safe_load((EDITGUARD / "code/options/test_editguard.yml").read_text())
    opt["network_G"]["scale"] = 4  # clean.pth has 12 IRN blocks; strict load verifies.
    model = define_G_v2(opt)
    state = torch.load(EDITGUARD / "checkpoints/clean.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval(), DWT(), IWT()


@torch.inference_mode()
def protect(model, dwt, original, source_seed):
    host = original.unsqueeze(0).cuda()
    secret = torch.zeros_like(host)
    secret[:, 2] = 1.0  # official blue auxiliary image
    rng = np.random.default_rng(source_seed)
    message = torch.from_numpy(rng.choice([-0.5, 0.5], size=(1, 64)).astype(np.float32)).cuda()
    _, container = model(x=dwt(host), x_h=[dwt(secret)], message=message)
    if tuple(container.shape[-2:]) != (256, 256):
        raise RuntimeError(f"Unexpected EditGuard protected size: {container.shape}")
    return container[0].clamp(0, 1), secret


@torch.inference_mode()
def localize(model, iwt, image, secret):
    # Official test() quantizes the protected/tampered image before reverse pass.
    quantized = (image.unsqueeze(0).cuda().clamp(0, 1) * 255).round() / 255
    _, recovered_secrets, _, _ = model(x=quantized, rev=True)
    # Match official image_recovery: threshold the raw recovered auxiliary image.
    recovered = iwt(recovered_secrets[0])
    residual = torch.abs(secret - recovered).amax(dim=1, keepdim=True)
    residual = F.interpolate(residual, size=(256, 256), mode="bilinear", align_corners=False)
    return residual[0, 0].cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-sources", type=int)
    args = parser.parse_args()
    model, dwt, iwt = load_model()
    results = {}
    for split, seed in (("val", 20260923), ("test", TEST_SEED)):
        records = records_for(split)
        dataset = ActiveSSPGDataset(records, seed)
        limit = len(records) if args.limit_sources is None else min(args.limit_sources, len(records))
        map_dir = OUT / ("validation" if split == "val" else "test") / "editguard_maps"
        map_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        pooled_eg_sse = pooled_sspg_sse = 0.0
        pooled_count = 0
        for i in range(limit):
            authentic = dataset[2*i]
            attacked = dataset[2*i+1]
            original = load_rgb_256(records[i]["image_path"])
            source_seed = int(authentic["source_seed"])
            eg, secret = protect(model, dwt, original, source_seed)
            sspg = authentic["received"]
            pooled_eg_sse += float(((eg.cpu() - original)**2).sum().item())
            pooled_sspg_sse += float(((sspg - original)**2).sum().item())
            pooled_count += original.numel()
            attack = attacked["attack"]
            rng = np.random.default_rng(source_seed + ATTACK_GEOMETRY_SEED_OFFSET)
            manipulated, mask = apply_controlled_attack(eg.cpu(), attack, rng)
            if not torch.equal(mask, attacked["mask"]):
                raise RuntimeError(f"Attack mask mismatch at source {i}")
            for flag, image, truth, name in ((0, eg.cpu(), authentic["mask"], "authentic"),
                                             (1, manipulated, mask, attack)):
                fmap = localize(model, iwt, image, secret)
                truth_array = truth.numpy()[0] >= 0.5
                counts = pixel_counts(fmap, truth_array, 0.2, strict=True)
                map_name = f"{i:04d}_{flag}.npy"
                np.save(map_dir / map_name, fmap)
                rows.append({"sample_index": 2*i+flag, "sample_key": authentic["sample_key"],
                             "source_path": records[i]["image_path"], "attack": name,
                             "detection_label": flag, "localization_map": map_name,
                             "mask_sha256": hashlib.sha256(truth.numpy().astype(np.uint8).tobytes()).hexdigest(),
                             "pixel_tp": counts["tp"],
                             "pixel_fp": counts["fp"],
                             "pixel_fn": counts["fn"]})
            if (i+1) % 10 == 0:
                print(f"EditGuard {split}: {i+1}/{limit} sources", flush=True)
        frame = pd.DataFrame(rows)
        dest = OUT / ("validation" if split == "val" else "test")
        frame.to_csv(dest / "editguard_predictions.csv", index=False)
        total = {"tp": int(frame.pixel_tp.sum()), "fp": int(frame.pixel_fp.sum()),
                 "fn": int(frame.pixel_fn.sum()),
                 "tn": len(frame) * 256 * 256 - int(frame[["pixel_tp", "pixel_fp", "pixel_fn"]].sum().sum())}
        loc = localization_metrics(total, 0.2)
        eg_mse = pooled_eg_sse / pooled_count
        sspg_mse = pooled_sspg_sse / pooled_count
        metrics = {"sources": limit, "samples": len(frame), "pixel_f1": loc["pixel_f1"],
                   "pixel_iou": loc["pixel_iou"], "pixel_tp": loc["tp"], "pixel_fp": loc["fp"],
                   "pixel_fn": loc["fn"], "localization_threshold": 0.2,
                   "detection_auc": None, "detection_f1": None,
                   "editguard_protected_psnr": -10*np.log10(eg_mse),
                   "sspg_protected_psnr": -10*np.log10(sspg_mse),
                   "editguard_pooled_mse": eg_mse, "sspg_pooled_mse": sspg_mse}
        (dest / "editguard_metrics.json").write_text(json.dumps(metrics, indent=2))
        results[split] = metrics
    if args.limit_sources is None:
        (OUT / "test/editguard_fidelity.json").write_text(json.dumps({
            "definition": "-10*log10(pooled squared error / pooled RGB element count)",
            "sources": 63, "editguard_psnr": results["test"]["editguard_protected_psnr"],
            "sspg_psnr": results["test"]["sspg_protected_psnr"]}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
