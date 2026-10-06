"""Audit the E8 EditGuard checkpoint adapter on an official example image."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e8_editguard import EDITGUARD, OUT, load_model, localize, protect  # noqa: E402


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    config = yaml.safe_load((EDITGUARD / "code/options/test_editguard.yml").read_text())
    ckpt = EDITGUARD / "checkpoints/clean.pth"
    commit = subprocess.check_output(
        ["git", "-C", str(EDITGUARD), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_changes = subprocess.check_output(
        ["git", "-C", str(EDITGUARD), "diff", "--name-only"], text=True
    ).strip()
    model, dwt, iwt = load_model()  # strict=True checkpoint load
    from models.networks import define_G_v2
    as_checked_in = define_G_v2(config)
    weights = torch.load(ckpt, map_location="cpu", weights_only=False)
    mismatch = as_checked_in.load_state_dict(weights, strict=False)
    checked_in_strict_load_passes = not mismatch.missing_keys and not mismatch.unexpected_keys
    example = EDITGUARD / "dataset/examples/0002.png"
    rgb = np.asarray(Image.open(example).convert("RGB").resize((256, 256)))
    original = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    protected, secret = protect(model, dwt, original, 20261006)
    clean_map = localize(model, iwt, protected, secret)
    manipulated = protected.clone()
    manipulated[:, 80:128, 80:128] = 0.0
    edit_map = localize(model, iwt, manipulated, secret)
    def official_raw_map(image):
        quantized = (image.unsqueeze(0).cuda().clamp(0, 1) * 255).round() / 255
        with torch.inference_mode():
            _, recovered_secrets, _, _ = model(x=quantized, rev=True)
            recovered = iwt(recovered_secrets[0])
            residual = torch.abs(secret - recovered).amax(dim=1, keepdim=True)
            residual = F.interpolate(residual, size=(256, 256), mode="bilinear", align_corners=False)
        return residual[0, 0].cpu().numpy()
    clean_raw = official_raw_map(protected)
    edit_raw = official_raw_map(manipulated)
    region = np.zeros((256, 256), dtype=bool)
    region[80:128, 80:128] = True
    clean = protected.cpu().numpy()
    mse = float(np.mean((clean - original.numpy()) ** 2))
    result = {
        "official_repo_commit": commit,
        "checkpoint_sha256": sha256(ckpt),
        "tracked_source_changes": tracked_changes.splitlines() if tracked_changes else [],
        "yaml_top_level_scale": config["scale"],
        "yaml_network_scale": config["network_G"]["scale"],
        "adapter_network_scale": 4,
        "block_num": config["network_G"]["block_num"],
        "loaded_irn_operations": len(model.irn.operations),
        "checked_in_irn_operations": len(as_checked_in.irn.operations),
        "checked_in_strict_load_passes": checked_in_strict_load_passes,
        "checked_in_missing_keys": len(mismatch.missing_keys),
        "checked_in_unexpected_keys": len(mismatch.unexpected_keys),
        "loaded_state_keys": len(model.state_dict()),
        "example": str(example.relative_to(EDITGUARD)),
        "input_size": list(original.shape),
        "protected_size": list(protected.shape),
        "protected_psnr_db": float(-10 * np.log10(mse)),
        "clean_map_finite": bool(np.isfinite(clean_map).all()),
        "edited_map_finite": bool(np.isfinite(edit_map).all()),
        "clean_map_positive_fraction": float(np.mean(clean_map > 0.2)),
        "edited_region_positive_fraction": float(np.mean(edit_map[region] > 0.2)),
        "edited_outside_positive_fraction": float(np.mean(edit_map[~region] > 0.2)),
        "clean_raw_vs_adapter_threshold_flip_fraction": float(np.mean((clean_raw > 0.2) != (clean_map > 0.2))),
        "edit_raw_vs_adapter_threshold_flip_fraction": float(np.mean((edit_raw > 0.2) != (edit_map > 0.2))),
    }
    if result["loaded_irn_operations"] != 12 or result["loaded_state_keys"] != 1281:
        raise RuntimeError("Checkpoint structure differs from the recorded E8 adapter")
    if result["checked_in_irn_operations"] != 6 or checked_in_strict_load_passes:
        raise RuntimeError("Checked-in YAML/checkpoint mismatch was not reproduced")
    if result["tracked_source_changes"]:
        raise RuntimeError("Official tracked source files have been modified")
    if not result["clean_map_finite"] or not result["edited_map_finite"]:
        raise RuntimeError("Non-finite demo output")
    dest = OUT / "editguard_adapter_audit.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
