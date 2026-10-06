"""Record immutable external code, checkpoint, and runtime identities."""
import hashlib
import json
import platform
import subprocess

import torch

from eval_test_final import ROOT

OUT = ROOT / "results/external_baselines/e8"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository(name):
    path = ROOT / "third_party" / name
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     text=True).strip()
    tracked_diff = subprocess.run(["git", "-C", str(path), "diff", "--quiet"],
                                  check=False).returncode
    if tracked_diff != 0:
        raise RuntimeError(f"{name}: tracked third-party source changed")
    return commit


def main():
    tru = ROOT / "third_party/TruFor/TruFor_train_test/pretrained_models/weights/trufor.pth.tar"
    eg = ROOT / "third_party/EditGuard/checkpoints/clean.pth"
    calibration = json.loads((OUT / "validation/trufor_calibration.json").read_text())
    record = {
        "trufor": {
            "repository_url": "https://github.com/grip-unina/TruFor",
            "commit": repository("TruFor"),
            "checkpoint_filename": tru.name,
            "checkpoint_sha256": sha(tru),
            "score_orientation": "higher=manipulation",
            "preprocessing": "Exact float E1 received tensor scaled by 255/256",
            "detection_threshold_source": "Validation Youden J",
            "detection_threshold": calibration["threshold"],
            "localization_threshold": 0.5,
            "inference_command": "python scripts/e8_trufor.py",
        },
        "editguard": {
            "repository_url": "https://github.com/xuanyuzhang21/EditGuard",
            "commit": repository("EditGuard"),
            "checkpoint_filename": eg.name,
            "checkpoint_sha256": sha(eg),
            "score_orientation": "higher=max RGB residual means manipulation",
            "preprocessing": "256x256 E1 original, official protection, E1 attack geometry",
            "localization_threshold": 0.2,
            "inference_command": "python scripts/e8_editguard.py",
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_size": 1,
            "precision": "float32",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "repo_versions.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Recorded official repository and checkpoint identities")


if __name__ == "__main__":
    main()
