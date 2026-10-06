"""Build an isolated, manuscript-free GitHub publication snapshot.

The new Git repository deliberately has no local development history: old
commits may have contained manuscript drafts even though the current tree does
not. Review the generated inventory before pushing its main branch.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".public-release-staging"
EXTRA = [
    "scripts/build_e8_release.py",
    "scripts/build_public_snapshot.py",
    "scripts/e8_archive.py",
    "scripts/e8_bootstrap.py",
    "scripts/e8_editguard.py",
    "scripts/e8_editguard_audit.py",
    "scripts/e8_finalize.py",
    "scripts/e8_freeze_protocol.py",
    "scripts/e8_metrics.py",
    "scripts/e8_preflight.py",
    "scripts/e8_trufor.py",
    "scripts/e8_verify.py",
    "scripts/e8_versions.py",
]
FORBIDDEN_SUFFIXES = {".pdf", ".tex", ".bib", ".pth", ".pt", ".ckpt", ".npy"}
FORBIDDEN_PARTS = {"paper", "deliverables", "manuscript_revisions", "submission", "third_party", "data"}


def main() -> None:
    if DEST.exists():
        raise RuntimeError(f"Staging exists; inspect or remove it first: {DEST}")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    e8 = [p.relative_to(ROOT).as_posix() for p in
          (ROOT / "results/external_baselines/e8_release").rglob("*") if p.is_file()]
    files = sorted(set(filter(None, tracked)) | set(EXTRA) | set(e8))
    inventory = []
    for rel in files:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        path = Path(rel)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(
            part.lower() in FORBIDDEN_PARTS for part in path.parts
        ):
            raise RuntimeError(f"Forbidden publication path: {rel}")
        if src.stat().st_size > 2_000_000:
            raise RuntimeError(f"Large file requires manual review: {rel}")
        dest = DEST / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if rel.startswith("results/extension_e1_e7_v1.0.0/e4/") and path.suffix == ".csv":
            text = dest.read_text(encoding="utf-8")
            text = text.replace("D:\\SSPG-Net\\", "")
            dest.write_text(text, encoding="utf-8", newline="\n")
        inventory.append({"path": rel, "bytes": dest.stat().st_size})
    for item in inventory:
        path = DEST / item["path"]
        if path.suffix.lower() in {".csv", ".json", ".md"}:
            text = path.read_text(encoding="utf-8")
            if "D:\\SSPG-Net" in text or "C:\\Users\\wunan" in text:
                raise RuntimeError(f"Private path remains: {item['path']}")
    (DEST / "PUBLIC_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Prepared {len(files)} files ({sum(i['bytes'] for i in inventory)} bytes) at {DEST}")


if __name__ == "__main__":
    main()
