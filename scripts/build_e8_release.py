"""Build the portable, paper-free compact E8 public artifact set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/external_baselines/e8"
TARGET = ROOT / "results/external_baselines/e8_release"
EXTENSIONS = {".csv", ".json", ".md"}
OMIT = {"archive_manifest.json", "ARCHIVE_README.md"}


def portable(text: str) -> str:
    # JSON encodes Windows separators twice; CSV and Markdown encode them once.
    text = text.replace("D:\\\\SSPG-Net\\\\", "")
    text = text.replace("D:\\SSPG-Net\\", "")
    text = text.replace("D:/SSPG-Net/", "")
    return text


def main() -> None:
    verification = json.loads((SOURCE / "e8_verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("E8 verification must pass before packaging")
    TARGET.mkdir(parents=True, exist_ok=True)
    records = []
    for src in sorted(SOURCE.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in EXTENSIONS or src.name in OMIT:
            continue
        rel = src.relative_to(SOURCE)
        dest = TARGET / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = portable(src.read_text(encoding="utf-8"))
        if "D:\\SSPG-Net" in content or "D:\\\\SSPG-Net" in content:
            raise RuntimeError(f"Absolute workspace path remains in {rel}")
        dest.write_text(content, encoding="utf-8", newline="\n")
        records.append({"path": rel.as_posix(), "bytes": dest.stat().st_size,
                        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    readme = """# E8 external same-protocol baseline release

This compact release contains frozen protocol identifiers, per-sample scores and
pixel confusion counts, aggregate metrics, source-cluster bootstrap intervals,
adapter audit, and verification reports for SSPG-Net, TruFor, and EditGuard.
Absolute local path prefixes were removed; numerical results were not edited.
`localization_map` values refer to dense local arrays that are not distributed.
The original IMD2020 images, third-party weights, and manuscript are excluded.
The source code for reproduction is in `scripts/e8_*.py` in this repository.

The `e8_verification.json` report records verification of the original local
artifacts. `archive_manifest.json` records SHA-256 hashes of this portable copy.
"""
    (TARGET / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    manifest = {"format": "SSPG-Net E8 portable compact artifacts v1",
                "source_verification": verification.get("status"),
                "files": records}
    (TARGET / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Packaged {len(records)} E8 artifacts in {TARGET}")


if __name__ == "__main__":
    main()
