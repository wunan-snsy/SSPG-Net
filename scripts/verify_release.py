"""Verify the SSPG-Net v1.0.0 paper-aligned artifact package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "results" / "release_v1.0.0"


TEXT_SUFFIXES = {
    ".bib", ".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml",
}


def sha256(path: Path) -> str:
    """Hash release artifacts reproducibly across Windows and Linux.

    Git checks text files out with the platform/configured line ending.  The
    release manifest therefore hashes UTF-8 text after normalizing CRLF/CR to
    LF, while binary artifacts continue to be hashed byte-for-byte.
    """
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = path.read_text(encoding="utf-8-sig")
        payload = data.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_hash_manifest(hash_file: Path) -> None:
    lines = []
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, relative = line.split("  ", 1)
        lines.append(f"{sha256(RELEASE / relative)}  {relative}")
    hash_file.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle))


def assert_close(row: dict[str, str], key: str, expected: float) -> None:
    actual = float(row[key])
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-7):
        raise AssertionError(f"{key}: expected {expected}, found {actual}")


def main() -> None:
    required = [
        RELEASE / "joint" / "e53r_joint_v1_summary.csv",
        RELEASE / "ablation" / "e53r_4d_ablation_paper_table.csv",
        RELEASE / "causal" / "e53r_4c2_validation_causal_summary.csv",
        RELEASE / "loao" / "e53r_4c4_loao_erase_summary.csv",
        RELEASE / "provenance" / "stego_structured_v33_reliability_summary.csv",
        ROOT / "manifests" / "imd2020_manifest_e5_clean.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing release files: " + ", ".join(missing))

    joint = read_first_row(required[0])
    assert_close(joint, "detection_auc", 0.9963579604578564)
    assert_close(joint, "detection_balanced_accuracy", 0.9838709677419355)
    assert_close(joint, "detection_f1", 0.9836065573770492)
    assert_close(joint, "localization_pixel_f1", 0.7752914754418514)
    assert_close(joint, "localization_pixel_iou", 0.6330416257383051)
    if joint["test_accessed"].strip().lower() != "false":
        raise AssertionError("The formal joint summary must record test_accessed=False")

    for protocol in RELEASE.rglob("*_protocol.json"):
        payload = json.loads(protocol.read_text(encoding="utf-8"))
        if payload.get("test_accessed") not in (False, None):
            raise AssertionError(f"Unexpected test access in {protocol.relative_to(ROOT)}")

    hash_file = RELEASE / "SHA256SUMS"
    if not hash_file.is_file():
        raise FileNotFoundError("Missing results/release_v1.0.0/SHA256SUMS")
    if "--rewrite-hashes" in sys.argv[1:]:
        rewrite_hash_manifest(hash_file)
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = RELEASE / relative
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch: {relative}")

    print("SSPG-Net v1.0.0 release verification: PASS")
    print(f"Verified {len(required)} required files and the recorded SHA-256 checksums.")


if __name__ == "__main__":
    main()
