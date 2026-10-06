"""Inventory compact E8 artifacts while leaving weights and dense maps local."""
import hashlib
import json

from eval_test_final import ROOT

OUT = ROOT / "results/external_baselines/e8"
FORMATS = {".csv", ".json", ".md", ".tex"}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if json.loads((OUT / "e8_verification.json").read_text())["status"] != "PASS":
        raise RuntimeError("Cannot archive unverified E8 results")
    readme = """# Compact E8 archive

These files contain the frozen protocol, official code/checkpoint identities,
per-sample scores and pixel counts, metrics, bootstrap intervals, tables,
and verification report. Dense per-sample score maps and model checkpoints
stay local because of size. Regenerate them with the commands in
e8_summary.md, then run scripts/e8_verify.py. The archive does not contain
or redistribute IMD2020 source images or external model weights.

Table B compares methods on matched source images and attack geometry,
using each method's own protected image. EditGuard's 256x256 input adaptation
and clean-checkpoint compatibility correction are documented in
compatibility_notes.md. The manuscript E8 revision is delivered separately.
"""
    (OUT / "ARCHIVE_README.md").write_text(readme, encoding="utf-8")
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "archive_manifest.json" or path.suffix.lower() not in FORMATS:
            continue
        files.append({"path": path.relative_to(OUT).as_posix(),
                      "bytes": path.stat().st_size, "sha256": sha(path)})
    record = {"format": "SSPG-Net E8 compact archive v1",
              "dense_maps_archived": False,
              "checkpoints_archived": False,
              "files": files}
    (OUT / "archive_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Archived {len(files)} compact E8 artifacts")


if __name__ == "__main__":
    main()
