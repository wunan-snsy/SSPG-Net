"""Verify the public v1.1.0 E9, matched-control, and Figure 8 records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "results/release_v1.1.0"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, found {actual}")


def main() -> None:
    assert RELEASE.is_dir()
    lines = (RELEASE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 70
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = RELEASE / relative
        assert path.is_file(), relative
        assert sha(path) == expected, relative

    protocol = json.loads((RELEASE / "e9/protocol.json").read_text(encoding="utf-8"))
    assert sha(RELEASE / "e9/source_manifest_frozen.csv") == protocol["source_manifest_sha256"]
    assert sha(RELEASE / "e9/attack_plan.csv") == protocol["attack_plan_sha256"]
    sources = csv_rows(RELEASE / "e9/source_manifest_relative.csv")
    attacks = csv_rows(RELEASE / "e9/attack_plan.csv")
    assert len(sources) == 200 and len({r["source_id"] for r in sources}) == 200
    assert len(attacks) == 2400
    assert all(r["path"].startswith("data/coco2017/val2017/") for r in sources)
    assert {r["band"] for r in attacks} == {"01_05", "05_10", "10_15", "15_30"}
    for variant in ("full", "no_provenance"):
        rows = csv_rows(RELEASE / f"e9/{variant}/predictions.csv")
        assert len(rows) == 2600
        run = json.loads((RELEASE / f"e9/{variant}/run.json").read_text(encoding="utf-8"))
        assert run["predictions_sha256"] == sha(RELEASE / f"e9/{variant}/predictions.csv")
        assert run["status"] == "COMPLETE"

    metrics = csv_rows(RELEASE / "e9/report/e9_metrics_with_95ci.csv")
    lookup = {(r["variant"], r["band"], r["metric"]): float(r["value"]) for r in metrics}
    close(lookup["full", "01_05", "pixel_f1"], 0.3632940038500308)
    close(lookup["full", "01_05", "detection_recall"], 0.4583333333333333)
    close(lookup["full", "15_30", "pixel_f1"], 0.7369, 0.0001)

    matched = RELEASE / "matched_no_graph"
    status = json.loads((matched / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "COMPLETE"
    assert status["completed"] == [20261001, 20261002, 20261003]
    paired = csv_rows(matched / "paired_seed_results.csv")
    assert len(paired) == 3
    for row in paired:
        seed = row["seed"]
        assert int(row["full_parameters"]) == 1_416_490
        assert int(row["matched_parameters"]) == 1_416_110
        lineage = json.loads((matched / seed / "lineage.json").read_text(encoding="utf-8"))
        assert lineage["test_accessed"] is False
        assert lineage["summary_sha256"] == sha(matched / seed / "metrics.csv")

    provenance = json.loads((RELEASE / "figure8/provenance_frozen.json").read_text(encoding="utf-8"))
    assert len(provenance["rows"]) == 5
    for row in provenance["rows"]:
        name = f"row{row['row']}_{Path(row['probability_path']).name}"
        assert sha(RELEASE / "figure8/prediction_masks" / name) == row["probability_sha256"]
    assert provenance["full_checkpoint_sha256"] == protocol["checkpoint_sha256"]["full"]

    weights = json.loads((RELEASE / "WEIGHT_INDEX.json").read_text(encoding="utf-8"))
    assert len(weights) == 10
    assert sum(r["bytes"] for r in weights) > 3_100_000_000
    print(f"SSPG-Net v1.1.0 release verification: PASS ({len(lines)} files, 10 weight hashes)")


if __name__ == "__main__":
    main()
