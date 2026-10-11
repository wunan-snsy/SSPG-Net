# SSPG-Net v1.1.0 experimental supplement

This version adds the frozen E9 COCO-source evaluation, the three-seed
parameter-matched `NO_GRAPH` control, and the Clean-142 FULL Figure 8
provenance record to the earlier v1.0.0/E1--E8 repository. It reports
the retained outcomes; it does not change the SSPG-Net method or retrain
the existing models.

## Contents

- `e9/`: the 200-source COCO val2017 selection, 2,400 controlled attack
  geometries, fixed thresholds, 2,600 prediction rows per evaluated variant,
  source-level bootstrap summaries, paired FULL minus NO_PROVENANCE intervals,
  and a report. The same 200 protected authentic samples are shared by the
  four area-band analyses.
- `matched_no_graph/`: three Validation-only training runs, with per-seed
  protocols, histories, predictions, metric rows, lineage hashes, and the
  paired three-seed summary. The preliminary one-seed run is excluded.
- `e2_comparators/`: the already completed same-seed FULL and original
  NO_GRAPH metric/protocol records used in the paired comparison.
- `figure8/`: the Clean-142 FULL five-row figure, frozen selection and
  provenance, portable path views, and five selected dense prediction maps.
  The examples are reused IMD2020 Test sources and are descriptive.
- `WEIGHT_INDEX.json`: exact checkpoint sizes and SHA-256 values.
- `weights_ARCHIVE_INDEX.json` and `weights_SHA256SUMS`: three separately
  prepared checkpoint archive names, sizes, and SHA-256 values.
- `PROVENANCE_BRIDGE.json`: hashes connecting the frozen local-path records
  to the supplemental portable path copies.

`SHA256SUMS` covers the bytes of each file in this directory, except itself.
Run `python scripts/verify_release_v1_1_0.py` at repository root to verify
the package and the core numeric claims. The verification script does not
require the third-party source images or checkpoints.

## Frozen versus portable source paths

`source_manifest_frozen.csv`, `selection_frozen.csv`, and
`provenance_frozen.json` retain the exact original bytes needed to verify
the recorded experiment hashes. Their Windows paths are historical locators.
The corresponding `*_relative.*` files replace only those locators with
paths relative to the repository. They are convenience views, not new sample
selections. COCO images should be placed at `data/coco2017/val2017/` under
their original file names. IMD2020 and CASIA2 images are not redistributed.

The E9 inference script checks the frozen source and attack manifests, then
resolves COCO files at their recorded path or at the portable repository
location. It rejects any source whose file hash differs from the frozen
manifest. Fresh inference also requires the checkpoint archives extracted
at their recorded repository-relative paths.

## What the evidence supports

E9 uses new COCO source images with controlled Erase, Copy--Move, and Roll
edits. It is not a benchmark of COCO-native forgeries or arbitrary attacks.
The parameter-matched comparison uses three seeds on one Validation split;
it tests a specific tokenwise capacity control and does not isolate graph
topology from all other forms of spatial communication. Figure 8 uses
previously inspected IMD2020 Test sources and is not independent evidence.

## Checkpoint archives

The large `.pth` files are deliberately excluded from Git. Their ten
per-file hashes and the hashes of three ZIP bundles are published in the
indices above. The three ZIP bundles and `SHA256SUMS` are separate assets
of the [v1.1.0 GitHub release](https://github.com/wunan-snsy/SSPG-Net/releases/tag/v1.1.0).
GitHub reports matching asset sizes and SHA-256 digests. This release
does not assign a DOI to the new checkpoint bundles.
