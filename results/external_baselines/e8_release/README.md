# E8 external same-protocol baseline release

This compact release contains frozen protocol identifiers, per-sample scores and
pixel confusion counts, aggregate metrics, source-cluster bootstrap intervals,
adapter audit, and verification reports for SSPG-Net, TruFor, and EditGuard.
Absolute local path prefixes were removed; numerical results were not edited.
`localization_map` values refer to dense local arrays that are not distributed.
The original IMD2020 images, third-party weights, and manuscript are excluded.
The source code for reproduction is in `scripts/e8_*.py` in this repository.

The `e8_verification.json` report records verification of the original local
artifacts. `archive_manifest.json` records SHA-256 hashes of this portable copy.
