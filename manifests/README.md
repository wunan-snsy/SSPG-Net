# Data manifests

`imd2020_manifest_e5_clean.csv` contains the leakage-audited split used by the formal SSPG-Net experiments. Paths are relative to `data/imd2020`; the underlying images are not redistributed.

`imd2020_manifest.csv` is the base manifest used to train the V3.3 provenance channel. Regenerate it with `python scripts/build_imd2020_manifest.py` after placing IMD2020 under `data/imd2020`.
