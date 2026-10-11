# SSPG-Net v1.1.0

This experimental supplement adds the independent COCO-source controlled
E9 evaluation, the three-seed parameter-matched NO_GRAPH Validation
control, and the Clean-142 FULL Figure 8 provenance record. The model
method and earlier frozen results are unchanged. See
`results/release_v1.1.0/README.md` and `results/release_v1.1.0/SHA256SUMS`.
Large checkpoint bundles are stored outside Git and identified by the
published `WEIGHT_INDEX.json` and `weights_SHA256SUMS` files. The archive
location must be filled in only after a public deposit is verified.

## Earlier v1.0.0 release

This is the first public reference release accompanying the manuscript *Self-Steganographic Provenance Graphs for Image Forgery Detection and Localization*.

The repository contains the model and training code, frozen protocol definitions, the leakage-clean split manifest, result tables and prediction artifacts, and verification utilities. Unpublished manuscript sources are not distributed. The reported joint and ablation results use the predefined validation split; the final test split was not accessed by these experiments.

The IMD2020 images and large trained checkpoints are not bundled. Dataset access remains subject to the original provider's terms. See `docs/DATA.md`, `docs/REPRODUCIBILITY.md`, and `checkpoints/README.md`.
