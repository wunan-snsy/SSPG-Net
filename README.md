# SSPG-Net

Reference implementation and experiment artifacts for **Self-Steganographic Provenance Graphs for Image Forgery Detection and Localization**.

SSPG-Net is an active image-forensics framework. Before distribution, it embeds a structured 128-bit fragile provenance payload into an image. At verification time, a frozen residual-aware decoder turns payload inconsistency into a spatial evidence map. The model combines this evidence with visual patch representations and graph reasoning to perform joint image-level forgery detection and patch-level localization.

> **Evaluation scope.** The headline results in this repository were obtained on a predefined, leakage-clean validation split derived from IMD2020. The final test partition was not accessed in the formal joint, causal, LOAO, or ablation experiments. These numbers should not be described as independent test-set or cross-dataset performance.

## Results included in v1.0.0

| Experiment | Detection ROC-AUC | Balanced accuracy | Detection F1 | Localization F1 | Localization IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full SSPG-Net | 0.996358 | 0.983871 | 0.983607 | 0.775291 | 0.633042 |
| No provenance | 0.990895 | 0.975806 | 0.975207 | 0.611990 | 0.440911 |
| No graph | 0.983611 | 0.983871 | 0.983871 | 0.737585 | 0.584265 |
| Visual only | 0.964620 | 0.943548 | 0.941176 | 0.508963 | 0.341348 |

The frozen V3.3 provenance channel reports 94.3902% payload bit accuracy, 5.6098% bit error rate, 1.0033% exact 128-bit recovery, and 40.0850 dB protected-image PSNR. Machine-readable source artifacts are under [`results/release_v1.0.0`](results/release_v1.0.0). The additional E1--E7 analyses are under [`results/extension_e1_e7_v1.0.0`](results/extension_e1_e7_v1.0.0).

The E8 same-protocol external baseline artifacts are under [`results/external_baselines/e8_release`](results/external_baselines/e8_release). On the 63-source final test split, SSPG-Net and TruFor were evaluated on identical SSPG-protected inputs; EditGuard was evaluated on the same sources and attack geometry using its own protection. The E8 release includes aggregate tables, per-sample scores and pixel counts, bootstrap intervals, protocol metadata, and adapter audits. It excludes source images and third-party model weights. See the E8 README for scope and limitations.

## Repository layout

```text
configs/                   smoke-test and experiment configuration
datasets/                  IMD2020 dataset loader
manifests/                 versioned relative-path split manifests
models/                    provenance, visual, graph, and joint models
preprocessing/             embedding, manipulation, and evidence construction
training/                  frozen joint, LOAO, ablation, and channel protocols
tests/                     smoke, protocol, model, and audit checks
results/release_v1.0.0/    paper-aligned CSV/JSON artifacts
results/extension_e1_e7_v1.0.0/  additional experiment artifacts
results/external_baselines/e8_release/  compact E8 baseline artifacts
docs/                      data and reproducibility documentation
```

## Installation

Python 3.12 was used for the local release verification. Create an isolated environment, install the PyTorch build appropriate for your platform, then install the remaining dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The exact locally tested package versions are recorded in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). CUDA is required by the frozen formal training protocol; the compact smoke test can be adapted to CPU by changing the configuration.

## Quick verification

```bash
python main.py --config configs/rtx3060_6gb.yaml --mode smoke_test
python scripts/verify_release.py
```

The first command checks forward and backward passes through the reference pipeline. The second checks the release artifact inventory, headline metrics, protocol status, and SHA-256 manifest.

## Reproducing the formal experiments

1. Obtain IMD2020 from its original provider and place its contents under `data/imd2020`.
2. Confirm that paths in `manifests/imd2020_manifest_e5_clean.csv` resolve relative to that directory.
3. Place `stego_structured_v33_128bit_best.pth` under `checkpoints/`, or train the V3.3 channel first.
4. Run the frozen joint protocol:

```bash
python -m training.train_e53r_joint_v1
```

Formal LOAO and ablation entry points are:

```bash
python -m training.train_e53r_loao_v1 --held-out erase --preflight
python -m training.train_e53r_loao_v1 --held-out erase
python -m training.train_e53r_ablation_v1 --variant no_provenance
python -m training.train_e53r_ablation_v1 --variant no_graph
python -m training.train_e53r_ablation_v1 --variant visual_only --aggregate
```

Use `--help` on an entry point before launching a run. Formal scripts refuse unsupported variants and preserve the validation-only protocol. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for expected inputs, outputs, and limitations.

## Data, weights, and availability

The IMD2020 images are not redistributed. Large trained checkpoints are excluded from Git and are intended for a Zenodo checkpoint record. The clean split manifest, per-run protocols, histories, predictions, and summary tables are included. See [`docs/DATA.md`](docs/DATA.md) and [`checkpoints/README.md`](checkpoints/README.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Add the article DOI and archival software DOI only after they exist; this repository does not invent persistent identifiers.

## License

The code and repository-authored materials are released under the [MIT License](LICENSE). Third-party datasets, pretrained backbones, and external dependencies remain subject to their own terms.
