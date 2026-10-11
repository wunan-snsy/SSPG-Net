# Checkpoints

Trained checkpoints are intentionally excluded from Git because the formal joint and ablation files are approximately 334--344 MB each. Place checkpoints in this directory using the filenames expected by the training and audit scripts.

The frozen provenance decoder required by the formal protocol is:

`stego_structured_v33_128bit_best.pth`

Local v1.0.0 SHA-256: `47e8d0131a15f6d2735c8751a80ee40c7883074c6e21c068bc7436a36e6a049e`

The earlier v1.0.0 checkpoint named above is recorded by hash but is not
included in the v1.1.0 archive. No download URL or DOI is asserted for it.

The v1.1.0 supplement lists ten Clean-142 decoder, E9, and paired E2
checkpoints in `results/release_v1.1.0/WEIGHT_INDEX.json`. The three
separate ZIP bundles and their archive checksums are listed in
`results/release_v1.1.0/weights_ARCHIVE_INDEX.json`. Extract them at
repository root so that each `.pth` lands at its recorded relative path.
The three ZIP bundles and `SHA256SUMS` are available as separate assets of
the [v1.1.0 GitHub release](https://github.com/wunan-snsy/SSPG-Net/releases/tag/v1.1.0).
GitHub reports SHA-256 digests matching the published archive checksums.
