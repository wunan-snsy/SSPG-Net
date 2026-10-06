# E8 layout, EditGuard adapter, and bootstrap audit

## Manuscript layout

The revised TeX already loads `placeins` with the `section` option. An explicit
`\FloatBarrier` was added immediately before Section 5.8. The compiled PDF now
places Tables 15–17 and Figure 8 before the Section 5.8 heading; Tables 18 and
19 follow the E8 text. The two new table numbers remain 18 and 19.

## EditGuard checkpoint and adapter

- Official repository commit: `174778fb371640bb31b32fcb9f7bfd440c88766b`.
- `clean.pth` SHA-256: `b967b91b94a464ab73f82db63a57ee818e6fc5d99d7ae6aa094a1777cac977d5`.
- No tracked third-party source files differ from that commit.
- Checked-in `options/test_editguard.yml`: top-level `scale: 4`, but
  `network_G.scale: 2` and `block_num: [6, 6]`. The network constructor uses
  `network_G.scale` to compute the number of invertible stages. At 2 it builds
  six blocks; strict checkpoint loading fails with 504 unexpected keys.
- The E8 adapter sets `network_G.scale: 4` in memory, building twelve blocks.
  All 1,281 state-dict keys load strictly. This **changes the instantiated
  architecture relative to the checked-in YAML**, while matching the official
  checkpoint. It does not change checkpoint values or the official
  forward/reverse implementation. The scale option is separate from the input
  image resolution.
- The adapter now computes localization residuals from the **unclamped**
  recovered auxiliary image, matching the official `image_recovery` path.
  Re-evaluation changed a few raw pixel counts, but Table 19 and its reported
  paired intervals are unchanged at four decimal places.
- Official repository example `dataset/examples/0002.png` was run through the
  E8 256×256 protection and recovery path. Strict loading succeeded; the
  protected image had PSNR 37.22 dB. A simple 48×48 black edit produced
  99.96% positives inside the edit and 0.67% outside, at threshold 0.2.
  This is a sanity check, not a validation of performance across all edits.
- **Remaining boundary:** the official demo resizes inputs to 512×512. E8 uses
  256×256 to match SSPG's Test protocol. Consequently, strict checkpoint
  matching and the example check do not establish equivalence to the native
  512×512 pipeline. The manuscript discloses this.

The machine-readable example and architecture check is in
`editguard_adapter_audit.json`.

## Paired source-cluster bootstrap

`scripts/e8_bootstrap.py` groups the 126 Test rows into 63 source IDs, checks
that every ID has exactly two rows, and samples 63 IDs with replacement for
each of 2,000 replicates (seed 20261006). It concatenates both rows for each
sampled source. SSPG, TruFor, and EditGuard predictions and pixel counts thus
remain paired within every replicate. Detection AUC and F1 are recomputed from
the sampled scores and labels. Pixel TP/FP/FN are summed before F1 and IoU are
recomputed. No pixel-level resampling or mean of per-image F1 is used.

The bootstrap was rerun after the EditGuard residual correction. Its six
differences and intervals agree with the manuscript at four decimal places;
the full-precision values are in `bootstrap/bootstrap_results.csv`. The shared
map/mask evaluator was rerun and returned `PASS` in `e8_verification.json`.
