# v1.0.0 result artifacts

This directory contains the machine-readable evidence aligned with the manuscript. It deliberately excludes earlier exploratory and test-set experiments.

- `joint/`: frozen full-model history, protocol, validation predictions, and summary.
- `causal/`: payload/evidence interventions and representation audits.
- `loao/`: leave-one-attack-out histories, calibration predictions, held-out predictions, protocols, and summaries.
- `ablation/`: independently trained component ablations and the paper table.
- `provenance/`: V3.3 channel reliability, causal, error-distribution, and multiseed artifacts.
- `SHA256SUMS`: integrity hashes generated from the files above.

All formal joint, causal, LOAO, and ablation claims are scoped to validation-derived experiments. Refer to the per-run protocol JSON files before reusing a value.
