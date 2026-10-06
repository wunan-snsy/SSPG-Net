# E8 audit report

- **PASS** E1 four published metrics reproduced locally
- **PASS** Validation 62 sources/124 active samples and Test 63/126
- **PASS** Frozen source manifest, masks, attack metadata, checkpoint and thresholds unchanged
- **PASS** Official TruFor and EditGuard repositories/checkpoints only
- **PASS** TruFor identical E1 in-memory active images and exact masks
- **PASS** TruFor score orientation from official detection target and map class
- **PASS** TruFor detection threshold selected on Validation only
- **PASS** TruFor no inverse-map maximization or boundary removal
- **PASS** EditGuard same source IDs, attack families, exact geometry and masks
- **PASS** EditGuard protected original directly, no double protection
- **PASS** EditGuard official clean checkpoint and fixed 0.2 localization threshold
- **PASS** EditGuard detection AUC/F1 not fabricated
- **PASS** Localization includes authentic all-zero masks in micro counts
- **PASS** Source-cluster bootstrap: 63 paired clusters, 2000 draws, seed 20261006
- **PASS** Shared evaluator rechecked saved E1, TruFor, and EditGuard score maps
- **PASS** No Test-set tuning
- **PASS** Third-party architecture/source files unedited; adapter options documented

Scope note: EditGuard uses method-specific protected pixels at the same 256x256 source resolution; TruFor receives identical SSPG E1 active pixels. The EditGuard checkpoint/config scale discrepancy and runtime alias are disclosed in `compatibility_notes.md`.
