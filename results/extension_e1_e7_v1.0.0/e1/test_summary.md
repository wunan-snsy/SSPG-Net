# E1 Final Test Evaluation

- Frozen checkpoint: `e53r_joint_v1_best.pth`
- Validation detection threshold: 0.960034966469
- Test seed (pre-evaluation): 20260924
- Sources / active samples: 63 / 126
- Device: cuda

## Detection

```json
{
  "roc_auc": 0.9979843789367598,
  "balanced_accuracy": 0.9761904761904762,
  "f1": 0.975609756097561,
  "sensitivity": 0.9523809523809523,
  "specificity": 1.0,
  "threshold": 0.960034966468811,
  "tn": 63,
  "fp": 0,
  "fn": 3,
  "tp": 60
}
```

## Localization

```json
{
  "pixel_f1": 0.7758791307179755,
  "pixel_iou": 0.6338255887860541,
  "threshold": 0.5,
  "tp": 740925,
  "fp": 250905,
  "fn": 177143,
  "tn": 7088563
}
```
