# Three-seed, parameter-matched NO_GRAPH comparison

The three matched-control runs pair by seed with previously completed FULL E2 runs. The preliminary single-seed run is excluded. All checkpoints were chosen by Validation joint loss. Test and E9 were not used for training or selection.

| Seed | FULL Pixel F1 | Matched NO_GRAPH Pixel F1 | FULL - matched | FULL AUC | Matched AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261001 | 0.7536 | 0.7010 | +0.0526 | 0.9990 | 0.9891 |
| 20261002 | 0.7655 | 0.6550 | +0.1105 | 0.9953 | 0.9849 |
| 20261003 | 0.7530 | 0.7294 | +0.0236 | 0.9901 | 0.9867 |

FULL Pixel F1: 0.7574 ± 0.0070; matched NO_GRAPH: 0.6951 ± 0.0376; paired FULL - matched: +0.0622 ± 0.0443 (sample SD across three seeds).
The smaller existing NO_GRAPH runs scored 0.7217 ± 0.0072 Pixel F1, for context.

The matched control has 1,416,110 trainable parameters versus FULL's 1,416,490 (-0.027%). It applies two per-token residual MLP blocks with no token-to-token exchange. Its results test whether the FULL advantage survives this specific capacity match. They do not establish that graph topology, rather than spatial communication more generally, is uniquely responsible. Three seeds describe training-seed variability on one fixed Validation split; they do not measure cross-dataset generalization. The already inspected E9 set remains exploratory for this newly designed control.
