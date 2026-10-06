from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.imd2020 import IMD2020Dataset

from training.train_stego_structured_v33 import (
    StructuredStegoDecoderV33,
    embed_payload,
    make_payload,
    IMAGE_SIZE,
    PAYLOAD_BITS,
    GRID_ROWS,
    GRID_COLS,
    STRENGTH,
)

CHECKPOINT = (
    ROOT / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

MANIFEST = (
    ROOT / "manifests"
    / "imd2020_manifest_e5_clean.csv"
)

DATA_ROOT = ROOT / "data" / "imd2020"

OUTPUT = (
    ROOT / "results" / "csv"
    / "e53r_0b_block_causality_val.csv"
)

SUMMARY = (
    ROOT / "results" / "csv"
    / "e53r_0b_block_causality_summary.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 20260921
BATCH_SIZE = 8

ATTACKS = (
    "erase",
    "copy_move",
    "roll",
)

# A block is considered touched if >=10% of its
# spatial area is inside the controlled tamper mask.
#
# This criterion is fixed before observing 0B results.
TOUCH_THRESHOLD = 0.10

EPS = 1e-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_boxes(batch_size):
    boxes = []

    for _ in range(batch_size):

        h = random.randint(
            int(IMAGE_SIZE * 0.15),
            int(IMAGE_SIZE * 0.30),
        )

        w = random.randint(
            int(IMAGE_SIZE * 0.15),
            int(IMAGE_SIZE * 0.30),
        )

        y0 = random.randint(
            0,
            IMAGE_SIZE - h,
        )

        x0 = random.randint(
            0,
            IMAGE_SIZE - w,
        )

        boxes.append(
            (y0, y0 + h, x0, x0 + w)
        )

    return boxes


def boxes_to_mask(boxes, device):

    masks = torch.zeros(
        len(boxes),
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    for i, (y0, y1, x0, x1) in enumerate(boxes):
        masks[
            i, :, y0:y1, x0:x1
        ] = 1.0

    return masks


def apply_attack(images, boxes, attack):

    out = images.clone()

    for i, (y0, y1, x0, x1) in enumerate(boxes):

        h = y1 - y0
        w = x1 - x0

        if attack == "erase":

            fill = out[i].mean(
                dim=(1, 2),
                keepdim=True,
            )

            out[
                i, :, y0:y1, x0:x1
            ] = fill

        elif attack == "copy_move":

            # Choose source coordinates deterministically
            # under the fixed random seed.
            sy0 = random.randint(
                0,
                IMAGE_SIZE - h,
            )

            sx0 = random.randint(
                0,
                IMAGE_SIZE - w,
            )

            patch = out[
                i,
                :,
                sy0:sy0 + h,
                sx0:sx0 + w,
            ].clone()

            out[
                i, :, y0:y1, x0:x1
            ] = patch

        elif attack == "roll":

            patch = out[
                i, :, y0:y1, x0:x1
            ].clone()

            shift = max(
                1,
                w // 2,
            )

            patch = torch.roll(
                patch,
                shifts=shift,
                dims=2,
            )

            out[
                i, :, y0:y1, x0:x1
            ] = patch

        else:
            raise ValueError(
                f"Unknown attack: {attack}"
            )

    return out


def logits_grid(model, images):

    residual = model.highpass(images)

    combined = torch.cat(
        [images, residual],
        dim=1,
    )

    logits_map = model.features(
        combined
    )

    grid = model.pool(
        logits_map
    )

    return grid.flatten(start_dim=1)


def block_touch_fraction(mask):

    # Average the binary tamper mask over the exact
    # 8x16 provenance layout.
    block_fraction = F.adaptive_avg_pool2d(
        mask,
        (GRID_ROWS, GRID_COLS),
    )

    return block_fraction.flatten(
        start_dim=1
    )


def safe_auc(labels, scores):

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if np.unique(labels).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def main():

    print("=" * 78)
    print(
        "E5.3R-0B BLOCK-LEVEL PROVENANCE CAUSALITY TEST"
    )
    print("=" * 78)

    print("Device          :", DEVICE)
    print("Split           : validation only")
    print("Seed            :", SEED)
    print("Payload         : 128 bits")
    print("Grid            : 8 x 16")
    print("Touch threshold :", TOUCH_THRESHOLD)
    print("Attacks         :", ", ".join(ATTACKS))
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    set_seed(SEED)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    assert checkpoint["payload_bits"] == PAYLOAD_BITS == 128
    assert checkpoint["grid_rows"] == GRID_ROWS == 8
    assert checkpoint["grid_cols"] == GRID_COLS == 16
    assert abs(
        float(checkpoint["strength"]) - STRENGTH
    ) < 1e-12
    assert checkpoint["version"] == "V3.3"

    model = StructuredStegoDecoderV33().to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    assert sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ) == 0

    print("Checkpoint integrity : PASS")
    print("Decoder frozen       : PASS")
    print()

    dataset = IMD2020Dataset(
        root=DATA_ROOT,
        manifest=MANIFEST,
        split="val",
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    rows = []

    # --------------------------------------------------------
    # Attack loops are independent and reproducible.
    # Reset seed per attack so each attack receives the same
    # payload/box sequence where possible.
    # --------------------------------------------------------

    with torch.no_grad():

        for attack_index, attack in enumerate(ATTACKS):

            set_seed(
                SEED + attack_index * 1000
            )

            print(
                f"Running attack: {attack}"
            )

            for batch in loader:

                images = batch["image"].to(
                    DEVICE
                )

                batch_size = images.shape[0]

                payload = make_payload(
                    batch_size,
                    DEVICE,
                )

                protected = embed_payload(
                    images,
                    payload,
                )

                boxes = make_boxes(
                    batch_size
                )

                mask = boxes_to_mask(
                    boxes,
                    DEVICE,
                )

                tampered = apply_attack(
                    protected,
                    boxes,
                    attack,
                )

                clean_logits = logits_grid(
                    model,
                    protected,
                )

                tampered_logits = logits_grid(
                    model,
                    tampered,
                )

                clean_prob = torch.sigmoid(
                    clean_logits
                )

                tampered_prob = torch.sigmoid(
                    tampered_logits
                )

                # ----------------------------------------
                # Ground-truth bit correctness
                # ----------------------------------------

                clean_pred = (
                    clean_prob >= 0.5
                ).float()

                tampered_pred = (
                    tampered_prob >= 0.5
                ).float()

                clean_error = (
                    clean_pred != payload
                ).float()

                tampered_error = (
                    tampered_pred != payload
                ).float()

                # New errors attributable to attack:
                # clean was correct, tampered becomes wrong.
                induced_error = (
                    (clean_error == 0)
                    & (tampered_error == 1)
                ).float()

                # ----------------------------------------
                # Continuous degradation score
                #
                # Probability assigned to the TRUE bit
                # before vs after tampering.
                # ----------------------------------------

                clean_true_prob = (
                    payload * clean_prob
                    + (1.0 - payload)
                    * (1.0 - clean_prob)
                )

                tampered_true_prob = (
                    payload * tampered_prob
                    + (1.0 - payload)
                    * (1.0 - tampered_prob)
                )

                degradation = (
                    clean_true_prob
                    - tampered_true_prob
                )

                # Positive means tampering reduced confidence
                # in the correct payload bit.

                touch_fraction = (
                    block_touch_fraction(mask)
                )

                touched = (
                    touch_fraction
                    >= TOUCH_THRESHOLD
                )

                sample_ids = batch[
                    "sample_id"
                ]

                for i in range(batch_size):

                    touched_i = touched[i]
                    untouched_i = ~touched_i

                    n_touched = int(
                        touched_i.sum().item()
                    )

                    n_untouched = int(
                        untouched_i.sum().item()
                    )

                    if (
                        n_touched == 0
                        or n_untouched == 0
                    ):
                        continue

                    touched_ber = float(
                        tampered_error[i][
                            touched_i
                        ].mean().cpu()
                    )

                    untouched_ber = float(
                        tampered_error[i][
                            untouched_i
                        ].mean().cpu()
                    )

                    clean_touched_ber = float(
                        clean_error[i][
                            touched_i
                        ].mean().cpu()
                    )

                    clean_untouched_ber = float(
                        clean_error[i][
                            untouched_i
                        ].mean().cpu()
                    )

                    induced_touched = float(
                        induced_error[i][
                            touched_i
                        ].mean().cpu()
                    )

                    induced_untouched = float(
                        induced_error[i][
                            untouched_i
                        ].mean().cpu()
                    )

                    deg_touched = float(
                        degradation[i][
                            touched_i
                        ].mean().cpu()
                    )

                    deg_untouched = float(
                        degradation[i][
                            untouched_i
                        ].mean().cpu()
                    )

                    delta_ber = (
                        touched_ber
                        - untouched_ber
                    )

                    delta_induced = (
                        induced_touched
                        - induced_untouched
                    )

                    delta_deg = (
                        deg_touched
                        - deg_untouched
                    )

                    labels = (
                        touched_i
                        .float()
                        .cpu()
                        .numpy()
                    )

                    scores = (
                        degradation[i]
                        .cpu()
                        .numpy()
                    )

                    auc = safe_auc(
                        labels,
                        scores,
                    )

                    rows.append(
                        {
                            "sample_id":
                                int(sample_ids[i]),

                            "attack":
                                attack,

                            "n_touched":
                                n_touched,

                            "n_untouched":
                                n_untouched,

                            "clean_touched_ber":
                                clean_touched_ber,

                            "clean_untouched_ber":
                                clean_untouched_ber,

                            "tampered_touched_ber":
                                touched_ber,

                            "tampered_untouched_ber":
                                untouched_ber,

                            "delta_ber":
                                delta_ber,

                            "induced_touched_error":
                                induced_touched,

                            "induced_untouched_error":
                                induced_untouched,

                            "delta_induced_error":
                                delta_induced,

                            "degradation_touched":
                                deg_touched,

                            "degradation_untouched":
                                deg_untouched,

                            "delta_degradation":
                                delta_deg,

                            "block_localization_auc":
                                auc,

                            "mean_touch_fraction":
                                float(
                                    touch_fraction[i][
                                        touched_i
                                    ].mean().cpu()
                                ),
                        }
                    )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No valid samples evaluated."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Per-attack summary
    # --------------------------------------------------------

    summary_rows = []

    print()
    print("-" * 78)
    print("PER-ATTACK RESULTS")
    print("-" * 78)

    for attack in ATTACKS:

        part = df[
            df["attack"] == attack
        ]

        median_delta_ber = float(
            part["delta_ber"].median()
        )

        positive_delta_ber = float(
            (
                part["delta_ber"] > 0
            ).mean()
        )

        median_delta_induced = float(
            part[
                "delta_induced_error"
            ].median()
        )

        positive_delta_induced = float(
            (
                part[
                    "delta_induced_error"
                ] > 0
            ).mean()
        )

        median_delta_deg = float(
            part[
                "delta_degradation"
            ].median()
        )

        positive_delta_deg = float(
            (
                part[
                    "delta_degradation"
                ] > 0
            ).mean()
        )

        median_auc = float(
            part[
                "block_localization_auc"
            ].median()
        )

        mean_auc = float(
            part[
                "block_localization_auc"
            ].mean()
        )

        # --------------------------------------------
        # Pre-registered attack-level gate
        #
        # Primary evidence uses CONTINUOUS true-bit
        # confidence degradation rather than only
        # hard threshold bit errors.
        # --------------------------------------------

        g1 = (
            median_delta_deg > 0.0
        )

        g2 = (
            positive_delta_deg > 0.50
        )

        g3 = (
            median_auc > 0.50
        )

        attack_pass = all(
            [g1, g2, g3]
        )

        summary_rows.append(
            {
                "attack":
                    attack,

                "n_samples":
                    len(part),

                "median_delta_ber":
                    median_delta_ber,

                "delta_ber_gt0_fraction":
                    positive_delta_ber,

                "median_delta_induced_error":
                    median_delta_induced,

                "delta_induced_gt0_fraction":
                    positive_delta_induced,

                "median_delta_degradation":
                    median_delta_deg,

                "delta_degradation_gt0_fraction":
                    positive_delta_deg,

                "median_block_auc":
                    median_auc,

                "mean_block_auc":
                    mean_auc,

                "gate_delta_degradation":
                    g1,

                "gate_positive_fraction":
                    g2,

                "gate_auc":
                    g3,

                "attack_gate":
                    (
                        "PASS"
                        if attack_pass
                        else "FAIL"
                    ),
            }
        )

        print()
        print(
            f"[{attack.upper()}]"
        )

        print(
            "Median ΔBER                  :",
            f"{median_delta_ber:.6f}",
        )

        print(
            "ΔBER > 0                     :",
            f"{positive_delta_ber*100:.2f}%",
        )

        print(
            "Median induced-error Δ       :",
            f"{median_delta_induced:.6f}",
        )

        print(
            "Induced-error Δ > 0          :",
            f"{positive_delta_induced*100:.2f}%",
        )

        print(
            "Median confidence degradation:",
            f"{median_delta_deg:.6f}",
        )

        print(
            "Degradation Δ > 0            :",
            f"{positive_delta_deg*100:.2f}%",
        )

        print(
            "Median block localization AUC :",
            f"{median_auc:.6f}",
        )

        print(
            "Mean block localization AUC   :",
            f"{mean_auc:.6f}",
        )

        print(
            "Attack gate                   :",
            "PASS"
            if attack_pass
            else "FAIL",
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # Overall gate deliberately requires all three
    # attack classes to pass. No averaging can hide
    # an attack-specific failure.
    overall_pass = bool(
        (
            summary_df["attack_gate"]
            == "PASS"
        ).all()
    )

    summary_df[
        "overall_gate"
    ] = (
        "PASS"
        if overall_pass
        else "FAIL"
    )

    summary_df.to_csv(
        SUMMARY,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "E5.3R-0B FINAL GATE:",
        "PASS"
        if overall_pass
        else "FAIL",
    )
    print("=" * 78)

    print()
    print("Per-sample CSV :", OUTPUT)
    print("Summary CSV    :", SUMMARY)

    print()
    print(
        "Interpretation boundary:"
    )
    print(
        "This evaluates block-level provenance-channel "
        "causality, NOT final forgery localization performance."
    )


if __name__ == "__main__":
    main()
