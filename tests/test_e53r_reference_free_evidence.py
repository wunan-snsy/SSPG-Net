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
    / "e53r_1_reference_free_evidence_val.csv"
)

SUMMARY = (
    ROOT / "results" / "csv"
    / "e53r_1_reference_free_evidence_summary.csv"
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

TOUCH_THRESHOLD = 0.10


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

    mask = torch.zeros(
        len(boxes),
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    for i, (y0, y1, x0, x1) in enumerate(boxes):
        mask[i, :, y0:y1, x0:x1] = 1.0

    return mask


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

            out[i, :, y0:y1, x0:x1] = fill

        elif attack == "copy_move":

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

            out[i, :, y0:y1, x0:x1] = patch

        elif attack == "roll":

            patch = out[
                i, :, y0:y1, x0:x1
            ].clone()

            shift = max(1, w // 2)

            patch = torch.roll(
                patch,
                shifts=shift,
                dims=2,
            )

            out[i, :, y0:y1, x0:x1] = patch

        else:
            raise ValueError(attack)

    return out


def decode_prob(model, images):

    logits = model(images)

    return torch.sigmoid(logits)


def reference_free_evidence(prob, payload):
    """
    Evidence of disagreement between the known embedded
    payload and the payload decoded from the received image.

    true_prob_j =
        b_j * p_j + (1-b_j) * (1-p_j)

    evidence_j = 1 - true_prob_j

    Range: [0, 1]

    IMPORTANT:
    No clean reference image is used here.
    """

    true_prob = (
        payload * prob
        + (1.0 - payload)
        * (1.0 - prob)
    )

    return 1.0 - true_prob


def block_touch_fraction(mask):

    return F.adaptive_avg_pool2d(
        mask,
        (GRID_ROWS, GRID_COLS),
    ).flatten(start_dim=1)


def safe_auc(labels, scores):

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if np.unique(labels).size < 2:
        return np.nan

    return float(
        roc_auc_score(labels, scores)
    )


def main():

    print("=" * 78)
    print(
        "E5.3R-1 REFERENCE-FREE PROVENANCE EVIDENCE TEST"
    )
    print("=" * 78)

    print("Device          :", DEVICE)
    print("Split           : validation only")
    print("Seed            :", SEED)
    print("Payload         :", PAYLOAD_BITS, "bits")
    print(
        "Grid            :",
        f"{GRID_ROWS} x {GRID_COLS}"
    )
    print("Alpha           :", STRENGTH)
    print("Touch threshold :", TOUCH_THRESHOLD)
    print(
        "Attacks         :",
        ", ".join(ATTACKS)
    )
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    assert checkpoint["payload_bits"] == 128
    assert checkpoint["grid_rows"] == 8
    assert checkpoint["grid_cols"] == 16
    assert checkpoint["version"] == "V3.3"
    assert abs(
        float(checkpoint["strength"]) - STRENGTH
    ) < 1e-12

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
    print(
        "Clean reference input : FORBIDDEN"
    )
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

    with torch.no_grad():

        for attack_index, attack in enumerate(ATTACKS):

            # Independent, fixed sequence per attack.
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

                n = images.shape[0]

                payload = make_payload(
                    n,
                    DEVICE,
                )

                protected = embed_payload(
                    images,
                    payload,
                )

                boxes = make_boxes(n)

                mask = boxes_to_mask(
                    boxes,
                    DEVICE,
                )

                received = apply_attack(
                    protected,
                    boxes,
                    attack,
                )

                # ----------------------------------------
                # Decoder observations
                #
                # clean_prob is ONLY used as a clean
                # negative-control statistic.
                #
                # It is NOT used to construct received
                # evidence or localization scores.
                # ----------------------------------------

                clean_prob = decode_prob(
                    model,
                    protected,
                )

                received_prob = decode_prob(
                    model,
                    received,
                )

                clean_evidence = (
                    reference_free_evidence(
                        clean_prob,
                        payload,
                    )
                )

                received_evidence = (
                    reference_free_evidence(
                        received_prob,
                        payload,
                    )
                )

                # Critical guard:
                # localization score is received_evidence
                # directly. No subtraction from clean.
                localization_score = (
                    received_evidence
                )

                touch_fraction = (
                    block_touch_fraction(mask)
                )

                touched = (
                    touch_fraction
                    >= TOUCH_THRESHOLD
                )

                sample_ids = batch["sample_id"]

                for i in range(n):

                    t = touched[i]
                    u = ~t

                    if (
                        int(t.sum()) == 0
                        or int(u.sum()) == 0
                    ):
                        continue

                    clean_t = float(
                        clean_evidence[i][t]
                        .mean()
                        .cpu()
                    )

                    clean_u = float(
                        clean_evidence[i][u]
                        .mean()
                        .cpu()
                    )

                    recv_t = float(
                        received_evidence[i][t]
                        .mean()
                        .cpu()
                    )

                    recv_u = float(
                        received_evidence[i][u]
                        .mean()
                        .cpu()
                    )

                    delta_received = (
                        recv_t - recv_u
                    )

                    labels = (
                        t.float()
                        .cpu()
                        .numpy()
                    )

                    scores = (
                        localization_score[i]
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
                                int(t.sum()),

                            "n_untouched":
                                int(u.sum()),

                            "clean_touched_evidence":
                                clean_t,

                            "clean_untouched_evidence":
                                clean_u,

                            "received_touched_evidence":
                                recv_t,

                            "received_untouched_evidence":
                                recv_u,

                            "delta_received_evidence":
                                delta_received,

                            "block_auc":
                                auc,

                            "clean_global_evidence":
                                float(
                                    clean_evidence[i]
                                    .mean()
                                    .cpu()
                                ),

                            "received_global_evidence":
                                float(
                                    received_evidence[i]
                                    .mean()
                                    .cpu()
                                ),

                            "mean_touch_fraction":
                                float(
                                    touch_fraction[i][t]
                                    .mean()
                                    .cpu()
                                ),
                        }
                    )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No valid samples."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    summary_rows = []

    print()
    print("-" * 78)
    print("REFERENCE-FREE RESULTS")
    print("-" * 78)

    for attack in ATTACKS:

        part = df[
            df["attack"] == attack
        ]

        median_delta = float(
            part[
                "delta_received_evidence"
            ].median()
        )

        positive_fraction = float(
            (
                part[
                    "delta_received_evidence"
                ] > 0
            ).mean()
        )

        median_auc = float(
            part["block_auc"].median()
        )

        mean_auc = float(
            part["block_auc"].mean()
        )

        mean_clean_global = float(
            part[
                "clean_global_evidence"
            ].mean()
        )

        mean_received_global = float(
            part[
                "received_global_evidence"
            ].mean()
        )

        global_increase = (
            mean_received_global
            - mean_clean_global
        )

        # --------------------------------------------
        # PRE-REGISTERED E5.3R-1 GATE
        # --------------------------------------------

        g1 = median_delta > 0.0

        g2 = (
            positive_fraction > 0.50
        )

        g3 = median_auc > 0.50

        g4 = global_increase > 0.0

        attack_pass = all(
            [g1, g2, g3, g4]
        )

        summary_rows.append(
            {
                "attack":
                    attack,

                "n_samples":
                    len(part),

                "median_delta_received_evidence":
                    median_delta,

                "positive_delta_fraction":
                    positive_fraction,

                "median_block_auc":
                    median_auc,

                "mean_block_auc":
                    mean_auc,

                "mean_clean_global_evidence":
                    mean_clean_global,

                "mean_received_global_evidence":
                    mean_received_global,

                "global_evidence_increase":
                    global_increase,

                "g1_delta_positive":
                    g1,

                "g2_majority_positive":
                    g2,

                "g3_auc_above_chance":
                    g3,

                "g4_global_increase":
                    g4,

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
            "Median touched-untouched evidence :",
            f"{median_delta:.6f}",
        )

        print(
            "Positive spatial contrast         :",
            f"{positive_fraction*100:.2f}%",
        )

        print(
            "Median reference-free block AUC   :",
            f"{median_auc:.6f}",
        )

        print(
            "Mean reference-free block AUC     :",
            f"{mean_auc:.6f}",
        )

        print(
            "Mean clean global evidence        :",
            f"{mean_clean_global:.6f}",
        )

        print(
            "Mean received global evidence     :",
            f"{mean_received_global:.6f}",
        )

        print(
            "Global evidence increase          :",
            f"{global_increase:.6f}",
        )

        print(
            "Attack gate                       :",
            "PASS"
            if attack_pass
            else "FAIL",
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

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
    print("-" * 78)
    print("PRE-REGISTERED CONDITIONS")
    print("-" * 78)

    print(
        "G1 median touched evidence > untouched : > 0"
    )

    print(
        "G2 positive spatial contrast            : > 50%"
    )

    print(
        "G3 median block AUC                     : > 0.50"
    )

    print(
        "G4 received global evidence             : > clean"
    )

    print()
    print("=" * 78)

    print(
        "E5.3R-1 FINAL GATE:",
        "PASS"
        if overall_pass
        else "FAIL",
    )

    print("=" * 78)

    print()
    print(
        "Per-sample CSV :",
        OUTPUT,
    )

    print(
        "Summary CSV    :",
        SUMMARY,
    )

    print()
    print(
        "CRITICAL INTERPRETATION:"
    )

    print(
        "Localization scores use ONLY received image "
        "+ known embedded payload."
    )

    print(
        "Clean protected images are used only as a "
        "negative-control statistic, never as an input "
        "to the localization score."
    )


if __name__ == "__main__":
    main()
