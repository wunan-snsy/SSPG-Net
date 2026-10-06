from pathlib import Path
import sys
import csv
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    / "e53r_0a_spatial_causality_val.csv"
)

SUMMARY = (
    ROOT / "results" / "csv"
    / "e53r_0a_spatial_causality_summary.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 20260921
EPS = 1e-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spatial_logits(model, x):

    residual = model.highpass(x)

    combined = torch.cat(
        [x, residual],
        dim=1,
    )

    return model.features(combined)


def make_box_mask(
    batch_size,
    height,
    width,
    device,
):

    """
    Deterministic-under-seed rectangular tampering masks.

    Area is intentionally moderate:
    approximately 10%-25% of image dimensions per axis.

    This is a controlled mechanism test,
    not the final attack benchmark.
    """

    masks = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=device,
    )

    boxes = []

    for i in range(batch_size):

        bh = random.randint(
            int(height * 0.15),
            int(height * 0.30),
        )

        bw = random.randint(
            int(width * 0.15),
            int(width * 0.30),
        )

        y0 = random.randint(
            0,
            height - bh,
        )

        x0 = random.randint(
            0,
            width - bw,
        )

        y1 = y0 + bh
        x1 = x0 + bw

        masks[
            i, :, y0:y1, x0:x1
        ] = 1.0

        boxes.append(
            (y0, y1, x0, x1)
        )

    return masks, boxes


def apply_identical_tamper(
    protected,
    unprotected,
    boxes,
):

    """
    Apply the SAME spatial operation to protected and
    unprotected versions.

    Operation: erase using each image's own channel mean.

    This avoids introducing an arbitrary RGB constant.
    """

    p = protected.clone()
    u = unprotected.clone()

    for i, (y0, y1, x0, x1) in enumerate(boxes):

        p_fill = p[i].mean(
            dim=(1, 2),
            keepdim=True,
        )

        u_fill = u[i].mean(
            dim=(1, 2),
            keepdim=True,
        )

        p[
            i, :, y0:y1, x0:x1
        ] = p_fill

        u[
            i, :, y0:y1, x0:x1
        ] = u_fill

    return p, u


def region_means(degradation, mask):

    inside = (
        degradation * mask
    ).sum(dim=(1, 2, 3))

    inside_n = mask.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)

    outside_mask = 1.0 - mask

    outside = (
        degradation * outside_mask
    ).sum(dim=(1, 2, 3))

    outside_n = outside_mask.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)

    return (
        inside / inside_n,
        outside / outside_n,
    )


def bit_accuracy(logits, payload):

    pred = (
        torch.sigmoid(logits) >= 0.5
    ).float()

    return (
        pred == payload
    ).float().mean(dim=1)


def main():

    print("=" * 76)
    print(
        "E5.3R-0A ACTIVE PHYSICAL TRACE "
        "SPATIAL CAUSALITY TEST"
    )
    print("=" * 76)

    print("Device     :", DEVICE)
    print("Split      : validation only")
    print("Seed       :", SEED)
    print("Checkpoint :", CHECKPOINT)
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

    assert checkpoint["payload_bits"] == 128
    assert checkpoint["grid_rows"] == 8
    assert checkpoint["grid_cols"] == 16
    assert abs(
        float(checkpoint["strength"])
        - STRENGTH
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

    for p in model.parameters():
        p.requires_grad_(False)

    assert sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ) == 0

    print("Checkpoint integrity : PASS")
    print("Decoder frozen       : PASS")
    print("Payload bits         :", PAYLOAD_BITS)
    print(
        "Grid                 :",
        GRID_ROWS,
        "x",
        GRID_COLS,
    )
    print("Alpha                :", STRENGTH)
    print()

    # --------------------------------------------------------
    # IMPORTANT:
    # Validation only. No formal Test access.
    # --------------------------------------------------------

    dataset = IMD2020Dataset(
        root=DATA_ROOT,
        manifest=MANIFEST,
        split="val",
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    rows = []

    with torch.no_grad():

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

            masks, boxes = make_box_mask(
                batch_size,
                IMAGE_SIZE,
                IMAGE_SIZE,
                DEVICE,
            )

            (
                protected_tampered,
                unprotected_tampered,
            ) = apply_identical_tamper(
                protected,
                images,
                boxes,
            )

            # --------------------------------------------
            # Spatial decoder response
            # --------------------------------------------

            lp_clean = spatial_logits(
                model,
                protected,
            )

            lp_tampered = spatial_logits(
                model,
                protected_tampered,
            )

            lu_clean = spatial_logits(
                model,
                images,
            )

            lu_tampered = spatial_logits(
                model,
                unprotected_tampered,
            )

            # Decoder-response degradation maps
            dp = (
                torch.sigmoid(lp_tampered)
                - torch.sigmoid(lp_clean)
            ).abs()

            du = (
                torch.sigmoid(lu_tampered)
                - torch.sigmoid(lu_clean)
            ).abs()

            # spatial_logits is expected to remain HxW
            assert dp.shape[-2:] == (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

            assert du.shape[-2:] == (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

            (
                p_inside,
                p_outside,
            ) = region_means(
                dp,
                masks,
            )

            (
                u_inside,
                u_outside,
            ) = region_means(
                du,
                masks,
            )

            p_ratio = (
                p_inside
                / (p_outside + EPS)
            )

            u_ratio = (
                u_inside
                / (u_outside + EPS)
            )

            trace_advantage = (
                p_ratio - u_ratio
            )

            # --------------------------------------------
            # Bit-level diagnostic
            # --------------------------------------------

            protected_clean_logits = (
                model(protected)
            )

            protected_tampered_logits = (
                model(protected_tampered)
            )

            clean_ba = bit_accuracy(
                protected_clean_logits,
                payload,
            )

            tampered_ba = bit_accuracy(
                protected_tampered_logits,
                payload,
            )

            bit_drop = (
                clean_ba - tampered_ba
            )

            # --------------------------------------------
            # Save every sample
            # --------------------------------------------

            sample_ids = batch["sample_id"]

            for i in range(batch_size):

                y0, y1, x0, x1 = boxes[i]

                rows.append(
                    {
                        "sample_id":
                            int(sample_ids[i]),

                        "protected_inside":
                            float(
                                p_inside[i].cpu()
                            ),

                        "protected_outside":
                            float(
                                p_outside[i].cpu()
                            ),

                        "protected_ratio":
                            float(
                                p_ratio[i].cpu()
                            ),

                        "unprotected_inside":
                            float(
                                u_inside[i].cpu()
                            ),

                        "unprotected_outside":
                            float(
                                u_outside[i].cpu()
                            ),

                        "unprotected_ratio":
                            float(
                                u_ratio[i].cpu()
                            ),

                        "trace_advantage":
                            float(
                                trace_advantage[i].cpu()
                            ),

                        "clean_bit_accuracy":
                            float(
                                clean_ba[i].cpu()
                            ),

                        "tampered_bit_accuracy":
                            float(
                                tampered_ba[i].cpu()
                            ),

                        "bit_accuracy_drop":
                            float(
                                bit_drop[i].cpu()
                            ),

                        "y0": y0,
                        "y1": y1,
                        "x0": x0,
                        "x1": x1,

                        "mask_area_fraction":
                            float(
                                masks[i]
                                .mean()
                                .cpu()
                            ),
                    }
                )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No validation samples evaluated."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    median_cp = float(
        df["protected_ratio"].median()
    )

    median_cu = float(
        df["unprotected_ratio"].median()
    )

    median_adv = float(
        df["trace_advantage"].median()
    )

    protected_positive_fraction = float(
        (
            df["protected_ratio"] > 1.0
        ).mean()
    )

    advantage_positive_fraction = float(
        (
            df["trace_advantage"] > 0.0
        ).mean()
    )

    mean_clean_ba = float(
        df["clean_bit_accuracy"].mean()
    )

    mean_tampered_ba = float(
        df["tampered_bit_accuracy"].mean()
    )

    mean_bit_drop = float(
        df["bit_accuracy_drop"].mean()
    )

    # --------------------------------------------------------
    # PRE-REGISTERED GATE
    # Do not modify after seeing results.
    # --------------------------------------------------------

    gate_1 = median_cp > 1.0

    gate_2 = (
        protected_positive_fraction > 0.50
    )

    gate_3 = median_adv > 0.0

    gate_4 = (
        advantage_positive_fraction > 0.50
    )

    gate_pass = all(
        [
            gate_1,
            gate_2,
            gate_3,
            gate_4,
        ]
    )

    summary = {
        "experiment":
            "E5.3R-0A",

        "split":
            "validation",

        "n_samples":
            len(df),

        "seed":
            SEED,

        "payload_bits":
            PAYLOAD_BITS,

        "grid":
            f"{GRID_ROWS}x{GRID_COLS}",

        "strength":
            STRENGTH,

        "median_protected_ratio":
            median_cp,

        "median_unprotected_ratio":
            median_cu,

        "median_trace_advantage":
            median_adv,

        "protected_ratio_gt1_fraction":
            protected_positive_fraction,

        "trace_advantage_gt0_fraction":
            advantage_positive_fraction,

        "mean_clean_bit_accuracy":
            mean_clean_ba,

        "mean_tampered_bit_accuracy":
            mean_tampered_ba,

        "mean_bit_accuracy_drop":
            mean_bit_drop,

        "gate_1":
            gate_1,

        "gate_2":
            gate_2,

        "gate_3":
            gate_3,

        "gate_4":
            gate_4,

        "final_gate":
            "PASS"
            if gate_pass
            else "FAIL",
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY,
        index=False,
    )

    print("-" * 76)
    print("SPATIAL DEGRADATION")
    print("-" * 76)

    print(
        "Samples                         :",
        len(df),
    )

    print(
        "Median protected ratio          :",
        f"{median_cp:.6f}",
    )

    print(
        "Protected ratio > 1             :",
        f"{protected_positive_fraction*100:.2f}%",
    )

    print(
        "Median unprotected ratio        :",
        f"{median_cu:.6f}",
    )

    print(
        "Median trace advantage          :",
        f"{median_adv:.6f}",
    )

    print(
        "Protected > unprotected         :",
        f"{advantage_positive_fraction*100:.2f}%",
    )

    print()

    print(
        "Protected clean BitAcc          :",
        f"{mean_clean_ba*100:.4f}%",
    )

    print(
        "Protected tampered BitAcc       :",
        f"{mean_tampered_ba*100:.4f}%",
    )

    print(
        "Mean BitAcc drop                :",
        f"{mean_bit_drop*100:.4f} pp",
    )

    print()
    print("-" * 76)
    print("PRE-REGISTERED GATE")
    print("-" * 76)

    print(
        "G1 median protected ratio > 1   :",
        "PASS" if gate_1 else "FAIL",
    )

    print(
        "G2 protected ratio>1 >50%       :",
        "PASS" if gate_2 else "FAIL",
    )

    print(
        "G3 median trace advantage > 0   :",
        "PASS" if gate_3 else "FAIL",
    )

    print(
        "G4 protected>unprotected >50%   :",
        "PASS" if gate_4 else "FAIL",
    )

    print()
    print("=" * 76)

    print(
        "E5.3R-0A FINAL GATE:",
        "PASS" if gate_pass else "FAIL",
    )

    print("=" * 76)

    print()
    print("Per-sample CSV :", OUTPUT)
    print("Summary CSV    :", SUMMARY)

    print()
    print(
        "IMPORTANT: This is a mechanism-validation "
        "experiment, not final detection performance."
    )


if __name__ == "__main__":
    main()
