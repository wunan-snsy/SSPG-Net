from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.imd2020 import IMD2020Dataset


# ============================================================
# E4.12-A V3.3
# Residual-Aware 128-bit Block-Structured Provenance Channel
#
# Frozen relative to V3.2:
#   Payload       : 128 bits
#   Layout        : 8 x 16
#   Image size    : 256 x 256
#   Block size    : 32 x 16
#   Alpha         : 0.010
#   Epochs        : 20
#   Seed          : 42
#   Batch size    : 8
#   LR            : 1e-3
#   Weight decay  : 1e-5
#
# Main experimental change:
#   Decoder receives RGB + fixed high-pass residual features.
#
# Pre-registered screening criterion:
#   Best validation Bit Accuracy >= 90%
#
# V3.0 / V3.1 / V3.2 results must remain untouched.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "imd2020"

MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "imd2020_manifest.csv"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "csv"
)


BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "stego_structured_v33_128bit_best.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "stego_structured_v33_128bit_last.pth"
)

TRAINING_LOG = (
    RESULT_DIR
    / "stego_structured_v33_128bit_training_log.csv"
)


IMAGE_SIZE = 256

PAYLOAD_BITS = 128

GRID_ROWS = 8
GRID_COLS = 16

BLOCK_HEIGHT = IMAGE_SIZE // GRID_ROWS
BLOCK_WIDTH = IMAGE_SIZE // GRID_COLS

STRENGTH = 0.010

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

SEED = 42


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset helper
# ============================================================


def extract_images(batch):

    if torch.is_tensor(batch):

        images = batch

    elif isinstance(batch, dict):

        if "image" in batch:

            images = batch["image"]

        elif "images" in batch:

            images = batch["images"]

        else:

            raise KeyError(
                "Dataset dictionary does not contain "
                f"image/images. Keys={list(batch.keys())}"
            )

    elif isinstance(batch, (tuple, list)):

        if len(batch) == 0:

            raise RuntimeError(
                "Dataset returned empty batch."
            )

        images = batch[0]

    else:

        raise TypeError(
            f"Unsupported batch type: {type(batch)}"
        )

    if not torch.is_tensor(images):

        raise TypeError(
            "Extracted image is not Tensor."
        )

    return images


# ============================================================
# Payload generation
# ============================================================


def make_payload(batch_size, device):

    """
    128-bit logical provenance payload.

    Current controlled channel experiment:

      0-31   : node/image field
      32-63  : source field
      64-95  : operation/provenance field
      96-127 : integrity field

    Values remain independently random in E4.12-A so that
    semantic correlations cannot artificially improve decoding.
    """

    return torch.randint(
        low=0,
        high=2,
        size=(batch_size, PAYLOAD_BITS),
        device=device,
        dtype=torch.float32,
    )


# ============================================================
# Payload -> spatial block map
# ============================================================


def payload_to_grid(payload):

    return payload.view(
        payload.shape[0],
        1,
        GRID_ROWS,
        GRID_COLS,
    )


def payload_to_spatial_map(payload):

    grid = payload_to_grid(payload)

    spatial_map = F.interpolate(
        grid,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )

    return spatial_map


# ============================================================
# Embedding
# ============================================================


def embed_payload(images, payload):

    """
    Same embedding rule as V3.2.

    bit 0 -> -alpha
    bit 1 -> +alpha

    Each bit occupies one fixed 32x16 spatial block.

    IMPORTANT:
    Encoder is unchanged relative to V3.2.
    """

    spatial_bits = payload_to_spatial_map(
        payload
    )

    signed_map = (
        spatial_bits * 2.0
        - 1.0
    )

    residual_rgb = signed_map.repeat(
        1,
        3,
        1,
        1,
    )

    stego = torch.clamp(
        images
        + STRENGTH * residual_rgb,
        0.0,
        1.0,
    )

    return stego


# ============================================================
# Fixed high-pass residual extractor
# ============================================================


class FixedHighPass(nn.Module):

    """
    Fixed, non-trainable high-pass residual extractor.

    Three classical residual kernels are applied independently
    to each RGB channel.

    Input:
        [B, 3, H, W]

    Output:
        [B, 9, H, W]

    The purpose is not to recover the clean image.
    It is to expose weak local residual structure before the
    trainable decoder.
    """

    def __init__(self):

        super().__init__()

        k1 = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        k2 = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 8.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=torch.float32,
        )

        k3 = torch.tensor(
            [
                [-1.0, 2.0, -1.0],
                [2.0, -4.0, 2.0],
                [-1.0, 2.0, -1.0],
            ],
            dtype=torch.float32,
        )

        kernels = torch.stack(
            [k1, k2, k3],
            dim=0,
        )

        # [3 kernels, 1, 3, 3]
        kernels = kernels.unsqueeze(1)

        # Apply the same three filters separately to
        # R, G and B using grouped convolution.
        #
        # Required layout:
        # [9, 1, 3, 3]
        kernels = kernels.repeat(
            3,
            1,
            1,
            1,
        )

        self.register_buffer(
            "kernels",
            kernels,
        )

    def forward(self, x):

        residual = F.conv2d(
            x,
            self.kernels,
            bias=None,
            stride=1,
            padding=1,
            groups=3,
        )

        return residual


# ============================================================
# V3.3 Residual-Aware Decoder
# ============================================================


class StructuredStegoDecoderV33(nn.Module):

    def __init__(self):

        super().__init__()

        self.highpass = FixedHighPass()

        # 3 RGB channels + 9 residual channels = 12 channels.
        self.features = nn.Sequential(

            nn.Conv2d(
                12,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),

            nn.Conv2d(
                16,
                1,
                kernel_size=1,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (GRID_ROWS, GRID_COLS)
        )

    def forward(self, x):

        residual = self.highpass(x)

        combined = torch.cat(
            [x, residual],
            dim=1,
        )

        logits_map = self.features(
            combined
        )

        logits_grid = self.pool(
            logits_map
        )

        logits = logits_grid.flatten(
            start_dim=1
        )

        return logits


# ============================================================
# Metrics
# ============================================================


def bit_statistics(logits, payload):

    predictions = (
        torch.sigmoid(logits) >= 0.5
    ).float()

    correct_bits = (
        predictions == payload
    ).sum().item()

    total_bits = payload.numel()

    exact_messages = (
        predictions == payload
    ).all(dim=1)

    exact_correct = (
        exact_messages.sum().item()
    )

    total_messages = payload.shape[0]

    return (
        correct_bits,
        total_bits,
        exact_correct,
        total_messages,
    )


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    loss_sum = 0.0

    correct_bits = 0
    total_bits = 0

    exact_correct = 0
    total_messages = 0

    batches = 0

    for batch in loader:

        images = extract_images(batch)

        images = images.to(
            DEVICE,
            dtype=torch.float32,
        )

        payload = make_payload(
            images.shape[0],
            DEVICE,
        )

        stego = embed_payload(
            images,
            payload,
        )

        logits = model(stego)

        loss = criterion(
            logits,
            payload,
        )

        (
            batch_correct_bits,
            batch_total_bits,
            batch_exact_correct,
            batch_total_messages,
        ) = bit_statistics(
            logits,
            payload,
        )

        loss_sum += loss.item()

        correct_bits += (
            batch_correct_bits
        )

        total_bits += (
            batch_total_bits
        )

        exact_correct += (
            batch_exact_correct
        )

        total_messages += (
            batch_total_messages
        )

        batches += 1

    mean_loss = (
        loss_sum / batches
    )

    bit_accuracy = (
        correct_bits / total_bits
    )

    exact_accuracy = (
        exact_correct / total_messages
    )

    return (
        mean_loss,
        bit_accuracy,
        exact_accuracy,
    )


# ============================================================
# PSNR diagnostic
# ============================================================


@torch.no_grad()
def estimate_psnr(loader):

    squared_error_sum = 0.0
    total_values = 0

    max_batches = 20

    for batch_index, batch in enumerate(loader):

        if batch_index >= max_batches:
            break

        images = extract_images(batch)

        images = images.to(
            DEVICE,
            dtype=torch.float32,
        )

        payload = make_payload(
            images.shape[0],
            DEVICE,
        )

        stego = embed_payload(
            images,
            payload,
        )

        difference = (
            stego - images
        )

        squared_error_sum += (
            difference
            .pow(2)
            .sum()
            .item()
        )

        total_values += (
            difference.numel()
        )

    mse = (
        squared_error_sum
        / total_values
    )

    psnr = (
        10.0
        * np.log10(
            1.0 / mse
        )
    )

    return mse, psnr


# ============================================================
# Main
# ============================================================


def main():

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "SSPG STRUCTURED STEGO V3.3"
    )

    print(
        "E4.12-A V3.3: RESIDUAL-AWARE STRUCTURED DECODER"
    )

    print("=" * 72)

    print(
        "Device        :",
        DEVICE,
    )

    print(
        "Manifest      :",
        MANIFEST,
    )

    print(
        "Payload bits  :",
        PAYLOAD_BITS,
    )

    print(
        "Grid          :",
        f"{GRID_ROWS} x {GRID_COLS}",
    )

    print(
        "Block size    :",
        f"{BLOCK_HEIGHT} x {BLOCK_WIDTH}",
    )

    print(
        "Strength      :",
        STRENGTH,
    )

    print(
        "Image size    :",
        IMAGE_SIZE,
    )

    print(
        "Batch size    :",
        BATCH_SIZE,
    )

    print(
        "Epochs        :",
        EPOCHS,
    )

    print(
        "Seed          :",
        SEED,
    )

    print(
        "Target        :",
        "best val BitAcc >= 90%",
    )

    print(
        "Decoder       :",
        "RGB + fixed high-pass residuals",
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = IMD2020Dataset(
        root=DATA_ROOT,
        manifest=MANIFEST,
        split="train",
        image_size=IMAGE_SIZE,
    )

    val_dataset = IMD2020Dataset(
        root=DATA_ROOT,
        manifest=MANIFEST,
        split="val",
        image_size=IMAGE_SIZE,
    )

    print()

    print(
        "Train samples :",
        len(train_dataset),
    )

    print(
        "Val samples   :",
        len(val_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Fidelity diagnostic
    # --------------------------------------------------------

    (
        diagnostic_mse,
        diagnostic_psnr,
    ) = estimate_psnr(
        val_loader
    )

    print()

    print(
        "Embedding MSE :",
        f"{diagnostic_mse:.8f}",
    )

    print(
        "Approx PSNR   :",
        f"{diagnostic_psnr:.4f} dB",
    )

    # Restore training random sequence after diagnostics.
    set_seed(SEED)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        StructuredStegoDecoderV33()
        .to(DEVICE)
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable params:",
        trainable_parameter_count,
    )

    print()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_bit_accuracy = -1.0
    best_val_exact_accuracy = 0.0
    best_epoch = 0

    log_rows = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        loss_sum = 0.0

        correct_bits = 0
        total_bits = 0

        exact_correct = 0
        total_messages = 0

        batches = 0

        for batch in train_loader:

            images = extract_images(batch)

            images = images.to(
                DEVICE,
                dtype=torch.float32,
            )

            payload = make_payload(
                images.shape[0],
                DEVICE,
            )

            stego = embed_payload(
                images,
                payload,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(stego)

            loss = criterion(
                logits,
                payload,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            (
                batch_correct_bits,
                batch_total_bits,
                batch_exact_correct,
                batch_total_messages,
            ) = bit_statistics(
                logits,
                payload,
            )

            loss_sum += loss.item()

            correct_bits += (
                batch_correct_bits
            )

            total_bits += (
                batch_total_bits
            )

            exact_correct += (
                batch_exact_correct
            )

            total_messages += (
                batch_total_messages
            )

            batches += 1

        train_loss = (
            loss_sum / batches
        )

        train_bit_accuracy = (
            correct_bits / total_bits
        )

        train_exact_accuracy = (
            exact_correct / total_messages
        )

        (
            val_loss,
            val_bit_accuracy,
            val_exact_accuracy,
        ) = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss={train_loss:.6f} | "
            f"bit={train_bit_accuracy * 100:.4f}% | "
            f"exact={train_exact_accuracy * 100:.4f}% | "
            f"val loss={val_loss:.6f} | "
            f"bit={val_bit_accuracy * 100:.4f}% | "
            f"exact={val_exact_accuracy * 100:.4f}%"
        )

        log_rows.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    f"{train_loss:.8f}",

                "train_bit_accuracy":
                    f"{train_bit_accuracy * 100:.4f}",

                "train_exact_accuracy":
                    f"{train_exact_accuracy * 100:.4f}",

                "val_loss":
                    f"{val_loss:.8f}",

                "val_bit_accuracy":
                    f"{val_bit_accuracy * 100:.4f}",

                "val_exact_accuracy":
                    f"{val_exact_accuracy * 100:.4f}",
            }
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_bit_accuracy > best_val_bit_accuracy:

            best_val_bit_accuracy = (
                val_bit_accuracy
            )

            best_val_exact_accuracy = (
                val_exact_accuracy
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "payload_bits":
                        PAYLOAD_BITS,

                    "grid_rows":
                        GRID_ROWS,

                    "grid_cols":
                        GRID_COLS,

                    "strength":
                        STRENGTH,

                    "seed":
                        SEED,

                    "epochs":
                        EPOCHS,

                    "val_bit_accuracy":
                        val_bit_accuracy,

                    "val_exact_accuracy":
                        val_exact_accuracy,

                    "version":
                        "V3.3",

                    "experiment":
                        "E4.12-A residual-aware decoder",

                    "decoder":
                        "RGB + fixed high-pass residuals",
                },
                BEST_CHECKPOINT,
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "payload_bits":
                    PAYLOAD_BITS,

                "grid_rows":
                    GRID_ROWS,

                "grid_cols":
                    GRID_COLS,

                "strength":
                    STRENGTH,

                "seed":
                    SEED,

                "epochs":
                    EPOCHS,

                "val_bit_accuracy":
                    val_bit_accuracy,

                "val_exact_accuracy":
                    val_exact_accuracy,

                "version":
                    "V3.3",

                "experiment":
                    "E4.12-A residual-aware decoder",

                "decoder":
                    "RGB + fixed high-pass residuals",
            },
            LAST_CHECKPOINT,
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        TRAINING_LOG,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_bit_accuracy",
                "train_exact_accuracy",
                "val_loss",
                "val_bit_accuracy",
                "val_exact_accuracy",
            ],
        )

        writer.writeheader()

        writer.writerows(log_rows)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "V3.3 RESIDUAL-AWARE STUDY COMPLETE"
    )

    print("-" * 72)

    print(
        "Best epoch            :",
        best_epoch,
    )

    print(
        "Best val BitAcc       : "
        f"{best_val_bit_accuracy * 100:.4f}%"
    )

    print(
        "ExactAcc at best epoch: "
        f"{best_val_exact_accuracy * 100:.4f}%"
    )

    print(
        "Pre-registered target : "
        ">= 90.0000%"
    )

    if best_val_bit_accuracy >= 0.90:

        print(
            "E4.12-A V3.3 result   : PASS"
        )

    else:

        print(
            "E4.12-A V3.3 result   : FAIL"
        )

    print()

    print(
        "Best checkpoint       :",
        BEST_CHECKPOINT,
    )

    print(
        "Last checkpoint       :",
        LAST_CHECKPOINT,
    )

    print(
        "Training CSV          :",
        TRAINING_LOG,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
