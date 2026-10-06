from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_stego_structured_v33 import (
    StructuredStegoDecoderV33,
    embed_payload,
    make_payload,
    PAYLOAD_BITS,
    GRID_ROWS,
    GRID_COLS,
    BLOCK_HEIGHT,
    BLOCK_WIDTH,
    IMAGE_SIZE,
    STRENGTH,
)


CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def spatial_logits(model, x):
    """
    Reproduce V3.3 forward path, but expose the spatial
    logits_map BEFORE AdaptiveAvgPool2d.

    No model parameter is changed.
    """
    residual = model.highpass(x)

    combined = torch.cat(
        [x, residual],
        dim=1,
    )

    logits_map = model.features(combined)

    logits_grid = model.pool(logits_map)

    logits = logits_grid.flatten(start_dim=1)

    return logits_map, logits_grid, logits


def main():

    print("=" * 72)
    print("E5.3R GATE-0: V3.3 CHANNEL INTERFACE PRECHECK")
    print("=" * 72)

    print("Device      :", DEVICE)
    print("Checkpoint  :", CHECKPOINT)
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    print("CHECKPOINT METADATA")
    print("-" * 72)

    for key in (
        "epoch",
        "payload_bits",
        "grid_rows",
        "grid_cols",
        "strength",
        "seed",
        "epochs",
        "val_bit_accuracy",
        "val_exact_accuracy",
        "version",
        "experiment",
        "decoder",
    ):
        print(
            f"{key:20s}:",
            checkpoint.get(key, "<missing>")
        )

    # --------------------------------------------------------
    # Frozen protocol integrity checks
    # --------------------------------------------------------

    assert checkpoint["payload_bits"] == PAYLOAD_BITS == 128
    assert checkpoint["grid_rows"] == GRID_ROWS == 8
    assert checkpoint["grid_cols"] == GRID_COLS == 16

    assert abs(
        float(checkpoint["strength"]) - STRENGTH
    ) < 1e-12

    assert checkpoint["version"] == "V3.3"

    assert IMAGE_SIZE == 256
    assert BLOCK_HEIGHT == 32
    assert BLOCK_WIDTH == 16

    print()
    print("Protocol metadata integrity : PASS")

    # --------------------------------------------------------
    # Load frozen V3.3 decoder
    # --------------------------------------------------------

    model = StructuredStegoDecoderV33().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    assert trainable == 0

    print("Decoder strict load         : PASS")
    print("Decoder frozen              : PASS")
    print("Trainable parameters        :", trainable)

    # --------------------------------------------------------
    # Synthetic tensor ONLY for interface verification.
    #
    # This is NOT an experimental sample and must never be
    # reported as a scientific result.
    # --------------------------------------------------------

    torch.manual_seed(20260921)

    image = torch.rand(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE,
    )

    payload = make_payload(
        batch_size=2,
        device=DEVICE,
    )

    stego = embed_payload(
        image,
        payload,
    )

    with torch.no_grad():

        normal_logits = model(stego)

        (
            logits_map,
            logits_grid,
            exposed_logits,
        ) = spatial_logits(
            model,
            stego,
        )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert payload.shape == (2, 128)

    assert normal_logits.shape == (2, 128)

    assert logits_grid.shape == (
        2,
        1,
        GRID_ROWS,
        GRID_COLS,
    )

    assert exposed_logits.shape == (
        2,
        128,
    )

    if logits_map.shape[0] != 2:
        raise RuntimeError(
            f"Unexpected spatial batch shape: "
            f"{tuple(logits_map.shape)}"
        )

    if logits_map.shape[1] != 1:
        raise RuntimeError(
            f"Expected one spatial logit channel, got "
            f"{tuple(logits_map.shape)}"
        )

    # --------------------------------------------------------
    # Exact forward-equivalence check
    # --------------------------------------------------------

    max_abs_diff = (
        normal_logits
        - exposed_logits
    ).abs().max().item()

    print()
    print("SPATIAL INTERFACE")
    print("-" * 72)

    print(
        "Input shape                :",
        tuple(image.shape),
    )

    print(
        "Payload shape              :",
        tuple(payload.shape),
    )

    print(
        "Stego shape                :",
        tuple(stego.shape),
    )

    print(
        "Spatial logits_map shape   :",
        tuple(logits_map.shape),
    )

    print(
        "Pooled logits_grid shape   :",
        tuple(logits_grid.shape),
    )

    print(
        "Final logits shape         :",
        tuple(exposed_logits.shape),
    )

    print(
        "Forward max abs difference :",
        f"{max_abs_diff:.12g}",
    )

    if max_abs_diff > 1e-7:
        raise RuntimeError(
            "Exposed spatial path does not reproduce "
            "the frozen V3.3 forward pass."
        )

    print("Forward equivalence         : PASS")

    # --------------------------------------------------------
    # Basic embedding sanity check
    # --------------------------------------------------------

    mse = F.mse_loss(
        stego,
        image,
    ).item()

    psnr = (
        10.0
        * torch.log10(
            torch.tensor(
                1.0 / mse,
                device=DEVICE,
            )
        )
    ).item()

    print()
    print("EMBEDDING SANITY")
    print("-" * 72)

    print(
        "Embedding MSE              :",
        f"{mse:.8f}",
    )

    print(
        "Embedding PSNR             :",
        f"{psnr:.4f} dB",
    )

    # --------------------------------------------------------
    # Decoder sanity
    # --------------------------------------------------------

    predicted = (
        torch.sigmoid(exposed_logits) >= 0.5
    ).float()

    bit_accuracy = (
        predicted == payload
    ).float().mean().item()

    print(
        "Random-tensor BitAcc       :",
        f"{bit_accuracy * 100:.4f}%",
    )

    print()
    print(
        "NOTE: Random-tensor BitAcc is diagnostic only."
    )
    print(
        "It is NOT an E5.3R scientific result."
    )

    print()
    print("=" * 72)
    print("GATE-0 RESULT: PASS")
    print("=" * 72)
    print(
        "V3.3 can now be used for the controlled "
        "E5.3R-0A spatial causality experiment."
    )


if __name__ == "__main__":
    main()