from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.active_sspg_dataset import (
    ATTACKS,
    stable_seed,
    make_payload_deterministic,
    apply_controlled_attack,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 78)
    print(
        "E5.3R-4A ACTIVE SSPG DATA "
        "PROTOCOL GATE"
    )
    print("=" * 78)

    print("Device      :", DEVICE)
    print("Image size  : 256 x 256")
    print("Payload     : 128 bits")
    print("Attacks     :", ATTACKS)
    print("Area target : 15% - 30%")
    print()

    # --------------------------------------------------------
    # Stable seed
    # --------------------------------------------------------

    seed_a = stable_seed(
        20260922,
        "example/source/image.png",
    )

    seed_b = stable_seed(
        20260922,
        "example/source/image.png",
    )

    seed_c = stable_seed(
        20260923,
        "example/source/image.png",
    )

    assert seed_a == seed_b
    assert seed_a != seed_c

    print(
        "Stable per-sample seed : PASS"
    )

    # --------------------------------------------------------
    # Payload determinism
    # --------------------------------------------------------

    payload_a = (
        make_payload_deterministic(
            seed_a
        )
    )

    payload_b = (
        make_payload_deterministic(
            seed_a
        )
    )

    assert torch.equal(
        payload_a,
        payload_b,
    )

    assert payload_a.shape == (
        128,
    )

    assert set(
        payload_a.unique().tolist()
    ).issubset(
        {0.0, 1.0}
    )

    print(
        "Payload determinism     : PASS"
    )

    # --------------------------------------------------------
    # Synthetic image for protocol audit only.
    # --------------------------------------------------------

    torch.manual_seed(
        20260922
    )

    protected = torch.rand(
        3,
        256,
        256,
        device=DEVICE,
    )

    print()
    print("-" * 78)
    print("ATTACK AUDIT")
    print("-" * 78)

    for attack_index, attack in enumerate(
        ATTACKS
    ):

        rng1 = np.random.default_rng(
            1000 + attack_index
        )

        rng2 = np.random.default_rng(
            1000 + attack_index
        )

        manipulated1, mask1 = (
            apply_controlled_attack(
                protected,
                attack,
                rng1,
            )
        )

        manipulated2, mask2 = (
            apply_controlled_attack(
                protected,
                attack,
                rng2,
            )
        )

        # Determinism
        assert torch.equal(
            manipulated1,
            manipulated2,
        )

        assert torch.equal(
            mask1,
            mask2,
        )

        # Shape
        assert manipulated1.shape == (
            3,
            256,
            256,
        )

        assert mask1.shape == (
            1,
            256,
            256,
        )

        # Binary mask
        unique_mask = set(
            mask1.unique().tolist()
        )

        assert unique_mask.issubset(
            {0.0, 1.0}
        )

        area_fraction = (
            mask1.mean().item()
        )

        # Integer rounding allows a tiny tolerance.
        assert (
            0.145
            <= area_fraction
            <= 0.305
        )

        changed = (
            manipulated1
            - protected
        ).abs().amax(
            dim=0,
            keepdim=True,
        ) > 1e-8

        changed = (
            changed.float()
        )

        outside_change = (
            changed
            * (1.0 - mask1)
        ).sum().item()

        assert outside_change == 0.0

        changed_inside = (
            changed
            * mask1
        ).sum().item()

        assert changed_inside > 0.0

        print(
            f"{attack:10s} "
            f"mask area={area_fraction:.4f} "
            f"outside changes={outside_change:.0f}"
        )

    print()
    print(
        "Controlled attack audit : PASS"
    )

    # --------------------------------------------------------
    # Authentic definition
    # --------------------------------------------------------

    authentic_mask = torch.zeros(
        1,
        256,
        256,
    )

    authentic_label = 0
    manipulated_label = 1

    assert authentic_mask.sum() == 0
    assert authentic_label == 0
    assert manipulated_label == 1

    print(
        "Label/mask convention   : PASS"
    )

    print()
    print("=" * 78)

    print(
        "E5.3R-4A DATA PROTOCOL GATE: PASS"
    )

    print("=" * 78)

    print()
    print(
        "Protocol is now fixed BEFORE "
        "detection/localization training."
    )

    print(
        "No Train/Val/Test dataset images "
        "were accessed by this gate."
    )


if __name__ == "__main__":
    main()