from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
    apply_controlled_attack,
)


# ============================================================
# E5.3R-4C-4
# Leave-One-Attack-Out Active SSPG Dataset
#
# Revision:
#   V1-R1
#
# Reason for R1:
#   Attempt 1 called apply_controlled_attack() with a
#   non-existent keyword argument "seed".
#
#   The frozen ActiveSSPGDataset implementation expects:
#
#       apply_controlled_attack(
#           protected,
#           attack,
#           rng,
#       )
#
#   where rng is numpy.random.Generator.
#
# SCIENTIFIC PROTOCOL IS UNCHANGED.
#
# IMPORTANT:
#   - active_sspg_dataset.py is NOT modified.
#   - frozen payload generation is reused.
#   - frozen V3.3 embedding is reused.
#   - frozen controlled attacks are reused.
#   - Train uses exactly TWO seen attack families.
#   - Validation uses exactly ONE held-out attack family.
#   - Every source produces exactly:
#
#         1 authentic protected sample
#         1 manipulated protected sample
#
#   - Test is not involved in this class.
# ============================================================


ALL_ATTACKS = (
    "erase",
    "copy_move",
    "roll",
)

IMAGE_SIZE = 256


# ------------------------------------------------------------
# These offsets are part of the already frozen Active SSPG
# protocol.
#
# ATTACK_ASSIGNMENT_OFFSET:
#     deterministic split-level attack assignment
#
# GEOMETRY_SEED_OFFSET:
#     deterministic attack geometry
#
# We preserve their frozen numerical values here because the
# original module does not necessarily expose them under a
# stable public import name.
# ------------------------------------------------------------

ATTACK_ASSIGNMENT_OFFSET = 300007
GEOMETRY_SEED_OFFSET = 100003


# ============================================================
# ATTACK HELPERS
# ============================================================


def normalize_attack_name(
    attack: str,
) -> str:
    attack = str(
        attack
    ).strip().lower()

    if attack not in ALL_ATTACKS:
        raise ValueError(
            f"Unknown attack: {attack}"
        )

    return attack


def seen_attacks_for(
    held_out_attack: str,
):
    """
    Return the two attacks permitted during training when one
    attack family is held out.
    """

    held_out_attack = normalize_attack_name(
        held_out_attack
    )

    seen = tuple(
        attack
        for attack in ALL_ATTACKS
        if attack != held_out_attack
    )

    if len(seen) != 2:
        raise RuntimeError(
            "LOAO must contain exactly two seen attacks."
        )

    return seen


# ============================================================
# BALANCED TWO-ATTACK TRAIN ASSIGNMENT
# ============================================================


def build_two_attack_balanced_assignment(
    n_sources: int,
    attacks: Sequence[str],
    base_seed: int,
) -> List[str]:
    """
    Deterministically assign exactly two attack families to
    Train sources.

    With an odd number of sources, the two counts differ by
    exactly one.

    For the current Train split:

        289 manipulated samples

    therefore the expected distribution is:

        145 / 144

    The source ordering itself remains the already frozen
    canonical order supplied by the caller.
    """

    n_sources = int(
        n_sources
    )

    if n_sources <= 0:
        raise ValueError(
            "n_sources must be positive."
        )

    attacks = tuple(
        normalize_attack_name(
            attack
        )
        for attack in attacks
    )

    if len(attacks) != 2:
        raise ValueError(
            "Exactly two attacks are required."
        )

    if attacks[0] == attacks[1]:
        raise ValueError(
            "The two attacks must be different."
        )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        int(base_seed)
        + ATTACK_ASSIGNMENT_OFFSET
    )

    permutation = torch.randperm(
        n_sources,
        generator=generator,
    ).tolist()

    assignment = [
        None
        for _ in range(
            n_sources
        )
    ]

    for rank, source_index in enumerate(
        permutation
    ):
        attack = attacks[
            rank % 2
        ]

        assignment[
            source_index
        ] = attack

    if any(
        value is None
        for value in assignment
    ):
        raise RuntimeError(
            "Incomplete LOAO attack assignment."
        )

    counts = {
        attack:
            assignment.count(
                attack
            )
        for attack in attacks
    }

    if abs(
        counts[
            attacks[0]
        ]
        - counts[
            attacks[1]
        ]
    ) > 1:
        raise RuntimeError(
            "LOAO Train attacks are not balanced."
        )

    return assignment


# ============================================================
# DATASET
# ============================================================


class LOAOActiveSSPGDataset(Dataset):
    """
    Leave-One-Attack-Out wrapper around the frozen,
    previously validated ActiveSSPGDataset.

    ----------------------------------------------------------
    TRAIN
    ----------------------------------------------------------

    For each source:

        authentic:
            generated exactly by the frozen base dataset

        manipulated:
            starts from that SAME protected authentic image
            and SAME payload, then applies one of the two
            allowed seen attacks.

    The held-out attack occurs zero times in Train.

    ----------------------------------------------------------
    VALIDATION
    ----------------------------------------------------------

    For each source:

        authentic:
            generated exactly by the frozen base dataset

        manipulated:
            starts from the same protected image and applies
            ONLY the held-out attack.

    Therefore validation manipulated samples are exclusively
    from an unseen attack family.

    ----------------------------------------------------------
    IMPORTANT
    ----------------------------------------------------------

    The base dataset's original manipulated sample is never
    used. Only its authentic protected sample is reused.
    """

    def __init__(
        self,
        records: Sequence[Dict],
        base_seed: int,
        held_out_attack: str,
        mode: str,
    ):
        super().__init__()

        self.records = list(
            records
        )

        if len(
            self.records
        ) == 0:
            raise ValueError(
                "records cannot be empty."
            )

        self.base_seed = int(
            base_seed
        )

        self.held_out_attack = (
            normalize_attack_name(
                held_out_attack
            )
        )

        self.mode = str(
            mode
        ).strip().lower()

        if self.mode not in (
            "train",
            "val",
        ):
            raise ValueError(
                "mode must be either 'train' or 'val'."
            )

        # ----------------------------------------------------
        # Frozen base dataset.
        #
        # This preserves:
        #   source loading
        #   canonical sample key
        #   deterministic payload
        #   V3.3 embedding
        #   authentic protected sample
        # ----------------------------------------------------

        self.base_dataset = (
            ActiveSSPGDataset(
                records=self.records,
                base_seed=self.base_seed,
            )
        )

        if len(
            self.base_dataset
        ) != (
            2
            * len(
                self.records
            )
        ):
            raise RuntimeError(
                "Unexpected base dataset length."
            )

        self.seen_attacks = (
            seen_attacks_for(
                self.held_out_attack
            )
        )

        # ----------------------------------------------------
        # Manipulated attack assignment.
        # ----------------------------------------------------

        if self.mode == "train":
            self.attack_assignment = (
                build_two_attack_balanced_assignment(
                    n_sources=len(
                        self.records
                    ),
                    attacks=self.seen_attacks,
                    base_seed=self.base_seed,
                )
            )

        else:
            self.attack_assignment = [
                self.held_out_attack
                for _ in range(
                    len(
                        self.records
                    )
                )
            ]

        if len(
            self.attack_assignment
        ) != len(
            self.records
        ):
            raise RuntimeError(
                "Attack assignment length mismatch."
            )

        # ----------------------------------------------------
        # Defensive protocol checks.
        # ----------------------------------------------------

        if self.mode == "train":
            if (
                self.held_out_attack
                in self.attack_assignment
            ):
                raise RuntimeError(
                    "Held-out attack leaked into Train "
                    "assignment."
                )

            observed = set(
                self.attack_assignment
            )

            expected = set(
                self.seen_attacks
            )

            if observed != expected:
                raise RuntimeError(
                    "Train attack assignment does not "
                    "contain exactly the two seen attacks."
                )

        else:
            observed = set(
                self.attack_assignment
            )

            if observed != {
                self.held_out_attack
            }:
                raise RuntimeError(
                    "Validation assignment is not exclusively "
                    "the held-out attack."
                )

    # ========================================================
    # BASIC DATASET INTERFACE
    # ========================================================

    def __len__(
        self,
    ):
        return (
            2
            * len(
                self.records
            )
        )

    def manipulated_attack_for_source(
        self,
        source_index: int,
    ) -> str:
        source_index = int(
            source_index
        )

        if not (
            0
            <= source_index
            < len(
                self.records
            )
        ):
            raise IndexError(
                source_index
            )

        return self.attack_assignment[
            source_index
        ]

    # ========================================================
    # DETERMINISTIC ATTACK RNG
    # ========================================================

    @staticmethod
    def make_geometry_rng(
        source_seed: int,
    ):
        """
        Reproduce the frozen Active SSPG attack geometry RNG:

            np.random.default_rng(
                source_seed + 100003
            )

        A fresh generator is created for every manipulated
        sample so repeated access to the same dataset index
        produces identical geometry.
        """

        geometry_seed = (
            int(
                source_seed
            )
            + GEOMETRY_SEED_OFFSET
        )

        return np.random.default_rng(
            geometry_seed
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ):
        index = int(
            index
        )

        if not (
            0
            <= index
            < len(self)
        ):
            raise IndexError(
                index
            )

        source_index = (
            index // 2
        )

        manipulated_flag = (
            index % 2
        )

        # ----------------------------------------------------
        # Always recover the frozen authentic/protected member
        # of this source pair.
        #
        # Base ActiveSSPGDataset convention:
        #
        #     2 * source_index       -> authentic
        #     2 * source_index + 1   -> manipulated
        #
        # We intentionally never request the second member.
        # ----------------------------------------------------

        authentic = (
            self.base_dataset[
                2
                * source_index
            ]
        )

        if int(
            authentic[
                "detection_label"
            ].item()
        ) != 0:
            raise RuntimeError(
                "Base authentic sample has non-zero label."
            )

        if str(
            authentic[
                "attack"
            ]
        ) != "authentic":
            raise RuntimeError(
                "Base even-index sample is not authentic."
            )

        if float(
            authentic[
                "mask"
            ].sum().item()
        ) != 0.0:
            raise RuntimeError(
                "Base authentic mask is not zero."
            )

        # ----------------------------------------------------
        # Authentic branch.
        #
        # Return a new dictionary so callers cannot mutate the
        # base dictionary object.
        # ----------------------------------------------------

        if manipulated_flag == 0:
            return {
                "received":
                    authentic[
                        "received"
                    ],

                "payload":
                    authentic[
                        "payload"
                    ],

                "mask":
                    authentic[
                        "mask"
                    ],

                "detection_label":
                    authentic[
                        "detection_label"
                    ],

                "attack":
                    "authentic",

                "source_path":
                    authentic[
                        "source_path"
                    ],

                "sample_key":
                    authentic[
                        "sample_key"
                    ],

                "source_seed":
                    int(
                        authentic[
                            "source_seed"
                        ]
                    ),

                "source_index":
                    source_index,
            }

        # ----------------------------------------------------
        # Manipulated branch.
        #
        # Start from the exact protected authentic image.
        # ----------------------------------------------------

        protected = (
            authentic[
                "received"
            ].clone()
        )

        payload = (
            authentic[
                "payload"
            ].clone()
        )

        source_seed = int(
            authentic[
                "source_seed"
            ]
        )

        attack = (
            self.manipulated_attack_for_source(
                source_index
            )
        )

        # ----------------------------------------------------
        # Strong LOAO isolation checks.
        # ----------------------------------------------------

        if self.mode == "train":
            if attack == self.held_out_attack:
                raise RuntimeError(
                    "Held-out attack encountered in Train."
                )

            if attack not in self.seen_attacks:
                raise RuntimeError(
                    "Unexpected Train attack."
                )

        elif self.mode == "val":
            if attack != self.held_out_attack:
                raise RuntimeError(
                    "Validation manipulated sample is not "
                    "the held-out attack."
                )

        # ----------------------------------------------------
        # Correct frozen attack API:
        #
        #     apply_controlled_attack(
        #         protected,
        #         attack,
        #         rng,
        #     )
        #
        # NOT:
        #
        #     seed=...
        # ----------------------------------------------------

        rng = self.make_geometry_rng(
            source_seed
        )

        received, mask = (
            apply_controlled_attack(
                protected,
                attack,
                rng,
            )
        )

        # ----------------------------------------------------
        # Output validation.
        # ----------------------------------------------------

        if not torch.is_tensor(
            received
        ):
            raise RuntimeError(
                "Controlled attack did not return "
                "a tensor image."
            )

        if not torch.is_tensor(
            mask
        ):
            raise RuntimeError(
                "Controlled attack did not return "
                "a tensor mask."
            )

        if tuple(
            received.shape
        ) != (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise RuntimeError(
                "Unexpected manipulated image shape: "
                f"{tuple(received.shape)}"
            )

        if tuple(
            mask.shape
        ) != (
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise RuntimeError(
                "Unexpected manipulated mask shape: "
                f"{tuple(mask.shape)}"
            )

        if not torch.isfinite(
            received
        ).all():
            raise RuntimeError(
                "Manipulated image contains non-finite values."
            )

        if not torch.isfinite(
            mask
        ).all():
            raise RuntimeError(
                "Manipulated mask contains non-finite values."
            )

        if float(
            mask.sum().item()
        ) <= 0.0:
            raise RuntimeError(
                "Manipulated sample has an empty mask."
            )

        if float(
            received.min().item()
        ) < 0.0:
            raise RuntimeError(
                "Manipulated image contains values below 0."
            )

        if float(
            received.max().item()
        ) > 1.0:
            raise RuntimeError(
                "Manipulated image contains values above 1."
            )

        return {
            "received":
                received,

            "payload":
                payload,

            "mask":
                mask,

            "detection_label":
                torch.tensor(
                    1,
                    dtype=torch.long,
                ),

            "attack":
                attack,

            "source_path":
                authentic[
                    "source_path"
                ],

            "sample_key":
                authentic[
                    "sample_key"
                ],

            "source_seed":
                source_seed,

            "source_index":
                source_index,
        }