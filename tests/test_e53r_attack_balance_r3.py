from pathlib import Path
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT),
)


from preprocessing.active_sspg_dataset import (
    ATTACKS,
    build_balanced_attack_assignment,
)


# ============================================================
# E5.3R-4A-R3
# Deterministic Balanced Attack Assignment Gate
#
# NO images are loaded.
# NO model is trained.
# NO Test data is accessed.
# ============================================================


TRAIN_N = 289
VAL_N = 62

TRAIN_SEED = 20260922
VAL_SEED = 20260923


def audit_assignment(
    name,
    n_sources,
    seed,
):
    """
    Audit one split-level R3 assignment.
    """

    assignment_1 = (
        build_balanced_attack_assignment(
            n_sources=n_sources,
            base_seed=seed,
        )
    )

    assignment_2 = (
        build_balanced_attack_assignment(
            n_sources=n_sources,
            base_seed=seed,
        )
    )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if len(
        assignment_1
    ) != n_sources:

        raise RuntimeError(
            f"{name}: assignment length mismatch."
        )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    if any(
        attack is None
        for attack in assignment_1
    ):

        raise RuntimeError(
            f"{name}: incomplete assignment."
        )

    # --------------------------------------------------------
    # Valid attack names
    # --------------------------------------------------------

    invalid = [
        attack
        for attack in assignment_1
        if attack not in ATTACKS
    ]

    if invalid:

        raise RuntimeError(
            f"{name}: invalid attack names."
        )

    # --------------------------------------------------------
    # Determinism
    # --------------------------------------------------------

    if assignment_1 != assignment_2:

        raise RuntimeError(
            f"{name}: assignment is not deterministic."
        )

    # --------------------------------------------------------
    # Count distribution
    # --------------------------------------------------------

    counts = Counter(
        assignment_1
    )

    values = [
        counts[
            attack
        ]
        for attack in ATTACKS
    ]

    max_min_difference = (
        max(values)
        - min(values)
    )

    print()
    print("-" * 78)
    print(
        f"{name} ATTACK ASSIGNMENT"
    )
    print("-" * 78)

    print(
        "Sources :",
        n_sources,
    )

    print(
        "Seed    :",
        seed,
    )

    print()

    for attack in ATTACKS:

        count = counts[
            attack
        ]

        fraction = (
            count
            / n_sources
        )

        print(
            f"{attack:10s}: "
            f"{count:4d} "
            f"({fraction * 100:6.2f}%)"
        )

    print()

    print(
        "Max-min difference :",
        max_min_difference,
    )

    if max_min_difference > 1:

        raise RuntimeError(
            f"{name}: attack counts are not balanced."
        )

    # --------------------------------------------------------
    # Different seed should normally change source-to-attack
    # mapping, while aggregate balance remains unchanged.
    # --------------------------------------------------------

    assignment_other_seed = (
        build_balanced_attack_assignment(
            n_sources=n_sources,
            base_seed=seed + 1,
        )
    )

    changed_sources = sum(
        attack_a != attack_b
        for attack_a, attack_b in zip(
            assignment_1,
            assignment_other_seed,
        )
    )

    print(
        "Assignment deterministic : PASS"
    )

    print(
        "Balanced counts          : PASS"
    )

    print(
        "Sources changed with "
        "different seed          :",
        changed_sources,
    )

    if changed_sources == 0:

        raise RuntimeError(
            f"{name}: changing the split seed "
            "did not alter any source assignment."
        )

    return {
        "assignment":
            assignment_1,

        "counts":
            counts,

        "changed_sources":
            changed_sources,
    }


def main():

    print("=" * 78)
    print(
        "E5.3R-4A-R3 DETERMINISTIC "
        "ATTACK BALANCE GATE"
    )
    print("=" * 78)

    print(
        "Attacks    :",
        ATTACKS,
    )

    print(
        "Train N    :",
        TRAIN_N,
    )

    print(
        "Val N      :",
        VAL_N,
    )

    print(
        "Train seed :",
        TRAIN_SEED,
    )

    print(
        "Val seed   :",
        VAL_SEED,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_result = (
        audit_assignment(
            name="TRAIN",
            n_sources=TRAIN_N,
            seed=TRAIN_SEED,
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_result = (
        audit_assignment(
            name="VAL",
            n_sources=VAL_N,
            seed=VAL_SEED,
        )
    )

    # --------------------------------------------------------
    # Exact aggregate expectations.
    #
    # Because the cycle starts:
    #
    #   erase
    #   copy_move
    #   roll
    #
    # N=289:
    #   97 / 96 / 96
    #
    # N=62:
    #   21 / 21 / 20
    # --------------------------------------------------------

    expected_train = {
        "erase":
            97,

        "copy_move":
            96,

        "roll":
            96,
    }

    expected_val = {
        "erase":
            21,

        "copy_move":
            21,

        "roll":
            20,
    }

    actual_train = {
        attack:
            train_result[
                "counts"
            ][
                attack
            ]
        for attack in ATTACKS
    }

    actual_val = {
        attack:
            val_result[
                "counts"
            ][
                attack
            ]
        for attack in ATTACKS
    }

    print()
    print("-" * 78)
    print(
        "EXACT COUNT GATE"
    )
    print("-" * 78)

    print(
        "Expected Train :",
        expected_train,
    )

    print(
        "Actual Train   :",
        actual_train,
    )

    print()

    print(
        "Expected Val   :",
        expected_val,
    )

    print(
        "Actual Val     :",
        actual_val,
    )

    if (
        actual_train
        != expected_train
    ):

        raise RuntimeError(
            "Unexpected Train aggregate "
            "attack counts."
        )

    if (
        actual_val
        != expected_val
    ):

        raise RuntimeError(
            "Unexpected Validation aggregate "
            "attack counts."
        )

    print()
    print(
        "Exact aggregate counts : PASS"
    )

    # --------------------------------------------------------
    # Final gate
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "E5.3R-4A-R3 ATTACK "
        "BALANCE GATE: PASS"
    )

    print("=" * 78)

    print()
    print(
        "Frozen Train attack counts:"
    )

    print(
        "  erase      = 97"
    )

    print(
        "  copy_move  = 96"
    )

    print(
        "  roll       = 96"
    )

    print()
    print(
        "Frozen Validation attack counts:"
    )

    print(
        "  erase      = 21"
    )

    print(
        "  copy_move  = 21"
    )

    print(
        "  roll       = 20"
    )

    print()
    print(
        "No image was loaded."
    )

    print(
        "No model was trained."
    )

    print(
        "No Test data was accessed."
    )


if __name__ == "__main__":
    main()