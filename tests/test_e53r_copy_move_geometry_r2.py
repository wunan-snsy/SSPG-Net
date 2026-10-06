from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from preprocessing.active_sspg_dataset import (
    sample_copy_move_geometry,
    rectangles_overlap,
)


N_TRIALS = 5000
BASE_SEED = 20260922


def main():

    print("=" * 78)
    print(
        "E5.3R-4A-R2 COPY-MOVE "
        "GEOMETRY STRESS GATE"
    )
    print("=" * 78)

    print(
        "Trials       :",
        N_TRIALS,
    )

    print(
        "Base seed    :",
        BASE_SEED,
    )

    print(
        "Area target  : 15% - 30%"
    )

    print(
        "Overlap rule : source and destination "
        "must be disjoint"
    )

    geometry_failures = 0
    overlap_failures = 0
    area_failures = 0
    determinism_failures = 0

    areas = []
    valid_source_counts = []

    for trial in range(
        N_TRIALS
    ):

        seed = (
            BASE_SEED
            + trial
        )

        rng1 = (
            np.random.default_rng(
                seed
            )
        )

        rng2 = (
            np.random.default_rng(
                seed
            )
        )

        try:

            g1 = (
                sample_copy_move_geometry(
                    rng1
                )
            )

        except RuntimeError:

            geometry_failures += 1
            continue

        try:

            g2 = (
                sample_copy_move_geometry(
                    rng2
                )
            )

        except RuntimeError:

            determinism_failures += 1
            continue

        # ----------------------------------------------------
        # Determinism
        # ----------------------------------------------------

        if g1 != g2:

            determinism_failures += 1

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        overlap = rectangles_overlap(
            g1["source_top"],
            g1["source_left"],
            g1["box_h"],
            g1["box_w"],
            g1["destination_top"],
            g1["destination_left"],
            g1["box_h"],
            g1["box_w"],
        )

        if overlap:

            overlap_failures += 1

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        area = (
            g1["box_h"]
            * g1["box_w"]
            / (256 * 256)
        )

        areas.append(
            area
        )

        valid_source_counts.append(
            g1[
                "valid_source_count"
            ]
        )

        if not (
            0.145
            <= area
            <= 0.305
        ):

            area_failures += 1

    areas = np.asarray(
        areas,
        dtype=np.float64,
    )

    valid_source_counts = np.asarray(
        valid_source_counts,
        dtype=np.int64,
    )

    print()
    print("-" * 78)
    print(
        "FAILURE COUNTS"
    )
    print("-" * 78)

    print(
        "Geometry failures    :",
        geometry_failures,
    )

    print(
        "Overlap failures     :",
        overlap_failures,
    )

    print(
        "Area failures        :",
        area_failures,
    )

    print(
        "Determinism failures :",
        determinism_failures,
    )

    if len(
        areas
    ) > 0:

        print()
        print("-" * 78)
        print(
            "AREA DISTRIBUTION"
        )
        print("-" * 78)

        print(
            "Minimum :",
            f"{areas.min():.6f}",
        )

        print(
            "Median  :",
            f"{np.median(areas):.6f}",
        )

        print(
            "Mean    :",
            f"{areas.mean():.6f}",
        )

        print(
            "Maximum :",
            f"{areas.max():.6f}",
        )

        print()
        print("-" * 78)
        print(
            "VALID SOURCE POSITION COUNTS"
        )
        print("-" * 78)

        print(
            "Minimum :",
            int(
                valid_source_counts.min()
            ),
        )

        print(
            "Median  :",
            f"{np.median(valid_source_counts):.1f}",
        )

        print(
            "Mean    :",
            f"{valid_source_counts.mean():.1f}",
        )

        print(
            "Maximum :",
            int(
                valid_source_counts.max()
            ),
        )

    passed = (
        geometry_failures == 0
        and overlap_failures == 0
        and area_failures == 0
        and determinism_failures == 0
        and len(areas) == N_TRIALS
    )

    print()
    print("=" * 78)

    if passed:

        print(
            "E5.3R-4A-R2 COPY-MOVE "
            "GEOMETRY GATE: PASS"
        )

    else:

        print(
            "E5.3R-4A-R2 COPY-MOVE "
            "GEOMETRY GATE: FAIL"
        )

        raise RuntimeError(
            "R2 geometry stress gate failed."
        )

    print("=" * 78)

    print()
    print(
        "No dataset image was accessed."
    )

    print(
        "No model was trained."
    )

    print(
        "No Test data was accessed."
    )


if __name__ == "__main__":
    main()