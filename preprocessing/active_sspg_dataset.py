from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from training.train_stego_structured_v33 import embed_payload


# ============================================================
# E5.3R-4A-R3.1
# Active SSPG Controlled-Forgery Dataset
#
# FINAL PRE-TRAINING DATA-PROTOCOL CANDIDATE
#
# R1:
#   copy-move source/destination must not overlap.
#
# R2:
#   deterministic exhaustive legal-source enumeration.
#
# R3:
#   deterministic split-level balanced attack assignment.
#
# R3.1:
#   canonical sample identity is centralized here so that
#   audit and training use EXACTLY the same seed definition.
# ============================================================


ATTACKS = (
    "erase",
    "copy_move",
    "roll",
)

IMAGE_SIZE = 256

MIN_AREA = 0.15
MAX_AREA = 0.30

COPY_MOVE_MAX_BOX_ATTEMPTS = 100

ATTACK_ASSIGNMENT_SEED_OFFSET = 300007
ATTACK_GEOMETRY_SEED_OFFSET = 100003


# ============================================================
# CANONICAL SOURCE IDENTITY
# ============================================================


def canonical_sample_key(record):
    """
    Build the single canonical source identity used everywhere.

    Preferred manifest identity:

        group|sample_id|image_path

    This function must be shared by:
        - audit
        - dataset
        - payload seed generation
        - geometry seed generation

    This prevents audit/training seed drift.
    """

    if "sample_key" in record:
        value = record["sample_key"]

        if value is not None:
            value = str(value)

            if value.strip():
                return value

    group = str(
        record.get(
            "group",
            "",
        )
    )

    sample_id = str(
        record.get(
            "sample_id",
            "",
        )
    )

    image_path = str(
        record.get(
            "image_path",
            "",
        )
    )

    if not image_path:
        raise RuntimeError(
            "Record has no image_path."
        )

    return (
        f"{group}|"
        f"{sample_id}|"
        f"{image_path}"
    )


# ============================================================
# DETERMINISTIC SOURCE SEED
# ============================================================


def stable_seed(
    base_seed,
    sample_key,
):
    text = (
        f"{int(base_seed)}|"
        f"{str(sample_key)}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        text
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return (
        value
        % (2**31 - 1)
    )


# ============================================================
# PAYLOAD
# ============================================================


def make_payload_deterministic(
    sample_seed,
):
    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        int(sample_seed)
    )

    payload = torch.randint(
        0,
        2,
        (128,),
        generator=generator,
        dtype=torch.float32,
    )

    return payload


# ============================================================
# IMAGE LOADING
# ============================================================


def load_rgb_256(
    path,
):
    path = Path(
        path
    )

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ).copy()

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1,
    )

    tensor = (
        tensor
        / 255.0
    )

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    return tensor.clamp(
        0.0,
        1.0,
    )


# ============================================================
# RECTANGLE SAMPLING
# ============================================================


def sample_box(
    rng,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
    min_area=MIN_AREA,
    max_area=MAX_AREA,
):
    target_fraction = float(
        rng.uniform(
            min_area,
            max_area,
        )
    )

    aspect = float(
        rng.uniform(
            0.5,
            2.0,
        )
    )

    target_area = (
        target_fraction
        * height
        * width
    )

    box_h = int(
        round(
            np.sqrt(
                target_area
                / aspect
            )
        )
    )

    box_w = int(
        round(
            box_h
            * aspect
        )
    )

    box_h = max(
        1,
        min(
            box_h,
            height,
        ),
    )

    box_w = max(
        1,
        min(
            box_w,
            width,
        ),
    )

    top = int(
        rng.integers(
            0,
            height - box_h + 1,
        )
    )

    left = int(
        rng.integers(
            0,
            width - box_w + 1,
        )
    )

    return (
        top,
        left,
        box_h,
        box_w,
    )


# ============================================================
# RECTANGLE OVERLAP
# ============================================================


def rectangles_overlap(
    top_a,
    left_a,
    height_a,
    width_a,
    top_b,
    left_b,
    height_b,
    width_b,
):
    bottom_a = (
        top_a
        + height_a
    )

    right_a = (
        left_a
        + width_a
    )

    bottom_b = (
        top_b
        + height_b
    )

    right_b = (
        left_b
        + width_b
    )

    return not (
        bottom_a <= top_b
        or bottom_b <= top_a
        or right_a <= left_b
        or right_b <= left_a
    )


# ============================================================
# COPY-MOVE LEGAL SOURCE ENUMERATION
# ============================================================


def enumerate_non_overlapping_sources(
    destination_top,
    destination_left,
    box_h,
    box_w,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
):
    valid_positions = []

    max_src_top = (
        height
        - box_h
    )

    max_src_left = (
        width
        - box_w
    )

    for src_top in range(
        max_src_top + 1
    ):

        for src_left in range(
            max_src_left + 1
        ):

            overlap = rectangles_overlap(
                src_top,
                src_left,
                box_h,
                box_w,
                destination_top,
                destination_left,
                box_h,
                box_w,
            )

            if not overlap:
                valid_positions.append(
                    (
                        src_top,
                        src_left,
                    )
                )

    return valid_positions


# ============================================================
# COPY-MOVE GEOMETRY — R2
# ============================================================


def sample_copy_move_geometry(
    rng,
    height=IMAGE_SIZE,
    width=IMAGE_SIZE,
    max_box_attempts=COPY_MOVE_MAX_BOX_ATTEMPTS,
):
    for _ in range(
        max_box_attempts
    ):

        (
            destination_top,
            destination_left,
            box_h,
            box_w,
        ) = sample_box(
            rng,
            height=height,
            width=width,
            min_area=MIN_AREA,
            max_area=MAX_AREA,
        )

        valid_sources = (
            enumerate_non_overlapping_sources(
                destination_top,
                destination_left,
                box_h,
                box_w,
                height,
                width,
            )
        )

        if not valid_sources:
            continue

        selected_index = int(
            rng.integers(
                0,
                len(valid_sources),
            )
        )

        (
            source_top,
            source_left,
        ) = valid_sources[
            selected_index
        ]

        return {
            "destination_top":
                destination_top,

            "destination_left":
                destination_left,

            "source_top":
                source_top,

            "source_left":
                source_left,

            "box_h":
                box_h,

            "box_w":
                box_w,

            "valid_source_count":
                len(valid_sources),
        }

    raise RuntimeError(
        "Could not generate a valid "
        "copy-move pair."
    )


# ============================================================
# R3 BALANCED ATTACK ASSIGNMENT
# ============================================================


def build_balanced_attack_assignment(
    n_sources,
    base_seed,
):
    n_sources = int(
        n_sources
    )

    if n_sources <= 0:
        raise ValueError(
            "n_sources must be positive."
        )

    generator = np.random.default_rng(
        int(base_seed)
        + ATTACK_ASSIGNMENT_SEED_OFFSET
    )

    permutation = generator.permutation(
        n_sources
    )

    assignment = [
        None
        for _ in range(
            n_sources
        )
    ]

    for (
        order_position,
        source_index,
    ) in enumerate(
        permutation
    ):

        attack = ATTACKS[
            order_position
            % len(ATTACKS)
        ]

        assignment[
            int(source_index)
        ] = attack

    if any(
        x is None
        for x in assignment
    ):
        raise RuntimeError(
            "Incomplete attack assignment."
        )

    return assignment


# ============================================================
# CONTROLLED MANIPULATION
# ============================================================


def apply_controlled_attack(
    protected,
    attack,
    rng,
):
    expected_shape = (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(
        protected.shape
    ) != expected_shape:
        raise RuntimeError(
            "Expected [3,256,256], "
            f"got {tuple(protected.shape)}."
        )

    if attack not in ATTACKS:
        raise ValueError(
            f"Unknown attack: {attack}"
        )

    # --------------------------------------------------------
    # COPY-MOVE
    # --------------------------------------------------------

    if attack == "copy_move":

        geometry = (
            sample_copy_move_geometry(
                rng
            )
        )

        top = geometry[
            "destination_top"
        ]

        left = geometry[
            "destination_left"
        ]

        src_top = geometry[
            "source_top"
        ]

        src_left = geometry[
            "source_left"
        ]

        box_h = geometry[
            "box_h"
        ]

        box_w = geometry[
            "box_w"
        ]

    # --------------------------------------------------------
    # ERASE / ROLL
    # --------------------------------------------------------

    else:

        (
            top,
            left,
            box_h,
            box_w,
        ) = sample_box(
            rng
        )

    manipulated = (
        protected.clone()
    )

    mask = torch.zeros(
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=protected.dtype,
        device=protected.device,
    )

    mask[
        :,
        top:top + box_h,
        left:left + box_w,
    ] = 1.0

    # --------------------------------------------------------
    # ERASE
    # --------------------------------------------------------

    if attack == "erase":

        manipulated[
            :,
            top:top + box_h,
            left:left + box_w,
        ] = 0.0

    # --------------------------------------------------------
    # COPY-MOVE
    # --------------------------------------------------------

    elif attack == "copy_move":

        if rectangles_overlap(
            src_top,
            src_left,
            box_h,
            box_w,
            top,
            left,
            box_h,
            box_w,
        ):
            raise RuntimeError(
                "Copy-move overlap detected."
            )

        source = (
            protected[
                :,
                src_top:src_top + box_h,
                src_left:src_left + box_w,
            ]
            .clone()
        )

        manipulated[
            :,
            top:top + box_h,
            left:left + box_w,
        ] = source

    # --------------------------------------------------------
    # ROLL
    # --------------------------------------------------------

    elif attack == "roll":

        region = (
            protected[
                :,
                top:top + box_h,
                left:left + box_w,
            ]
            .clone()
        )

        shift_y = max(
            1,
            box_h // 4,
        )

        shift_x = max(
            1,
            box_w // 4,
        )

        rolled = torch.roll(
            region,
            shifts=(
                shift_y,
                shift_x,
            ),
            dims=(
                1,
                2,
            ),
        )

        manipulated[
            :,
            top:top + box_h,
            left:left + box_w,
        ] = rolled

    return (
        manipulated.clamp(
            0.0,
            1.0,
        ),
        mask,
    )


# ============================================================
# ACTIVE SSPG DATASET
# ============================================================


class ActiveSSPGDataset(
    Dataset
):
    """
    Input:
        authentic source records only.

    Each source produces:
        authentic protected sample
        manipulated protected sample

    Therefore:
        detection balance = exactly 1:1.
    """

    def __init__(
        self,
        records,
        base_seed,
    ):
        self.records = list(
            records
        )

        self.base_seed = int(
            base_seed
        )

        if not self.records:
            raise RuntimeError(
                "No source records."
            )

        self.attack_assignment = (
            build_balanced_attack_assignment(
                len(self.records),
                self.base_seed,
            )
        )

    def __len__(
        self,
    ):
        return (
            len(self.records)
            * 2
        )

    def __getitem__(
        self,
        index,
    ):
        if index < 0:
            index = (
                len(self)
                + index
            )

        if (
            index < 0
            or index >= len(self)
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

        record = self.records[
            source_index
        ]

        image_path = Path(
            record[
                "image_path"
            ]
        )

        sample_key = (
            canonical_sample_key(
                record
            )
        )

        seed = stable_seed(
            self.base_seed,
            sample_key,
        )

        original = load_rgb_256(
            image_path
        )

        payload = (
            make_payload_deterministic(
                seed
            )
        )

        # Active V3.3 protection.
        protected = embed_payload(
            original.unsqueeze(0),
            payload.unsqueeze(0),
        ).squeeze(0)

        # ----------------------------------------------------
        # AUTHENTIC
        # --------------------------------------------------------

        if manipulated_flag == 0:

            received = protected

            mask = torch.zeros(
                1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                dtype=protected.dtype,
            )

            detection_label = (
                torch.tensor(
                    0.0,
                    dtype=torch.float32,
                )
            )

            attack = "authentic"

        # ----------------------------------------------------
        # MANIPULATED
        # --------------------------------------------------------

        else:

            attack = (
                self.attack_assignment[
                    source_index
                ]
            )

            rng = np.random.default_rng(
                seed
                + ATTACK_GEOMETRY_SEED_OFFSET
            )

            received, mask = (
                apply_controlled_attack(
                    protected,
                    attack,
                    rng,
                )
            )

            detection_label = (
                torch.tensor(
                    1.0,
                    dtype=torch.float32,
                )
            )

        return {
            "received":
                received,

            "payload":
                payload,

            "mask":
                mask,

            "detection_label":
                detection_label,

            "attack":
                attack,

            "source_path":
                str(image_path),

            "sample_key":
                sample_key,

            "source_seed":
                seed,

            "source_index":
                source_index,
        }