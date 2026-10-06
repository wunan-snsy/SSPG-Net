import random

import torch


def _random_box(
    h,
    w,
    min_ratio=0.10,
    max_ratio=0.40,
):

    box_h = random.randint(
        max(2, int(h * min_ratio)),
        max(2, int(h * max_ratio)),
    )

    box_w = random.randint(
        max(2, int(w * min_ratio)),
        max(2, int(w * max_ratio)),
    )

    y0 = random.randint(
        0,
        h - box_h,
    )

    x0 = random.randint(
        0,
        w - box_w,
    )

    y1 = y0 + box_h
    x1 = x0 + box_w

    return (
        y0,
        y1,
        x0,
        x1,
    )


def _copy_move_single(
    image,
    mask,
):

    _, h, w = image.shape

    y0, y1, x0, x1 = _random_box(
        h,
        w,
    )

    box_h = y1 - y0
    box_w = x1 - x0

    src_y = random.randint(
        0,
        h - box_h,
    )

    src_x = random.randint(
        0,
        w - box_w,
    )

    source = image[
        :,
        src_y:src_y + box_h,
        src_x:src_x + box_w,
    ].clone()

    image[
        :,
        y0:y1,
        x0:x1,
    ] = source

    mask[
        :,
        y0:y1,
        x0:x1,
    ] = 1.0

    return image, mask


def _roll_single(
    image,
    mask,
):

    _, h, w = image.shape

    y0, y1, x0, x1 = _random_box(
        h,
        w,
    )

    patch = image[
        :,
        y0:y1,
        x0:x1,
    ].clone()

    patch_width = (
        x1 - x0
    )

    shift = random.randint(
        1,
        max(
            1,
            patch_width // 2,
        ),
    )

    patch = torch.roll(
        patch,
        shifts=shift,
        dims=2,
    )

    image[
        :,
        y0:y1,
        x0:x1,
    ] = patch

    mask[
        :,
        y0:y1,
        x0:x1,
    ] = 1.0

    return image, mask


def _erase_single(
    image,
    mask,
):

    _, h, w = image.shape

    y0, y1, x0, x1 = _random_box(
        h,
        w,
    )

    # Use the image's own channel means instead of
    # an arbitrary fixed RGB color.

    fill = image.mean(
        dim=(1, 2),
        keepdim=True,
    )

    image[
        :,
        y0:y1,
        x0:x1,
    ] = fill

    mask[
        :,
        y0:y1,
        x0:x1,
    ] = 1.0

    return image, mask


def synthetic_tamper(
    x,
    tamper_probability=0.5,
):

    """
    Active SSPG training tamper generator V1.

    For each protected image independently:

        probability 1-p:
            authentic protected sample
            label = 0
            mask = zero

        probability p:
            manipulated protected sample
            label = 1
            mask = actual manipulated region

    This is an engineering V1 protocol and is NOT claimed
    to reproduce the exact attack protocol of the manuscript.
    """

    y = x.clone()

    b, _, h, w = x.shape

    masks = torch.zeros(
        (
            b,
            1,
            h,
            w,
        ),
        device=x.device,
        dtype=x.dtype,
    )

    labels = torch.zeros(
        b,
        device=x.device,
        dtype=x.dtype,
    )

    attack_types = []

    for i in range(b):

        # ----------------------------------------------------
        # Authentic protected sample
        # ----------------------------------------------------

        if random.random() >= tamper_probability:

            attack_types.append(
                "authentic"
            )

            continue

        # ----------------------------------------------------
        # Manipulated protected sample
        # ----------------------------------------------------

        attack = random.choice(
            [
                "copy_move",
                "roll",
                "erase",
            ]
        )

        image_i = y[i]
        mask_i = masks[i]

        if attack == "copy_move":

            image_i, mask_i = (
                _copy_move_single(
                    image_i,
                    mask_i,
                )
            )

        elif attack == "roll":

            image_i, mask_i = (
                _roll_single(
                    image_i,
                    mask_i,
                )
            )

        elif attack == "erase":

            image_i, mask_i = (
                _erase_single(
                    image_i,
                    mask_i,
                )
            )

        else:

            raise RuntimeError(
                f"Unknown attack: {attack}"
            )

        y[i] = image_i
        masks[i] = mask_i

        labels[i] = 1.0

        attack_types.append(
            attack
        )

    return (
        y,
        masks,
        labels,
        attack_types,
    )