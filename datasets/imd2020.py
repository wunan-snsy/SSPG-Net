from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class IMD2020Dataset(Dataset):
    """
    IMD2020 dataset based on the frozen manifest.

    label:
        0 = authentic
        1 = manipulated

    For authentic images:
        localization mask = all zeros

    For manipulated images:
        localization mask = official/provided GT mask

    IMPORTANT:
        image and mask use different interpolation methods:
        image -> bilinear
        mask  -> nearest
    """

    def __init__(
        self,
        root,
        manifest,
        split="train",
        image_size=128,
    ):
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.split = split
        self.image_size = image_size

        df = pd.read_csv(self.manifest)

        self.df = (
            df[df["split"] == split]
            .reset_index(drop=True)
        )

        if len(self.df) == 0:
            raise RuntimeError(
                f"No samples found for split={split}"
            )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = self.root / row["image_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.image_transform(image)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Localization mask
        # --------------------------------------------------

        if int(row["label"]) == 1:

            mask_path = self.root / row["mask_path"]

            with Image.open(mask_path) as mask:
                mask = mask.convert("L")
                mask = self.mask_transform(mask)

            # Force binary GT after nearest-neighbor resize
            mask = (mask > 0.5).float()

        else:

            mask = torch.zeros(
                (1, self.image_size, self.image_size),
                dtype=torch.float32,
            )

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "sample_id": int(row["sample_id"]),
            "group": str(row["group"]),
            "image_path": str(row["image_path"]),
        }