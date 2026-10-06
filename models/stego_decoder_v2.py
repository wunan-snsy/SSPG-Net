import torch.nn as nn


class StegoDecoderV2(nn.Module):
    """
    Blind steganographic payload decoder V2.

    Input:
        stego image [B, 3, H, W]

    Output:
        payload logits [B, 1, H, W]

    IMPORTANT:
        The decoder receives ONLY the stego image.
        It never receives the original image.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                32, 16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                16, 1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        return self.net(x)