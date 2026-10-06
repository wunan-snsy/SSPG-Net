from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "results" / "robustness" / "Table_benign_attack.csv"
OUTFILE = ROOT / "results" / "robustness" / "Figure_benign_attack.png"


def main() -> None:
    frame = pd.read_csv(INFILE)
    labels = frame["condition"].str.replace("gaussian_blur", "blur", regex=False)
    x = range(len(frame))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, frame["roc_auc"], "o-", label="ROC-AUC")
    axes[0].plot(x, frame["bit_accuracy"], "s-", label="Bit accuracy")
    axes[0].set_ylim(0.7, 1.01)
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(x, frame["localization_pixel_f1"], "o-", label="Pixel F1")
    axes[1].plot(x, frame["localization_pixel_iou"], "s-", label="Pixel IoU")
    axes[1].set_ylim(0.2, 0.85)
    axes[1].set_ylabel("Localization score")
    axes[1].set_xticks(list(x), labels, rotation=35, ha="right")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    fig.suptitle("SSPG-Net robustness to benign processing")
    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTFILE)


if __name__ == "__main__":
    main()
