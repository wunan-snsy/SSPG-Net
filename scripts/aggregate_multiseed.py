from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "multiseed"
rows = []
for model in ("full", "no_graph", "no_provenance"):
    for seed in (20261001, 20261002, 20261003):
        p = BASE / model / str(seed) / "metrics.csv"
        row = pd.read_csv(p).iloc[0].to_dict()
        row.update(model=model.upper(), seed=seed)
        rows.append(row)
raw = pd.DataFrame(rows)
metrics = ["detection_auc", "detection_f1", "localization_pixel_f1", "localization_pixel_iou"]
raw[["model", "seed", *metrics]].to_csv(BASE / "seed_results.csv", index=False)
summary = raw.groupby("model")[metrics].agg(["mean", "std"])
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary.reset_index().to_csv(BASE / "mean_std_summary.csv", index=False)
paper = summary.reset_index()
for metric in metrics:
    paper[metric] = paper.apply(lambda r: f"{r[metric+'_mean']:.4f} ± {r[metric+'_std']:.4f}", axis=1)
paper[["model", *metrics]].to_csv(BASE / "Table_multiseed.csv", index=False)
print(paper[["model", *metrics]].to_string(index=False))
