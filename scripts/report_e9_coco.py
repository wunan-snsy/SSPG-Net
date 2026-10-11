"""Create an honest E9 results report, figure, and paired provenance CIs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_e9_coco import BANDS, MANIFEST_ROOT, OUT_ROOT, REPLICATES, SEED, metric


OUT = OUT_ROOT / "report"
LABELS = {"01_05": "1–5%", "05_10": "5–10%",
          "10_15": "10–15%", "15_30": "15–30%"}
VARIANTS = ("full", "no_provenance")


def paired_delta(full: pd.DataFrame, no_prov: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ids = sorted(full.source_id.unique())
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, len(ids), size=(REPLICATES, len(ids)))
    rows = []
    for band in BANDS:
        subsets = []
        for frame in (full, no_prov):
            part = frame[frame.band.isin(("authentic", band))].copy()
            part["source_order"] = pd.Categorical(part.source_id, categories=ids, ordered=True)
            part = part.sort_values(["source_order", "attack"]).reset_index(drop=True)
            if len(part) != 800 or not (part.source_id.to_numpy().reshape(200, 4) == np.asarray(ids)[:, None]).all():
                raise RuntimeError("FULL/NO_PROVENANCE source pairing mismatch")
            subsets.append(part)
        first, second = subsets
        point_a, point_b = metric(first, threshold), metric(second, threshold)
        values = {name: [] for name in point_a}
        blocks = np.arange(800).reshape(200, 4)
        for picks in resamples:
            positions = blocks[picks].ravel()
            a = metric(first.iloc[positions], threshold)
            b = metric(second.iloc[positions], threshold)
            for name in values:
                values[name].append(a[name] - b[name])
        for name in values:
            rows.append({"band": band, "metric": name,
                         "full_minus_no_provenance": point_a[name] - point_b[name],
                         "ci_low": float(np.percentile(values[name], 2.5)),
                         "ci_high": float(np.percentile(values[name], 97.5)),
                         "bootstrap_replicates": REPLICATES, "source_clusters": len(ids)})
        print(f"paired {band} complete", flush=True)
    return pd.DataFrame(rows)


def figure(summary: pd.DataFrame) -> None:
    colors = {"full": "#16639a", "no_provenance": "#d0783e"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(4)
    for axis, metric_name, title in [
        (axes[0], "pixel_f1", "Pixel F1 (authentic + manipulated)"),
        (axes[1], "detection_recall", "Manipulated-image detection recall"),
    ]:
        for variant in VARIANTS:
            part = summary[(summary.variant == variant) & (summary.metric == metric_name)]
            part = part.set_index("band").loc[list(BANDS)]
            y = part.value.to_numpy(dtype=float)
            lower = y - part.ci_low.to_numpy(dtype=float)
            upper = part.ci_high.to_numpy(dtype=float) - y
            axis.errorbar(x, y, yerr=np.vstack((lower, upper)), marker="o",
                          capsize=3, linewidth=2, color=colors[variant],
                          label=variant.upper().replace("_", " "))
        axis.set_xticks(x, [LABELS[b] for b in BANDS])
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Actual tampered-mask area")
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "e9_area_performance.png", dpi=300)
    fig.savefig(OUT / "e9_area_performance.pdf")
    plt.close(fig)


def main() -> None:
    protocol = json.loads((MANIFEST_ROOT / "protocol.json").read_text(encoding="utf-8"))
    for variant in VARIANTS:
        status = json.loads((OUT_ROOT / variant / "run.json").read_text(encoding="utf-8"))
        if status["status"] != "COMPLETE":
            raise RuntimeError(f"{variant} E9 inference incomplete")
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(OUT_ROOT / "full/predictions.csv")
    no_prov = pd.read_csv(OUT_ROOT / "no_provenance/predictions.csv")
    summary = pd.concat([pd.read_csv(OUT_ROOT / variant / "summary_with_95ci.csv")
                         for variant in VARIANTS], ignore_index=True)
    summary.to_csv(OUT / "e9_metrics_with_95ci.csv", index=False)
    deltas = paired_delta(full, no_prov, float(protocol["detection_threshold"]))
    deltas.to_csv(OUT / "full_minus_no_provenance_paired_95ci.csv", index=False)
    figure(summary)
    lines = [
        "# E9: COCO val2017 independent source-image and small-region evaluation",
        "",
        "Protocol fixed before E9 inference: 200 selected source images, 200 protected authentic samples, and 2,400 manipulated samples across three attacks and four realized area bands. No model was retrained. The 15–30% band plus authentic samples is the primary cross-dataset comparison (800 samples); smaller bands are the prespecified localization stress test.",
        "",
        f"Detection threshold {protocol['detection_threshold']:.12f}; localization threshold 0.5; alpha 0.010; k=8. All reported intervals use {REPLICATES:,} source-level bootstrap resamples (seed {SEED}).",
        "",
        "| Model | Area | AUC [95% CI] | Detection recall [95% CI] | Pixel F1 [95% CI] | IoU [95% CI] | Boundary F1, 2 px [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for band in BANDS:
            part = summary[(summary.variant == variant) & (summary.band == band)].set_index("metric")
            def fmt(name: str) -> str:
                row = part.loc[name]
                return f"{row.value:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
            lines.append(f"| {variant.upper()} | {LABELS[band]} | {fmt('auc')} | {fmt('detection_recall')} | {fmt('pixel_f1')} | {fmt('pixel_iou')} | {fmt('boundary_f1_2px')} |")
    lines += [
        "",
        "Pixel F1 and IoU in the table pool the 200 authentic and 600 manipulated samples within each band, matching the existing E4 aggregation. Boundary F1 is computed on manipulated samples only. The complete CSV also contains manipulated-only Pixel F1/IoU and Detection F1.",
        "",
        "## Interpretation",
        "",
        "The four area bands expose a material loss of detection and localization performance on smaller manipulations. Report this performance limit explicitly; do not describe the experiment as validation on native COCO forgeries or unconstrained real-world attacks. NO_PROVENANCE is the existing trained structural ablation; these paired results do not establish parameter-matched causality.",
        "",
        "FULL's Pixel F1 advantage over NO_PROVENANCE in the 1–5% band is small, and its paired 95% interval includes zero. The larger-area Pixel F1 differences have positive paired intervals. See `full_minus_no_provenance_paired_95ci.csv` for exact estimates.",
        "",
        "The 2-pixel Boundary F1 measures tight boundary alignment and is substantially lower than region overlap scores; the current 14×14 localization grid limits boundary precision. The GT masks are rectangles generated by the controlled attacks.",
        "",
        "## Reproducibility",
        "",
        f"Source manifest SHA-256: `{protocol['source_manifest_sha256']}`  ",
        f"Attack plan SHA-256: `{protocol['attack_plan_sha256']}`  ",
        f"FULL checkpoint SHA-256: `{protocol['checkpoint_sha256']['full']}`  ",
        f"Decoder SHA-256: `{protocol['checkpoint_sha256']['decoder']}`",
        "",
        "Hash and perceptual-hash screening is a conservative contamination check, not proof against all crops or semantic overlap.",
    ]
    (OUT / "E9_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "E9_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
