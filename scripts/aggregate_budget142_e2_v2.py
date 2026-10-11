"""Verify and aggregate the nine fixed budget142 E2 Validation replications."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/clean_revision_v2/budget142_e2'
MODELS = ('full', 'no_graph', 'no_provenance')
SEEDS = (20261001, 20261002, 20261003)
METRICS = ('detection_auc', 'detection_f1', 'localization_pixel_f1', 'localization_pixel_iou')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status = json.loads((BASE / 'status.json').read_text(encoding='utf-8'))
    if status['state'] != 'COMPLETE' or len(status['completed']) != 9:
        raise RuntimeError('E2 is not complete')
    if (BASE / 'seed_results.csv').exists():
        raise RuntimeError('Refusing to overwrite E2 aggregation')
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            d = BASE / model / str(seed)
            lineage = json.loads((d / 'lineage.json').read_text(encoding='utf-8'))
            if (digest(d / 'best.pth') != lineage['checkpoint_sha256'] or
                    digest(d / 'metrics.csv') != lineage['summary_sha256']):
                raise RuntimeError(f'Artifact mismatch: {model}/{seed}')
            row = pd.read_csv(d / 'metrics.csv').iloc[0].to_dict()
            if bool(row['test_accessed']):
                raise RuntimeError(f'Test accessed: {model}/{seed}')
            row.update(model=model.upper(), seed=seed)
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw[['model', 'seed', *METRICS]].to_csv(BASE / 'seed_results.csv', index=False)
    summary = raw.groupby('model', sort=False)[list(METRICS)].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(BASE / 'mean_std_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
