"""Sequential, Validation-only E2 replications with the fixed budget142 decoder."""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_clean_stage_a_v2 import gate, sha, write_json

OUT = ROOT / 'results/clean_revision_v2/budget142_e2'
DEC = ROOT / 'checkpoints/clean_revision_v2/decoder_budget142/best.pth'
DEC_STATUS = ROOT / 'results/clean_revision_v2/decoder_budget142/status.json'
SEEDS = (20261001, 20261002, 20261003)
MODELS = ('full', 'no_graph', 'no_provenance')


def verify():
    gate()
    state = json.loads(DEC_STATUS.read_text(encoding='utf-8'))
    if state['state'] != 'COMPLETE' or sha(DEC) != state['best_checkpoint_sha256']:
        raise RuntimeError('Budget decoder checkpoint or status changed')


def paths(model, seed):
    d = OUT / model / str(seed)
    return {'checkpoint': d / 'best.pth', 'history': d / 'training_log.csv',
            'predictions': d / 'val_predictions.csv', 'summary': d / 'metrics.csv',
            'protocol': d / 'protocol.json', 'failure': d / 'failure.json'}


def run_one(model, seed):
    verify()
    p = paths(model, seed)
    p['summary'].parent.mkdir(parents=True, exist_ok=False)
    import torch
    import training.e53r_training_protocol_v1 as proto
    import training.train_e53r_joint_v1 as joint
    torch.set_num_threads(4)
    protocol = replace(proto.PROTOCOL, training_seed=seed,
                       experiment_id=f'CLEAN-BUDGET142-E2-{model.upper()}-{seed}-V2',
                       protocol_status='CLEAN_BUDGET142_E2_FIXED_REPLICATION')
    proto.PROTOCOL = protocol
    joint.PROTOCOL = protocol
    digest = hashlib.sha256(proto.protocol_json().encode('utf-8')).hexdigest()
    joint.EXPECTED_PROTOCOL_SHA256 = digest
    joint.STEGO_CHECKPOINT = DEC
    if model == 'full':
        joint.BEST_CHECKPOINT = p['checkpoint']
        joint.HISTORY_PATH = p['history']
        joint.VAL_PREDICTIONS_PATH = p['predictions']
        joint.SUMMARY_PATH = p['summary']
        joint.PROTOCOL_SNAPSHOT_PATH = p['protocol']
        joint.main()
    else:
        import training.train_e53r_ablation_v1 as abl
        abl.PROTOCOL = protocol
        abl.paths = lambda variant: p
        abl.EXPERIMENTS[model] = protocol.experiment_id
        abl.train_variant(model)
    write_json(p['summary'].parent / 'lineage.json', {
        'experiment_id': protocol.experiment_id, 'training_seed': seed,
        'protocol_sha256': digest, 'decoder_sha256': sha(DEC),
        'checkpoint_sha256': sha(p['checkpoint']),
        'summary_sha256': sha(p['summary']), 'test_accessed': False})


def pipeline():
    verify()
    if OUT.exists():
        raise RuntimeError('E2 output directory already exists')
    OUT.mkdir(parents=True)
    write_json(OUT / 'protocol.json', {
        'experiment_id': 'CLEAN-BUDGET142-E2-V2', 'decoder_sha256': sha(DEC),
        'models': MODELS, 'training_seeds': SEEDS,
        'source': 'Frozen historical E2 three-seed design; no Test access.',
        'selection': 'Each run uses its own Validation-selected checkpoint and threshold.',
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid()})
    completed = []
    for model in MODELS:
        for seed in SEEDS:
            job = f'{model}/{seed}'
            write_json(OUT / 'status.json', {'state': 'RUNNING', 'job': job,
                                             'completed': completed, 'pid': os.getpid()})
            with (OUT / f'{model}_{seed}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                         '--model', model, '--seed', str(seed)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED='42'))
            if result.returncode:
                write_json(OUT / 'status.json', {'state': 'FAILED', 'job': job,
                                                 'completed': completed,
                                                 'returncode': result.returncode})
                raise RuntimeError(f'E2 run failed: {job}')
            completed.append(job)
    write_json(OUT / 'status.json', {'state': 'COMPLETE', 'completed': completed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if (args.model is None) != (args.seed is None):
        parser.error('--model and --seed must be provided together')
    if args.model is None:
        pipeline()
    else:
        run_one(args.model, args.seed)
