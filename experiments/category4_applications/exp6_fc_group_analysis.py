#!/usr/bin/env python3
"""
Category 4: Applications

Experiment 6: Functional Connectivity Group Analysis (ASD vs NT)

Scaffold to drop in subject-level FC matrices and a metadata CSV, then compute
consciousness-relevant metrics across groups (ASD vs neurotypical).

Expected inputs (user-provided later):
- data/metadata.csv with columns: subject_id, group (values: ASD, NT)
- data/fc/<subject_id>.npy containing a square FC matrix (symmetric, weighted)

Outputs:
- experiments/category4_applications/results/exp6_fc_group_analysis/metrics.csv
- Figures (group comparisons) saved to the same folder

If data is missing, the script prints guidance and exits cleanly.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import metrics as met

# Configuration
SEED = 42
np.random.seed(SEED)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data'
FC_DIR = DATA_DIR / 'fc'
METADATA_CSV = DATA_DIR / 'metadata.csv'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'exp6_fc_group_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase coherence (Kuramoto)
# ---------------------------------------------------------------------------

def simulate_phase_coherence(W: np.ndarray, K: float = 1.0, steps: int = 300, dt: float = 0.05, seed: int = SEED):
    np.random.seed(seed)
    n = W.shape[0]
    deg = W.sum(axis=1, keepdims=True) + 1e-9
    W_norm = W / deg
    omega = np.random.normal(0.0, 0.1, size=n)
    theta = np.random.uniform(0, 2*np.pi, size=n)
    R_series = []
    for _ in range(steps):
        phase_diff = np.subtract.outer(theta, theta)
        coupling_term = (W_norm * np.sin(-phase_diff)).sum(axis=1)
        dtheta = omega + K * coupling_term
        theta = (theta + dt * dtheta) % (2*np.pi)
        Z = np.exp(1j * theta).mean()
        R_series.append(np.abs(Z))
    R_series = np.array(R_series)
    return float(R_series.mean()), float(R_series.std())

# ---------------------------------------------------------------------------
# Metrics from FC matrix
# ---------------------------------------------------------------------------

def compute_fc_metrics(W: np.ndarray):
    if W.shape[0] != W.shape[1]:
        raise ValueError('FC matrix must be square')
    if not np.allclose(W, W.T, atol=1e-6):
        W = (W + W.T) / 2  # enforce symmetry
    W = np.maximum(W, 0)  # remove negatives
    n = W.shape[0]

    degree = W.sum(axis=1)
    sparsity = 1 - (W > 0).sum() / (n * n)

    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(W[i, :] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
        else:
            sub = W[np.ix_(neighbors, neighbors)]
            possible = k * (k - 1) / 2
            actual = sub.sum() / 2
            clustering[i] = actual / possible if possible > 0 else 0.0

    eigenvalues = np.linalg.eigvalsh(W)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) > 0:
        power = eigenvalues / eigenvalues.sum()
    else:
        power = np.ones(20) / 20
    if len(power) < 20:
        power = np.pad(power, (0, 20 - len(power)), mode='constant')
    else:
        power = power[-20:]

    synth_eig = np.arange(1, 21, dtype=float)
    cons = met.compute_all_metrics(power, synth_eig)

    R_dyn, metastability = simulate_phase_coherence(W, K=1.0, steps=200)

    # Simple efficiency proxy via spectral gap
    spectral_gap = 0.0
    if len(eigenvalues) > 1:
        spectral_gap = eigenvalues[-1] - eigenvalues[-2]

    return {
        'mean_degree': degree.mean(),
        'std_degree': degree.std(),
        'clustering': clustering.mean(),
        'sparsity': sparsity,
        'R_dyn': R_dyn,
        'metastability': metastability,
        'spectral_gap': spectral_gap,
        **cons,
    }

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {'subject_id', 'group'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    return df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('='*70)
    print('FC Group Analysis (ASD vs NT)')
    print('='*70)

    if not FC_DIR.exists():
        print(f"No FC directory found at {FC_DIR}. Create it and add <subject>.npy files.")
        return

    fc_files = sorted(FC_DIR.glob('*.npy'))
    if len(fc_files) == 0:
        print(f"No FC .npy files found in {FC_DIR}. Add subject FC matrices and rerun.")
        return

    if not METADATA_CSV.exists():
        print(f"No metadata CSV found at {METADATA_CSV}.")
        print('Expected columns: subject_id, group (values: ASD, NT).')
        return

    meta = load_metadata(METADATA_CSV)
    meta = meta.set_index('subject_id')

    records = []
    for fc_path in tqdm(fc_files, desc='Subjects'):
        sid = fc_path.stem
        if sid not in meta.index:
            print(f"Warning: {sid} missing in metadata; skipping")
            continue
        group = meta.loc[sid, 'group']
        W = np.load(fc_path)
        metrics = compute_fc_metrics(W)
        records.append({
            'subject_id': sid,
            'group': group,
            **metrics,
        })

    if len(records) == 0:
        print('No subjects processed. Check metadata and FC files.')
        return

    df = pd.DataFrame(records)
    out_csv = OUTPUT_DIR / 'fc_group_metrics.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

    # Simple group comparison plots (if both groups present)
    groups = df['group'].unique()
    if len(groups) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, col in zip(axes, ['R_dyn', 'clustering', 'C']):
            for g in groups:
                vals = df[df['group'] == g][col]
                ax.scatter([g]*len(vals), vals, alpha=0.6, label=g)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig_path = OUTPUT_DIR / 'group_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
    else:
        print('Only one group present; skipping comparison plots.')

    print('\nDone.')


if __name__ == '__main__':
    main()
