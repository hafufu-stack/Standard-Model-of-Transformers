# -*- coding: utf-8 -*-
"""
Phase 370: OOD Detection via Mach Number
==========================================
Test whether the Mach number (and other thermodynamic variables)
can detect out-of-distribution inputs more effectively than
standard methods.

Method:
1. Define in-distribution prompts (normal English text).
2. Define OOD prompts (corrupted, foreign scripts, code, random tokens).
3. Compare Mach number and other thermodynamic features between ID and OOD.
4. Measure AUROC for OOD detection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

# In-distribution: normal English text
ID_PROMPTS = [
    "The weather today is expected to be sunny with",
    "Scientists have discovered a new species of",
    "The stock market showed signs of recovery as",
    "In the latest developments, researchers found that",
    "The conference proceedings will be published in",
    "According to recent studies, exercise can improve",
    "The government announced new policies regarding",
    "Technology companies are investing heavily in",
]

# Out-of-distribution
OOD_PROMPTS = [
    # Random tokens / corrupted
    "xkcd fhqwhgads qwerty asdf zxcv bnm",
    "!@#$%^&*() 12345 abcde fghij",
    "aaaa bbbb cccc dddd eeee ffff gggg",
    # Code
    "def __init__(self, x): return x.reshape(-1, 3).sum(axis=0)",
    "SELECT * FROM users WHERE id = 1; DROP TABLE",
    # Extreme repetition
    "the the the the the the the the the the",
    # Mixed scripts
    "Hello world 42 foo bar baz qux corge grault",
    "Lorem ipsum dolor sit amet consectetur adipiscing",
]


def extract_ood_features(thermo_list):
    """Extract features for OOD detection."""
    Us = [t['U'] for t in thermo_list]
    Ts = [t['T'] for t in thermo_list]
    PRs = [t['PR'] for t in thermo_list]
    PRTs = [t['PRT'] for t in thermo_list]

    dU = np.diff(Us)
    c_s = np.std(dU) + 1e-10

    return {
        'T_final': Ts[-1],
        'T_mean': np.mean(Ts),
        'U_final': Us[-1],
        'PR_final': PRs[-1],
        'PRT_final': PRTs[-1],
        'mach_final': abs(dU[-1]) / c_s if len(dU) > 0 else 0,
        'mach_mean': np.mean(np.abs(dU)) / c_s,
        'eta': 1 - min(Ts) / (max(Ts) + 1e-10),
        'entropy_total': sum(np.abs(np.diff([np.log(p+1e-10) + t for p, t in zip(PRs, Ts)]))),
        'U_var': np.var(Us),
    }


def main():
    print("=" * 70)
    print("Phase 370: OOD Detection via Mach Number")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        features = []
        labels = []

        # ID prompts
        for prompt in ID_PROMPTS:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
            feat = extract_ood_features(thermo)
            features.append(feat)
            labels.append(0)

        # OOD prompts
        for prompt in OOD_PROMPTS:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
            feat = extract_ood_features(thermo)
            features.append(feat)
            labels.append(1)

        feature_names = list(features[0].keys())
        X = np.array([[f[n] for n in feature_names] for f in features])
        y = np.array(labels)

        # Single-feature AUROC for each thermodynamic variable
        single_aurocs = {}
        for fi, fname in enumerate(feature_names):
            try:
                auroc = roc_auc_score(y, X[:, fi])
                # Also try inverse (lower = more OOD)
                auroc_inv = roc_auc_score(y, -X[:, fi])
                single_aurocs[fname] = float(max(auroc, auroc_inv))
            except Exception:
                single_aurocs[fname] = 0.5

        # Multi-feature AUROC
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        clf = LogisticRegression(max_iter=1000)
        try:
            y_pred = cross_val_predict(clf, X, y, cv=min(4, len(y)//2), method='predict_proba')[:, 1]
            multi_auroc = roc_auc_score(y, y_pred)
        except Exception:
            multi_auroc = 0.5

        best_single = max(single_aurocs, key=single_aurocs.get)

        results[size] = {
            'single_aurocs': single_aurocs,
            'best_single_feature': best_single,
            'best_single_auroc': single_aurocs[best_single],
            'multi_feature_auroc': float(multi_auroc),
        }

        print(f"  Best single: {best_single} (AUROC={single_aurocs[best_single]:.3f})")
        print(f"  Multi-feature AUROC: {multi_auroc:.3f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 370: OOD Detection via Thermodynamics", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        r = results[size]
        fnames = list(r['single_aurocs'].keys())
        aurocs = [r['single_aurocs'][f] for f in fnames]
        colors = ['#e74c3c' if a > 0.7 else '#3498db' for a in aurocs]
        ax.barh(fnames, aurocs, color=colors, alpha=0.8)
        ax.axvline(0.5, color='gray', ls='--', alpha=0.5, label='Random')
        ax.axvline(0.7, color='orange', ls='--', alpha=0.5, label='Good')
        ax.set_xlabel('AUROC')
        ax.set_title(f'Qwen2.5-{size} (Multi={r["multi_feature_auroc"]:.3f})', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase370_ood')
    plt.close()

    save_results('phase370_ood', {
        'experiment': 'OOD Detection via Mach Number',
        'results': results,
    })


if __name__ == '__main__':
    main()
