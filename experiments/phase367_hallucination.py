# -*- coding: utf-8 -*-
"""
Phase 367: Hallucination Detection via Thermodynamics
=====================================================
Test whether thermodynamic signatures can distinguish factual
vs. hallucinated/nonsensical outputs.

Method:
1. Provide factual prompts (model should produce low-entropy, confident outputs).
2. Provide prompts designed to elicit hallucination (contradictory, nonsensical).
3. Compare thermodynamic signatures: T, PR, Mach, Carnot efficiency.
4. Train a simple classifier and measure AUROC.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

# Factual prompts (should be low temperature / high confidence)
FACTUAL = [
    "The chemical formula for water is",
    "The speed of light is approximately",
    "DNA stands for deoxyribonucleic",
    "The Earth orbits the Sun in approximately",
    "Gravity on Earth is approximately 9.8",
    "The capital of Japan is",
    "Photosynthesis converts carbon dioxide and water into",
    "The human body has 206",
]

# Hallucination-inducing prompts (contradictory, nonsensical, or ambiguous)
HALLUCINATION = [
    "The square root of negative banana is",
    "Yesterday, the year 3057 witnessed the",
    "The invisible color of magnetic sounds",
    "When rocks dream, they typically imagine",
    "The mathematical proof that 1 equals 2 shows",
    "The president of the moon declared that",
    "Breathing underwater without equipment is achieved by",
    "The weight of a thought in kilograms is",
]


def extract_features(thermo_list):
    """Extract classification features from thermodynamic profile."""
    Us = [t['U'] for t in thermo_list]
    Ts = [t['T'] for t in thermo_list]
    PRs = [t['PR'] for t in thermo_list]

    n = len(Us)
    # Features
    T_final = Ts[-1]
    T_mean = np.mean(Ts)
    T_var = np.var(Ts)
    U_final = Us[-1]
    U_growth = Us[-1] / (Us[0] + 1e-10)
    PR_final = PRs[-1]
    PR_growth = PRs[-1] / (PRs[0] + 1e-10)

    # Mach-like: dU/dl / mean_dU
    dU = np.diff(Us)
    c_s = np.std(dU) + 1e-10
    mach_final = abs(dU[-1]) / c_s if len(dU) > 0 else 0

    # Carnot-like efficiency
    T_hot = max(Ts)
    T_cold = min(Ts) if min(Ts) > 0 else 0.01
    eta = 1 - T_cold / (T_hot + 1e-10)

    # Entropy production
    S = [np.log(p + 1e-10) + t for p, t in zip(PRs, Ts)]
    dS = np.diff(S)
    entropy_prod = np.sum(np.abs(dS))

    return [T_final, T_mean, T_var, U_final, U_growth,
            PR_final, PR_growth, mach_final, eta, entropy_prod]


def main():
    print("=" * 70)
    print("Phase 367: Hallucination Detection")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        features = []
        labels = []

        # Factual prompts (label=0)
        for prompt in FACTUAL:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
            feat = extract_features(thermo)
            features.append(feat)
            labels.append(0)

        # Hallucination prompts (label=1)
        for prompt in HALLUCINATION:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
            feat = extract_features(thermo)
            features.append(feat)
            labels.append(1)

        X = np.array(features)
        y = np.array(labels)

        # Simple classifier (leave-one-out for small dataset)
        from sklearn.model_selection import cross_val_predict
        clf = LogisticRegression(max_iter=1000)
        try:
            y_pred = cross_val_predict(clf, X, y, cv=min(5, len(y)//2), method='predict_proba')[:, 1]
            auroc = roc_auc_score(y, y_pred)
        except Exception:
            auroc = 0.5

        # Feature importance
        clf.fit(X, y)
        feature_names = ['T_final', 'T_mean', 'T_var', 'U_final', 'U_growth',
                        'PR_final', 'PR_growth', 'Mach', 'Carnot_eta', 'Entropy_prod']
        importances = np.abs(clf.coef_[0])

        # Mean feature comparison
        factual_means = np.mean(X[:len(FACTUAL)], axis=0)
        halluc_means = np.mean(X[len(FACTUAL):], axis=0)

        results[size] = {
            'auroc': float(auroc),
            'feature_importances': {n: float(v) for n, v in zip(feature_names, importances)},
            'factual_means': {n: float(v) for n, v in zip(feature_names, factual_means)},
            'halluc_means': {n: float(v) for n, v in zip(feature_names, halluc_means)},
        }

        print(f"  AUROC: {auroc:.3f}")
        top3 = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:3]
        print(f"  Top features: {', '.join(f'{n}={v:.3f}' for n, v in top3)}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 367: Hallucination Detection via Thermodynamics", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        r = results[size]
        fnames = list(r['feature_importances'].keys())
        imps = [r['feature_importances'][f] for f in fnames]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(fnames)))
        ax.barh(fnames, imps, color=colors, alpha=0.8)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Qwen2.5-{size} (AUROC={r["auroc"]:.3f})', fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase367_hallucination')
    plt.close()

    save_results('phase367_hallucination', {
        'experiment': 'Hallucination Detection via Thermodynamics',
        'results': results,
    })


if __name__ == '__main__':
    main()
