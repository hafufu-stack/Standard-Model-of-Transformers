# -*- coding: utf-8 -*-
"""
Phase 371: Model Quality Index
================================
Define a single "Thermodynamic Quality Index" (TQI) that captures
overall model health from thermodynamic measurements.

Method:
1. Compute all thermodynamic variables for standard benchmark prompts.
2. Define TQI = weighted combination of:
   - Boltzmann fit R2
   - Carnot efficiency
   - Mach stability (closeness to 1)
   - P1T conservation (closeness to constant)
   - Negative specific heat presence
3. Compare TQI across model sizes and architectures.
4. Correlate TQI with downstream task performance proxies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import (load_model, load_any_model, save_results, save_figure,
                   measure_full_thermodynamics)

BENCHMARK_PROMPTS = [
    "The theory of relativity states that",
    "In quantum mechanics, the uncertainty principle",
    "Machine learning algorithms can be categorized",
    "The human genome contains approximately",
    "Water molecules consist of two hydrogen",
    "The speed of light in vacuum is",
    "Once upon a time in a distant galaxy",
    "The derivative of sin(x) is equal to",
]


def compute_tqi(model, tok, device, prompts):
    """Compute Thermodynamic Quality Index."""
    all_U = []
    all_T = []
    all_PR = []

    for prompt in prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
        all_U.append([t['U'] for t in thermo])
        all_T.append([t['T'] for t in thermo])
        all_PR.append([t['PR'] for t in thermo])

    all_U = np.array(all_U)
    all_T = np.array(all_T)
    all_PR = np.array(all_PR)

    # 1. Boltzmann R2: fit log(P) vs E at final layer
    U_final = all_U[:, -1]
    T_final = all_T[:, -1]
    # Use rank-based Boltzmann test
    if len(U_final) > 2:
        r_boltz, _ = stats.pearsonr(U_final, T_final)
        boltzmann_r2 = r_boltz ** 2
    else:
        boltzmann_r2 = 0.0

    # 2. Carnot efficiency (mean across prompts)
    eta_list = []
    for i in range(len(prompts)):
        T_hot = max(all_T[i])
        T_cold = min(all_T[i]) if min(all_T[i]) > 0 else 0.01
        eta = 1 - T_cold / (T_hot + 1e-10)
        eta_list.append(eta)
    carnot_eta = np.mean(eta_list)

    # 3. Mach stability: how close is final Mach to 1
    mach_list = []
    for i in range(len(prompts)):
        dU = np.diff(all_U[i])
        c_s = np.std(dU) + 1e-10
        mach = abs(dU[-1]) / c_s
        mach_list.append(mach)
    mach_stability = 1.0 / (1.0 + abs(np.mean(mach_list) - 1.0))

    # 4. P1T conservation: CV of P1*T across prompts
    p1t_list = []
    for i in range(len(prompts)):
        probs = np.exp(-all_T[i][-1])  # simplified P1
        p1t = probs * all_T[i][-1]
        p1t_list.append(p1t)
    p1t_cv = np.std(p1t_list) / (np.mean(p1t_list) + 1e-10)
    p1t_score = 1.0 / (1.0 + p1t_cv)

    # 5. Negative specific heat
    cv_negative = 0
    for i in range(len(prompts)):
        dU = np.diff(all_U[i])
        dT = np.diff(all_T[i])
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = dU / (dT + 1e-10)
        cv_negative += np.sum(cv < 0) / len(cv)
    cv_negative /= len(prompts)
    cv_score = cv_negative  # Higher = more negative Cv = expected

    # TQI = weighted average
    tqi = (0.25 * boltzmann_r2 +
           0.20 * carnot_eta +
           0.20 * mach_stability +
           0.20 * p1t_score +
           0.15 * cv_score)

    return {
        'tqi': float(tqi),
        'boltzmann_r2': float(boltzmann_r2),
        'carnot_eta': float(carnot_eta),
        'mach_stability': float(mach_stability),
        'p1t_score': float(p1t_score),
        'cv_score': float(cv_score),
        'mach_mean': float(np.mean(mach_list)),
    }


def main():
    print("=" * 70)
    print("Phase 371: Model Quality Index")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # Test across model sizes
    model_configs = [
        ('Qwen2.5-0.5B', lambda: load_model(device=device, size='0.5B')),
        ('Qwen2.5-1.5B', lambda: load_model(device=device, size='1.5B')),
    ]

    # Try TinyLlama if available
    try:
        test_model, test_tok = load_any_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                               device=device)
        del test_model, test_tok
        torch.cuda.empty_cache()
        model_configs.append(
            ('TinyLlama-1.1B', lambda: load_any_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                                       device=device))
        )
    except Exception:
        print("  TinyLlama not available, skipping")

    for name, loader in model_configs:
        print(f"\n=== {name} ===")
        model, tok = loader()
        tqi_result = compute_tqi(model, tok, device, BENCHMARK_PROMPTS)
        results[name] = tqi_result
        print(f"  TQI = {tqi_result['tqi']:.4f}")
        print(f"    Boltzmann R2: {tqi_result['boltzmann_r2']:.3f}")
        print(f"    Carnot eta: {tqi_result['carnot_eta']:.3f}")
        print(f"    Mach stability: {tqi_result['mach_stability']:.3f}")
        print(f"    P1T score: {tqi_result['p1t_score']:.3f}")
        print(f"    Cv score: {tqi_result['cv_score']:.3f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    n_models = len(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 371: Thermodynamic Quality Index", fontweight='bold')

    # (a) TQI comparison
    ax = axes[0]
    models = list(results.keys())
    tqis = [results[m]['tqi'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71'][:n_models]
    ax.bar(models, tqis, color=colors, alpha=0.8)
    ax.set_ylabel('TQI')
    ax.set_title('(a) Model Comparison', fontweight='bold')
    for i, (m, t) in enumerate(zip(models, tqis)):
        ax.text(i, t + 0.01, f'{t:.3f}', ha='center', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # (b) Component breakdown
    ax = axes[1]
    components = ['boltzmann_r2', 'carnot_eta', 'mach_stability', 'p1t_score', 'cv_score']
    comp_labels = ['Boltz R2', 'Carnot', 'Mach', 'P1T', 'Cv']
    x = np.arange(len(comp_labels))
    width = 0.8 / n_models
    for mi, m in enumerate(models):
        vals = [results[m][c] for c in components]
        ax.bar(x + mi * width - 0.4 + width/2, vals, width,
              label=m, color=colors[mi], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel('Score')
    ax.set_title('(b) Component Breakdown', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase371_tqi')
    plt.close()

    save_results('phase371_tqi', {
        'experiment': 'Thermodynamic Quality Index',
        'results': results,
    })


if __name__ == '__main__':
    main()
