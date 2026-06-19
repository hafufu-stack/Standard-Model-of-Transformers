# -*- coding: utf-8 -*-
"""
Phase 365: Entropy Production Rate
=====================================
Measure the rate of entropy production through the transformer layers,
and test if it follows the minimum entropy production principle
(Prigogine's theorem) near steady state.

Method:
1. Compute entropy S_i at each layer from PR and T.
2. Entropy production rate: dS/dl = S_{i+1} - S_i
3. Test if entropy production decreases towards final layers
   (minimum entropy production near steady state).
4. Compute total entropy production and compare with information
   processing rate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of relativity states that",
    "In quantum mechanics, the uncertainty principle",
    "Machine learning algorithms can be categorized",
    "The human genome contains approximately",
    "Water molecules consist of two hydrogen",
    "The speed of light in vacuum is",
    "Once upon a time in a distant galaxy",
    "The derivative of sin(x) is equal to",
]


def main():
    print("=" * 70)
    print("Phase 365: Entropy Production Rate")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        all_entropy_rate = []
        all_total_production = []

        for prompt in PROMPTS:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)

            # Entropy at each layer: S = log(PR) + T (information entropy)
            S_layers = []
            for t in thermo:
                S = np.log(t['PR'] + 1e-10) + t['T']
                S_layers.append(S)
            S_layers = np.array(S_layers)

            # Entropy production rate: dS/dl
            dS = np.diff(S_layers)
            all_entropy_rate.append(dS)

            # Total entropy production
            total_prod = np.sum(np.abs(dS))
            all_total_production.append(total_prod)

        all_entropy_rate = np.array(all_entropy_rate)  # (n_prompts, n_layers-1)
        mean_rate = np.mean(all_entropy_rate, axis=0)

        # Test: does entropy production rate decrease? (Prigogine)
        n_half = len(mean_rate) // 2
        rate_first_half = np.mean(np.abs(mean_rate[:n_half]))
        rate_second_half = np.mean(np.abs(mean_rate[n_half:]))
        prigogine_ratio = rate_second_half / (rate_first_half + 1e-10)

        # Trend test: is |dS/dl| decreasing?
        abs_rate = np.abs(mean_rate)
        layers = np.arange(len(abs_rate))
        slope, intercept, r_trend, p_trend, _ = stats.linregress(layers, abs_rate)

        # Total entropy production
        mean_total = float(np.mean(all_total_production))

        results[size] = {
            'prigogine_ratio': float(prigogine_ratio),
            'rate_first_half': float(rate_first_half),
            'rate_second_half': float(rate_second_half),
            'trend_slope': float(slope),
            'trend_r2': float(r_trend**2),
            'trend_pvalue': float(p_trend),
            'total_entropy_production': mean_total,
            'mean_rate_profile': mean_rate.tolist(),
        }

        print(f"  Prigogine ratio (late/early): {prigogine_ratio:.4f}")
        print(f"  Trend slope: {slope:.6f} (R2={r_trend**2:.3f})")
        print(f"  Total entropy production: {mean_total:.4f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 365: Entropy Production Rate", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        rate = results[size]['mean_rate_profile']
        ax.plot(range(len(rate)), rate, 'o-', color='#e74c3c', lw=2, markersize=4)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Layer transition')
        ax.set_ylabel('dS/dl (entropy production rate)')
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold')
        ax.grid(alpha=0.3)

        prig = results[size]['prigogine_ratio']
        ax.text(0.95, 0.95, f'Prigogine ratio: {prig:.3f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    save_figure(fig, 'phase365_entropy_production')
    plt.close()

    save_results('phase365_entropy_production', {
        'experiment': 'Entropy Production Rate',
        'results': results,
    })


if __name__ == '__main__':
    main()
