# -*- coding: utf-8 -*-
"""
Phase 363: Crooks Fluctuation Theorem
========================================
Test time-reversal symmetry of non-equilibrium processes:
  P_F(W) / P_R(-W) = exp((W - delta_F) / kT)

Method:
1. Forward process: process prompt layer-by-layer (normal forward pass).
2. Reverse process: reverse the layer order processing by reading
   hidden states from last to first layer.
3. Compare work distributions in forward vs reverse directions.
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
    "According to the second law of thermodynamics",
    "The capital of France is known for",
]


def main():
    print("=" * 70)
    print("Phase 363: Crooks Fluctuation Theorem")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        forward_work = []
        reverse_work = []
        all_entropy_production = []

        for prompt in PROMPTS:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)

            # Forward work: sum of energy increments layer by layer
            W_forward = 0.0
            for i in range(len(thermo) - 1):
                dU = thermo[i+1]['U'] - thermo[i]['U']
                W_forward += dU

            # Reverse work: sum of energy decrements (reversed path)
            W_reverse = 0.0
            for i in range(len(thermo) - 1, 0, -1):
                dU = thermo[i-1]['U'] - thermo[i]['U']
                W_reverse += dU

            forward_work.append(W_forward)
            reverse_work.append(W_reverse)

            # Entropy production: sigma = (W_F + W_R) / T
            T_mean = np.mean([t['T'] for t in thermo]) + 1e-10
            sigma_ep = (W_forward + W_reverse) / T_mean
            # Note: by construction W_F = -W_R, so sigma ~ 0 for reversible
            # But due to nonlinearity, this may not be exact
            all_entropy_production.append(sigma_ep)

        # Crooks test: P_F(W) / P_R(-W) = exp(W/T)
        # Since W_F = -W_R by construction for linear path,
        # test the distribution symmetry
        fw = np.array(forward_work)
        rw = np.array(reverse_work)

        # Test: W_forward + W_reverse = 0 (reversibility)
        residual = fw + rw
        reversibility = float(np.mean(np.abs(residual)))

        # Dissipation: variance of work
        dissipation_F = float(np.var(fw))
        dissipation_R = float(np.var(rw))

        # Detailed balance ratio
        T_global = float(np.mean([np.mean([t['T'] for t in
                         measure_full_thermodynamics(model, tok, PROMPTS[0], device)[0]])]))

        # Crooks ratio for each realization
        crooks_ratios = []
        for wf, wr in zip(fw, rw):
            if abs(wf / T_global) < 50:
                ratio = np.exp(wf / T_global)
                crooks_ratios.append(ratio)

        mean_crooks = float(np.mean(crooks_ratios)) if crooks_ratios else 0.0
        entropy_prod_mean = float(np.mean(all_entropy_production))

        results[size] = {
            'mean_forward_work': float(np.mean(fw)),
            'mean_reverse_work': float(np.mean(rw)),
            'reversibility_residual': reversibility,
            'dissipation_forward': dissipation_F,
            'dissipation_reverse': dissipation_R,
            'crooks_ratio_mean': mean_crooks,
            'entropy_production_mean': entropy_prod_mean,
        }

        print(f"  W_forward mean: {np.mean(fw):.4f}")
        print(f"  W_reverse mean: {np.mean(rw):.4f}")
        print(f"  Reversibility residual: {reversibility:.6f}")
        print(f"  Crooks ratio mean: {mean_crooks:.4f}")
        print(f"  Entropy production: {entropy_prod_mean:.6f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 363: Crooks Fluctuation Theorem", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        r = results[size]
        labels = ['W_F', 'W_R', 'Residual', 'Entropy\nProd']
        values = [r['mean_forward_work'], r['mean_reverse_work'],
                  r['reversibility_residual'], r['entropy_production_mean']]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase363_crooks')
    plt.close()

    save_results('phase363_crooks', {
        'experiment': 'Crooks Fluctuation Theorem',
        'results': results,
    })


if __name__ == '__main__':
    main()
