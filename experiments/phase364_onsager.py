# -*- coding: utf-8 -*-
"""
Phase 364: Onsager Reciprocal Relations
==========================================
Test whether transport coefficients between layers satisfy Onsager symmetry:
  L_ij = L_ji (near equilibrium)

Method:
1. Define thermodynamic "forces" at each layer: X_i = -grad(F_i)
2. Define thermodynamic "fluxes": J_i = dU_i/dt (rate of energy change)
3. Construct transport matrix L: J_i = sum_j L_ij * X_j
4. Test symmetry: L_ij = L_ji
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    "The mitochondria is the powerhouse of",
    "In philosophy, the concept of free will",
]


def main():
    print("=" * 70)
    print("Phase 364: Onsager Reciprocal Relations")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        all_forces = []
        all_fluxes = []

        for prompt in PROMPTS:
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)

            # Forces: negative gradient of free energy F = U - T*log(PR)
            F_vals = []
            for t in thermo:
                F = t['U'] - t['T'] * np.log(t['PR'] + 1e-10)
                F_vals.append(F)
            F_vals = np.array(F_vals)

            # Force = -dF/dl (negative gradient)
            forces = -np.diff(F_vals)  # (n_layers - 1,)

            # Fluxes: rate of energy change dU/dl
            U_vals = np.array([t['U'] for t in thermo])
            fluxes = np.diff(U_vals)  # (n_layers - 1,)

            all_forces.append(forces)
            all_fluxes.append(fluxes)

        all_forces = np.array(all_forces)  # (n_prompts, n_layers-1)
        all_fluxes = np.array(all_fluxes)

        n_transport = all_forces.shape[1]

        # Construct transport matrix L via linear regression
        # J_i = sum_j L_ij * X_j
        # Use least squares: L = J @ X^T @ (X @ X^T)^-1
        X = all_forces.T  # (n_layers-1, n_prompts)
        J = all_fluxes.T  # (n_layers-1, n_prompts)

        try:
            XXT = X @ X.T + 1e-8 * np.eye(n_transport)
            L = J @ X.T @ np.linalg.inv(XXT)
        except np.linalg.LinAlgError:
            L = np.zeros((n_transport, n_transport))

        # Test Onsager symmetry: L_ij = L_ji
        L_sym = 0.5 * (L + L.T)
        L_asym = 0.5 * (L - L.T)

        sym_norm = np.linalg.norm(L_sym)
        asym_norm = np.linalg.norm(L_asym)
        symmetry_ratio = sym_norm / (sym_norm + asym_norm + 1e-10)

        # Reciprocity score: correlation between L_ij and L_ji
        upper = []
        lower = []
        for i in range(n_transport):
            for j in range(i+1, n_transport):
                upper.append(L[i, j])
                lower.append(L[j, i])
        from scipy.stats import pearsonr
        if len(upper) > 2:
            r_onsager, p_onsager = pearsonr(upper, lower)
        else:
            r_onsager, p_onsager = 0.0, 1.0

        results[size] = {
            'symmetry_ratio': float(symmetry_ratio),
            'onsager_correlation': float(r_onsager),
            'onsager_pvalue': float(p_onsager),
            'sym_norm': float(sym_norm),
            'asym_norm': float(asym_norm),
            'transport_matrix_size': n_transport,
        }

        print(f"  Onsager symmetry ratio: {symmetry_ratio:.4f}")
        print(f"  L_ij vs L_ji correlation: r={r_onsager:.4f}, p={p_onsager:.4f}")
        print(f"  Symmetric norm: {sym_norm:.4f}")
        print(f"  Antisymmetric norm: {asym_norm:.4f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 364: Onsager Reciprocal Relations", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        r = results[size]
        labels = ['Symmetry\nRatio', 'Onsager\nCorr']
        values = [r['symmetry_ratio'], r['onsager_correlation']]
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase364_onsager')
    plt.close()

    save_results('phase364_onsager', {
        'experiment': 'Onsager Reciprocal Relations',
        'results': results,
    })


if __name__ == '__main__':
    main()
