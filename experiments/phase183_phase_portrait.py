# -*- coding: utf-8 -*-
"""
Phase 183: Thermodynamic Phase Portrait
Create full U-T-S phase space visualization with vector fields.
Identify attractor structures in the thermodynamic state space.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "DNA stores hereditary genetic information",
    "Entropy measures disorder in physical systems",
    "The Turing machine is a fundamental model of computation",
    "Superconductors carry current with zero resistance",
    "The human genome contains approximately three billion base pairs",
    "Artificial neural networks are inspired by biological neurons",
    "Thermodynamics governs the flow of energy in systems",
    "The Schrodinger equation describes quantum wave functions",
    "Information entropy quantifies the uncertainty in a message",
    "The strong nuclear force binds quarks into hadrons",
    "Riemann hypothesis relates to the distribution of prime numbers",
    "Enzyme catalysis accelerates biochemical reactions",
    "The cosmic web is the large scale structure of the universe",
    "Bayesian inference updates beliefs with new evidence",
    "Topological insulators conduct on their surface but not inside",
    "The observer effect in quantum mechanics alters measurements",
]


def main():
    print("=" * 70)
    print("Phase 183: Thermodynamic Phase Portrait")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    all_U, all_T, all_S = [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_vals, T_vals, S_vals = [], [], []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()

            U_vals.append(U if not np.isnan(U) else 0)
            T_vals.append(T if not np.isnan(T) else 0)
            S_vals.append(S if not np.isnan(S) else 0)

        all_U.append(U_vals)
        all_T.append(T_vals)
        all_S.append(S_vals)

    U_arr = np.array(all_U)
    T_arr = np.array(all_T)
    S_arr = np.array(all_S)

    U_mean = np.mean(U_arr, axis=0)
    T_mean = np.mean(T_arr, axis=0)
    S_mean = np.mean(S_arr, axis=0)

    # Vector field: dU, dT, dS per layer
    dU = np.diff(U_mean)
    dT = np.diff(T_mean)
    dS = np.diff(S_mean)

    # === Visualization ===
    fig = plt.figure(figsize=(18, 12))

    # (a) 3D Phase Portrait
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i in range(min(10, len(PROMPTS))):
        ax1.plot(U_arr[i], T_arr[i], S_arr[i], '-', alpha=0.3, linewidth=0.8)
    ax1.plot(U_mean, T_mean, S_mean, 'k-', linewidth=3, label='Mean trajectory')
    ax1.scatter(U_mean[0], T_mean[0], S_mean[0], s=100, c='green', marker='^')
    ax1.scatter(U_mean[-1], T_mean[-1], S_mean[-1], s=100, c='red', marker='v')
    ax1.set_xlabel('$U$ (Energy)')
    ax1.set_ylabel('$T$ (Temperature)')
    ax1.set_zlabel('$S$ (Entropy)')
    ax1.set_title('(a) 3D Phase Portrait')

    # (b) T-S diagram with vector field
    ax2 = fig.add_subplot(2, 3, 2)
    for i in range(min(10, len(PROMPTS))):
        ax2.plot(S_arr[i], T_arr[i], '-', alpha=0.15, linewidth=0.8, color='gray')
    ax2.plot(S_mean, T_mean, 'o-', color='#2c3e50', markersize=4, linewidth=2, label='Mean')
    # Vector arrows
    for li in range(0, n_layers - 1, 2):
        ax2.annotate('', xy=(S_mean[li+1], T_mean[li+1]),
                     xytext=(S_mean[li], T_mean[li]),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    ax2.scatter(S_mean[0], T_mean[0], s=120, c='green', marker='^', zorder=5, label='Input')
    ax2.scatter(S_mean[-1], T_mean[-1], s=120, c='red', marker='v', zorder=5, label='Output')
    ax2.set_xlabel('Entropy $S$')
    ax2.set_ylabel('Temperature $T$')
    ax2.set_title('(b) T-S Diagram (Thermodynamic Cycle)')
    ax2.legend(fontsize=8)

    # (c) U-T diagram
    ax3 = fig.add_subplot(2, 3, 3)
    for i in range(min(10, len(PROMPTS))):
        ax3.plot(U_arr[i], T_arr[i], '-', alpha=0.15, linewidth=0.8, color='gray')
    scatter = ax3.scatter(U_mean, T_mean, c=np.arange(n_layers), cmap='coolwarm',
                          s=60, edgecolors='black', zorder=5)
    ax3.plot(U_mean, T_mean, 'k-', linewidth=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax3, label='Layer')
    ax3.set_xlabel('Energy $U$')
    ax3.set_ylabel('Temperature $T$')
    ax3.set_title('(c) U-T Phase Space')

    # (d) Stream plot of dT vs dU
    ax4 = fig.add_subplot(2, 3, 4)
    layers_mid = np.arange(n_layers - 1) + 0.5
    # Quiver plot
    ax4.quiver(U_mean[:-1], T_mean[:-1], dU, dT, layers_mid,
               cmap='coolwarm', scale_units='xy', angles='xy', alpha=0.8)
    ax4.scatter(U_mean[0], T_mean[0], s=120, c='green', marker='^', zorder=5, label='Start')
    ax4.scatter(U_mean[-1], T_mean[-1], s=120, c='red', marker='v', zorder=5, label='End')
    ax4.set_xlabel('$U$')
    ax4.set_ylabel('$T$')
    ax4.set_title('(d) Thermodynamic Flow Field')
    ax4.legend(fontsize=8)

    # (e) Variance landscape
    ax5 = fig.add_subplot(2, 3, 5)
    U_var = np.var(U_arr, axis=0)
    T_var = np.var(T_arr, axis=0)
    S_var = np.var(S_arr, axis=0)
    layers = np.arange(n_layers)
    ax5.plot(layers, U_var / (U_var.max() + 1e-10), 'o-', label='$U$ var', linewidth=2, markersize=3)
    ax5.plot(layers, T_var / (T_var.max() + 1e-10), 's-', label='$T$ var', linewidth=2, markersize=3)
    ax5.plot(layers, S_var / (S_var.max() + 1e-10), '^-', label='$S$ var', linewidth=2, markersize=3)
    ax5.axvline(x=21, color='#f39c12', linestyle='--', label='$L_0$')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Normalized Variance')
    ax5.set_title('(e) Fluctuation Landscape')
    ax5.legend(fontsize=7)

    # (f) Free energy F = U - TS
    ax6 = fig.add_subplot(2, 3, 6)
    F_mean = U_mean - T_mean * S_mean
    ax6.plot(layers, F_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    ax6.axvline(x=21, color='#f39c12', linestyle='--', label='$L_0$')
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('Free Energy $F = U - TS$')
    ax6.set_title('(f) Free Energy Landscape')
    ax6.legend(fontsize=8)

    fig.suptitle('Phase 183: Thermodynamic Phase Portrait', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase183_phase_portrait')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"U range: [{U_mean.min():.1f}, {U_mean.max():.1f}]")
    print(f"T range: [{T_mean.min():.1f}, {T_mean.max():.1f}]")
    print(f"S range: [{S_mean.min():.1f}, {S_mean.max():.1f}]")
    print(f"F range: [{F_mean.min():.1f}, {F_mean.max():.1f}]")
    print(f"{'=' * 70}")

    save_results('phase183_phase_portrait', {
        'experiment': 'Thermodynamic Phase Portrait',
        'U_mean': [float(x) for x in U_mean],
        'T_mean': [float(x) for x in T_mean],
        'S_mean': [float(x) for x in S_mean],
        'F_mean': [float(x) for x in F_mean],
        'U_var': [float(x) for x in U_var],
        'T_var': [float(x) for x in T_var],
        'S_var': [float(x) for x in S_var],
        'n_prompts': len(PROMPTS),
    })


if __name__ == '__main__':
    main()
