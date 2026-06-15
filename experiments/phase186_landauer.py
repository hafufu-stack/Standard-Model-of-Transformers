# -*- coding: utf-8 -*-
"""
Phase 186: Landauer's Erasure Bound
=======================================
Landauer's principle: erasing 1 bit costs at minimum kT*ln(2) energy.
Token prediction erases ~17 bits (vocab 151K -> 1 token).

KEY QUESTION: Does the energy dissipated per layer match or exceed
              the Landauer bound for the information erased?

If yes, transformers are thermodynamically optimal information processors.
The "Semantic Friction" from Phase 174 would be exactly the excess
dissipation above Landauer's minimum.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
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
]


def main():
    print("=" * 70)
    print("Phase 186: Landauer's Erasure Bound")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_S = []       # Shannon entropy of token distribution per layer
    all_U = []       # Energy (L2 norm) per layer
    all_dI = []      # Information erased per transition
    all_dU = []      # Energy change per transition
    all_landauer = [] # Landauer bound per transition
    all_ratio = []   # Actual/Landauer ratio

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_vals = []
        U_vals = []

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()
            U_vals.append(U if not np.isnan(U) else 0)

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_vals.append(S if not np.isnan(S) else 0)

        all_S.append(S_vals)
        all_U.append(U_vals)

        # Per-transition analysis
        dI_vals = []
        dU_vals = []
        land_vals = []
        ratio_vals = []

        for i in range(n_layers - 1):
            dS = S_vals[i + 1] - S_vals[i]  # Entropy change
            dI = max(0, -dS)  # Information erased (entropy decrease = erasure)
            dU = abs(U_vals[i + 1] - U_vals[i])  # Energy dissipated

            # Effective temperature at this layer (use token entropy as T)
            T_eff = (S_vals[i] + S_vals[i + 1]) / 2 + 1e-10

            # Landauer bound: minimum energy = T * dI (in natural units, nats)
            landauer_min = T_eff * dI

            # Ratio: how far above Landauer bound?
            ratio = dU / (landauer_min + 1e-10) if dI > 0.01 else float('nan')

            dI_vals.append(dI)
            dU_vals.append(dU)
            land_vals.append(landauer_min)
            ratio_vals.append(ratio)

        all_dI.append(dI_vals)
        all_dU.append(dU_vals)
        all_landauer.append(land_vals)
        all_ratio.append(ratio_vals)

    S_mean = np.mean(all_S, axis=0)
    U_mean = np.mean(all_U, axis=0)
    dI_mean = np.mean(all_dI, axis=0)
    dU_mean = np.mean(all_dU, axis=0)
    land_mean = np.mean(all_landauer, axis=0)

    # Compute mean ratio only where erasure happens (dI > threshold)
    ratio_arr = np.array(all_ratio)
    ratio_valid = []
    for li in range(n_layers - 1):
        col = ratio_arr[:, li]
        valid = col[~np.isnan(col)]
        ratio_valid.append(np.mean(valid) if len(valid) > 0 else float('nan'))
    ratio_valid = np.array(ratio_valid)

    layers_t = np.arange(n_layers - 1) + 0.5

    # Total information erased and total energy dissipated
    total_I = np.sum(dI_mean)
    total_dU = np.sum(dU_mean)
    total_landauer = np.sum(land_mean)
    overall_ratio = total_dU / (total_landauer + 1e-10)

    # Layers where Landauer is violated (ratio < 1 AND significant erasure)
    erasure_layers = [i for i in range(n_layers - 1) if dI_mean[i] > 0.1]
    violated = [i for i in erasure_layers if not np.isnan(ratio_valid[i]) and ratio_valid[i] < 1.0]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Shannon entropy profile
    axes[0, 0].plot(np.arange(n_layers), S_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Shannon Entropy $S$ (nats)')
    axes[0, 0].set_title('(a) Token Distribution Entropy')
    axes[0, 0].legend(fontsize=8)

    # (b) Information erased per layer
    colors_b = ['#e74c3c' if di > 0.1 else '#bdc3c7' for di in dI_mean]
    axes[0, 1].bar(layers_t, dI_mean, color=colors_b, edgecolor='black', width=0.8, alpha=0.7)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Information Erased $\\Delta I$ (nats)')
    axes[0, 1].set_title('(b) Landauer Erasure per Layer')

    # (c) Actual energy vs Landauer bound
    axes[0, 2].plot(layers_t, dU_mean, 'o-', color='#c0392b', markersize=4, linewidth=2,
                    label='Actual $|\\Delta U|$')
    axes[0, 2].plot(layers_t, land_mean, 's-', color='#2980b9', markersize=4, linewidth=2,
                    label='Landauer min $T \\cdot \\Delta I$')
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Energy')
    axes[0, 2].set_title('(c) Actual vs Landauer Bound')
    axes[0, 2].legend(fontsize=8)

    # (d) Landauer ratio
    valid_mask = ~np.isnan(ratio_valid)
    r_plot = np.where(valid_mask, ratio_valid, 0)
    colors_d = ['#2ecc71' if r >= 1 else '#e74c3c' for r in r_plot]
    axes[1, 0].bar(layers_t, r_plot, color=colors_d, edgecolor='black', width=0.8, alpha=0.7)
    axes[1, 0].axhline(y=1, color='black', linestyle='--', linewidth=2, label='Landauer bound')
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$|\\Delta U| / (T \\cdot \\Delta I)$')
    axes[1, 0].set_title('(d) Landauer Ratio (green=satisfied)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(0, min(10, max(r_plot) * 1.2) if max(r_plot) > 0 else 10)

    # (e) Scatter: dU vs T*dI with Landauer line
    for pi in range(min(5, len(all_dI))):
        valid_pts = [(all_landauer[pi][i], all_dU[pi][i])
                     for i in range(n_layers - 1) if all_dI[pi][i] > 0.1]
        if valid_pts:
            x, y = zip(*valid_pts)
            axes[1, 1].scatter(x, y, s=20, alpha=0.3)
    lim = max(max(land_mean) * 2, max(dU_mean) * 1.2)
    axes[1, 1].plot([0, lim], [0, lim], 'k--', linewidth=2, label='Landauer bound')
    axes[1, 1].set_xlabel('$T \\cdot \\Delta I$ (Landauer minimum)')
    axes[1, 1].set_ylabel('$|\\Delta U|$ (Actual)')
    axes[1, 1].set_title('(e) Energy vs Landauer Minimum')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = (
        f"Landauer's Erasure Bound\n\n"
        f"Total info erased: {total_I:.2f} nats\n"
        f"  = {total_I / np.log(2):.1f} bits\n\n"
        f"Total energy: {total_dU:.2f}\n"
        f"Landauer min: {total_landauer:.2f}\n"
        f"Overall ratio: {overall_ratio:.2f}x\n\n"
        f"Violations: {len(violated)}/{len(erasure_layers)}\n"
        f"  erasure layers\n\n"
        f"Landauer {'SATISFIED' if len(violated) <= 2 else 'VIOLATED'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 186: Landauer's Erasure Bound", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase186_landauer')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Total info erased: {total_I:.2f} nats ({total_I / np.log(2):.1f} bits)")
    print(f"Total energy dissipated: {total_dU:.2f}")
    print(f"Landauer minimum: {total_landauer:.2f}")
    print(f"Ratio: {overall_ratio:.2f}x above Landauer")
    print(f"Violations: {len(violated)}/{len(erasure_layers)} erasure layers")
    print(f"{'=' * 70}")

    save_results('phase186_landauer', {
        'experiment': "Landauer's Erasure Bound",
        'S_mean': [float(x) for x in S_mean],
        'dI_mean': [float(x) for x in dI_mean],
        'dU_mean': [float(x) for x in dU_mean],
        'landauer_min_mean': [float(x) for x in land_mean],
        'ratio': [float(x) if not np.isnan(x) else None for x in ratio_valid],
        'summary': {
            'total_info_erased_nats': float(total_I),
            'total_info_erased_bits': float(total_I / np.log(2)),
            'total_energy': float(total_dU),
            'total_landauer_min': float(total_landauer),
            'overall_ratio': float(overall_ratio),
            'n_violations': len(violated),
            'n_erasure_layers': len(erasure_layers),
        }
    })


if __name__ == '__main__':
    main()
