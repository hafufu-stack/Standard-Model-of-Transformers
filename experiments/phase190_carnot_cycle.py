# -*- coding: utf-8 -*-
"""
Phase 190: Carnot Cycle Decomposition
========================================
The transformer processes information through a thermodynamic cycle.
If the Carnot analogy is exact, the T-S diagram should show 4 strokes:
  1. Isothermal expansion (T~const, S increases) - early layers absorb info
  2. Adiabatic expansion (S~const, T drops) - deep processing
  3. Isothermal compression (T~const, S decreases) - information crystallization
  4. Adiabatic compression (S~const, T rises) - final refinement

The area enclosed in T-S space = net thermodynamic work.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
    "Cryptographic hash functions ensure data integrity",
    "DNA stores hereditary genetic information",
    "Entropy measures disorder in physical systems",
    "Superconductors carry current with zero resistance",
    "Artificial neural networks are inspired by biological neurons",
]


def classify_stroke(dT, dS, T_tol=0.05, S_tol=0.05):
    """Classify each layer transition into a thermodynamic stroke type."""
    T_const = abs(dT) < T_tol * abs(dT + dS + 1e-10)
    S_const = abs(dS) < S_tol * abs(dT + dS + 1e-10)

    if T_const and not S_const:
        return 'isothermal' if dS > 0 else 'isothermal_comp'
    elif S_const and not T_const:
        return 'adiabatic' if dT < 0 else 'adiabatic_comp'
    elif dS > 0 and dT < 0:
        return 'expansion'
    elif dS < 0 and dT > 0:
        return 'compression'
    elif dS > 0 and dT > 0:
        return 'heating'
    else:
        return 'cooling'


def main():
    print("=" * 70)
    print("Phase 190: Carnot Cycle Decomposition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_T, all_S, all_U = [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals, S_vals, U_vals = [], [], []
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
            T_vals.append(T if not np.isnan(T) else 0)
            S_vals.append(S if not np.isnan(S) else 0)
            U_vals.append(U if not np.isnan(U) else 0)

        all_T.append(T_vals)
        all_S.append(S_vals)
        all_U.append(U_vals)

    T_mean = np.mean(all_T, axis=0)
    S_mean = np.mean(all_S, axis=0)
    U_mean = np.mean(all_U, axis=0)

    # Classify strokes
    stroke_types = []
    for i in range(n_layers - 1):
        dT = T_mean[i+1] - T_mean[i]
        dS = S_mean[i+1] - S_mean[i]
        stroke_types.append(classify_stroke(dT, dS))

    # Compute work = integral of T*dS along path
    W_segments = []
    for i in range(n_layers - 1):
        T_avg = (T_mean[i] + T_mean[i+1]) / 2
        dS = S_mean[i+1] - S_mean[i]
        W_segments.append(T_avg * dS)
    W_net = sum(W_segments)

    # T-S area (approximate using shoelace formula for open curve)
    Q_hot = sum(w for w in W_segments if w > 0)
    Q_cold = sum(-w for w in W_segments if w < 0)
    eta_carnot = W_net / (Q_hot + 1e-10) if Q_hot > 0 else 0

    # Stroke colors
    stroke_colors = {
        'isothermal': '#e74c3c', 'isothermal_comp': '#3498db',
        'adiabatic': '#f39c12', 'adiabatic_comp': '#2ecc71',
        'expansion': '#e67e22', 'compression': '#2980b9',
        'heating': '#d35400', 'cooling': '#1abc9c',
    }

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) T-S diagram with arrows (THE key plot)
    T_norm = (T_mean - T_mean.min()) / (T_mean.max() - T_mean.min() + 1e-10)
    S_norm = (S_mean - S_mean.min()) / (S_mean.max() - S_mean.min() + 1e-10)
    for i in range(n_layers - 1):
        color = stroke_colors.get(stroke_types[i], '#999999')
        axes[0, 0].annotate('', xy=(S_norm[i+1], T_norm[i+1]), xytext=(S_norm[i], T_norm[i]),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2))
    axes[0, 0].scatter(S_norm, T_norm, c=np.arange(n_layers), cmap='viridis', s=40,
                        edgecolors='black', zorder=5)
    axes[0, 0].scatter([S_norm[0]], [T_norm[0]], s=150, marker='s', c='green',
                        edgecolors='black', zorder=10, label='Layer 0')
    axes[0, 0].scatter([S_norm[-1]], [T_norm[-1]], s=150, marker='*', c='red',
                        edgecolors='black', zorder=10, label=f'Layer {n_layers-1}')
    axes[0, 0].scatter([S_norm[L0]], [T_norm[L0]], s=150, marker='D', c='#f39c12',
                        edgecolors='black', zorder=10, label=f'$L_0$={L0}')
    axes[0, 0].set_xlabel('Hidden Entropy $S$ (normalized)')
    axes[0, 0].set_ylabel('Temperature $T$ (normalized)')
    axes[0, 0].set_title('(a) T-S Diagram')
    axes[0, 0].legend(fontsize=7)

    # (b) T-S with raw values + individual prompts
    for pi in range(min(5, len(all_T))):
        axes[0, 1].plot(all_S[pi], all_T[pi], '-', alpha=0.3, linewidth=1)
    axes[0, 1].plot(S_mean, T_mean, 'ko-', markersize=4, linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('Hidden Entropy $S$ (nats)')
    axes[0, 1].set_ylabel('Temperature $T$ (nats)')
    axes[0, 1].set_title('(b) T-S Diagram (raw)')
    axes[0, 1].legend(fontsize=8)

    # (c) Stroke classification timeline
    stroke_nums = {'isothermal': 1, 'isothermal_comp': -1, 'adiabatic': 2, 'adiabatic_comp': -2,
                   'expansion': 3, 'compression': -3, 'heating': 4, 'cooling': -4}
    stroke_y = [stroke_nums.get(s, 0) for s in stroke_types]
    stroke_c = [stroke_colors.get(s, '#999') for s in stroke_types]
    axes[0, 2].bar(np.arange(n_layers-1) + 0.5, stroke_y, color=stroke_c, edgecolor='black', width=0.8)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Stroke Type')
    axes[0, 2].set_title('(c) Stroke Timeline')
    axes[0, 2].legend(fontsize=8)

    # (d) Work per layer (T*dS)
    colors_w = ['#e74c3c' if w > 0 else '#3498db' for w in W_segments]
    axes[1, 0].bar(np.arange(n_layers-1) + 0.5, W_segments, color=colors_w,
                   edgecolor='black', alpha=0.7, width=0.8)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$W = T \\cdot dS$')
    axes[1, 0].set_title('(d) Work per Layer (red=absorbed, blue=released)')

    # (e) P-V analog: U vs S diagram
    axes[1, 1].plot(S_mean, U_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1, 1].scatter([S_mean[0]], [U_mean[0]], s=150, marker='s', c='green',
                        edgecolors='black', zorder=10)
    axes[1, 1].scatter([S_mean[-1]], [U_mean[-1]], s=150, marker='*', c='red',
                        edgecolors='black', zorder=10)
    axes[1, 1].set_xlabel('Hidden Entropy $S$')
    axes[1, 1].set_ylabel('Energy $U$')
    axes[1, 1].set_title('(e) U-S Diagram (P-V analog)')

    # (f) Summary
    from collections import Counter
    stroke_counts = Counter(stroke_types)
    summary = (
        f"Carnot Cycle Decomposition\n\n"
        f"Q_hot (absorbed): {Q_hot:.2f}\n"
        f"Q_cold (released): {Q_cold:.2f}\n"
        f"W_net: {W_net:.2f}\n"
        f"eta = W/Q_hot: {eta_carnot:.4f}\n\n"
        f"Stroke counts:\n"
    )
    for s, c in stroke_counts.most_common():
        summary += f"  {s}: {c}\n"

    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 190: Carnot Cycle Decomposition', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase190_carnot_cycle')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Q_hot={Q_hot:.2f}, Q_cold={Q_cold:.2f}, W_net={W_net:.2f}")
    print(f"eta_carnot = {eta_carnot:.4f}")
    print(f"Strokes: {dict(stroke_counts)}")
    print(f"{'=' * 70}")

    save_results('phase190_carnot_cycle', {
        'experiment': 'Carnot Cycle Decomposition',
        'T_mean': [float(x) for x in T_mean],
        'S_mean': [float(x) for x in S_mean],
        'U_mean': [float(x) for x in U_mean],
        'W_segments': [float(x) for x in W_segments],
        'stroke_types': stroke_types,
        'summary': {
            'Q_hot': float(Q_hot), 'Q_cold': float(Q_cold),
            'W_net': float(W_net), 'eta_carnot': float(eta_carnot),
        }
    })


if __name__ == '__main__':
    main()
