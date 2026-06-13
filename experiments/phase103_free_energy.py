# -*- coding: utf-8 -*-
"""
Phase 103: Free Energy Landscape at Eta Transition
Compute F = U - TS at each layer. If the free energy shows a distinctive
minimum shift at L0, it confirms the thermodynamic nature of the transition.
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
    "The human genome encodes three billion base pairs",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Climate change affects global ecosystems",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "The cosmic microwave background reveals the early universe",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
]


def main():
    print("=" * 70)
    print("Phase 103: Free Energy Landscape")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect U, T, S at every layer for every prompt
    results = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        Us = []
        Ts = []
        Ss = []
        PRs = []

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()
            Us.append(U)

            # Participation ratio
            h_sq = h**2
            pr = (h_sq.sum()**2 / (h_sq**2).sum()).item()
            PRs.append(pr)

            # Temperature = entropy of output distribution
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0.0
            Ts.append(T)

            # Entropy of hidden state
            h_abs = h.abs()
            p_h = h_abs / (h_abs.sum() + 1e-10)
            S = -(p_h * torch.log(p_h + 1e-10)).sum().item()
            if np.isnan(S):
                S = 0.0
            Ss.append(S)

        # Free energy F = U - T*S
        Fs = [U - T * S for U, T, S in zip(Us, Ts, Ss)]

        results.append({
            'U': Us, 'T': Ts, 'S': Ss, 'F': Fs, 'PR': PRs,
        })

    # Average across prompts
    avg_U = np.mean([r['U'] for r in results], axis=0)
    avg_T = np.mean([r['T'] for r in results], axis=0)
    avg_S = np.mean([r['S'] for r in results], axis=0)
    avg_F = np.mean([r['F'] for r in results], axis=0)
    avg_PR = np.mean([r['PR'] for r in results], axis=0)

    # Derivatives (phase transition indicators)
    dF_dL = np.gradient(avg_F)
    d2F_dL2 = np.gradient(dF_dL)
    dU_dL = np.gradient(avg_U)
    dS_dL = np.gradient(avg_S)

    # Find free energy minimum
    F_min_layer = np.argmin(avg_F[2:]) + 2  # skip first 2
    # Find inflection point of F (d2F/dL2 = 0)
    sign_changes = []
    for i in range(1, len(d2F_dL2)):
        if d2F_dL2[i] * d2F_dL2[i-1] < 0:
            sign_changes.append(i)

    L0 = 21.7  # From Phase 97

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers = np.arange(n_layers)

    # (a) Free energy landscape
    axes[0,0].plot(layers, avg_F, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0={L0:.0f}$')
    axes[0,0].scatter([F_min_layer], [avg_F[F_min_layer]], s=150, marker='*',
                      color='#f39c12', zorder=10, label=f'Min: L{F_min_layer}')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$F = U - TS$')
    axes[0,0].set_title('(a) Free Energy Landscape')
    axes[0,0].legend(fontsize=8)

    # (b) U, TS decomposition
    axes[0,1].plot(layers, avg_U, 'o-', color='#c0392b', markersize=3, label='$U$ (energy)')
    axes[0,1].plot(layers, [t*s for t,s in zip(avg_T, avg_S)], 's-', color='#2980b9',
                   markersize=3, label='$TS$ (entropy)')
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', alpha=0.5)
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Value')
    axes[0,1].set_title('(b) Energy-Entropy Decomposition')
    axes[0,1].legend(fontsize=8)

    # (c) dF/dL (first derivative)
    colors_df = ['#27ae60' if v < 0 else '#c0392b' for v in dF_dL]
    axes[0,2].bar(layers, dF_dL, color=colors_df, alpha=0.7, edgecolor='black')
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=0, color='black', linewidth=0.5)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$dF/dL$')
    axes[0,2].set_title('(c) Free Energy Gradient')

    # (d) Temperature profile
    axes[1,0].plot(layers, avg_T, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$T$ (entropy)')
    axes[1,0].set_title('(d) Temperature Profile')

    # (e) Phase diagram: U vs T
    scatter = axes[1,1].scatter(avg_T, avg_U, c=layers, cmap='coolwarm', s=60,
                                edgecolors='black', zorder=5)
    for i in [0, 5, 10, 15, 20, 25, n_layers-1]:
        if i < n_layers:
            axes[1,1].annotate(f'L{i}', (avg_T[i], avg_U[i]),
                              textcoords="offset points", xytext=(5,5), fontsize=7)
    cb = plt.colorbar(scatter, ax=axes[1,1], shrink=0.7)
    cb.set_label('Layer')
    axes[1,1].set_xlabel('$T$ (entropy)')
    axes[1,1].set_ylabel('$U$ (energy)')
    axes[1,1].set_title('(e) Phase Diagram ($U$ vs $T$)')

    # (f) Summary
    # Ratio of energetic vs entropic driving
    pre_dF = np.mean(dF_dL[:int(L0)])
    post_dF = np.mean(dF_dL[int(L0):])
    summary = (
        f"Free Energy Analysis\n\n"
        f"F minimum: L{F_min_layer}\n"
        f"dF/dL (pre-L0): {pre_dF:.2f}\n"
        f"dF/dL (post-L0): {post_dF:.2f}\n\n"
        f"Inflection points: {sign_changes}\n\n"
        f"Pre-transition: {'energy-driven' if pre_dF > 0 else 'entropy-driven'}\n"
        f"Post-transition: {'energy-driven' if post_dF > 0 else 'entropy-driven'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 103: Free Energy Landscape (F min at L{F_min_layer})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase103_free_energy')
    plt.close()

    print(f"\n{'='*70}")
    print(f"F minimum: L{F_min_layer}")
    print(f"dF/dL pre-transition: {pre_dF:.2f}")
    print(f"dF/dL post-transition: {post_dF:.2f}")
    print(f"Inflection points: {sign_changes}")
    print(f"{'='*70}")

    save_results('phase103_free_energy', {
        'experiment': 'Free Energy Landscape',
        'averages': {
            'U': [float(v) for v in avg_U],
            'T': [float(v) for v in avg_T],
            'S': [float(v) for v in avg_S],
            'F': [float(v) for v in avg_F],
            'PR': [float(v) for v in avg_PR],
        },
        'summary': {
            'F_min_layer': int(F_min_layer),
            'pre_dF_mean': float(pre_dF),
            'post_dF_mean': float(post_dF),
            'inflection_points': [int(s) for s in sign_changes],
        }
    })


if __name__ == '__main__':
    main()
