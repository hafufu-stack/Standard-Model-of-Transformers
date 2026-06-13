# -*- coding: utf-8 -*-
"""
Phase 115: Correlation Length Divergence
Near a phase transition, the correlation length xi diverges.
Measure the "correlation" between hidden states at distance d apart
and see if it peaks at L0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
]


def main():
    print("=" * 70)
    print("Phase 115: Correlation Length Divergence")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Collect hidden states
    all_hidden = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers)]
        all_hidden.append(hs)

    # Compute correlation function C(l, d) = cos_sim(h_l, h_{l+d})
    max_d = 10
    # For each layer l, compute average C(l, d) for d=1..max_d
    C_matrix = np.zeros((n_layers, max_d + 1))

    for d in range(1, max_d + 1):
        for l in range(n_layers - d):
            cos_sims = []
            for hs in all_hidden:
                cos = torch.nn.functional.cosine_similarity(
                    hs[l].unsqueeze(0), hs[l + d].unsqueeze(0)).item()
                cos_sims.append(cos)
            C_matrix[l, d] = np.mean(cos_sims)

    # Correlation length at each layer: xi(l) = distance d at which C drops below threshold
    threshold = 0.8
    xi = []
    for l in range(n_layers):
        corr_length = 0
        for d in range(1, max_d + 1):
            if l + d < n_layers and C_matrix[l, d] > threshold:
                corr_length = d
            else:
                break
        xi.append(corr_length)

    # Also compute "connected correlation" G(l, d) = C(l, d) - C(l, inf)
    # Use C(l, max_d) as C(l, inf)
    G_matrix = np.zeros_like(C_matrix)
    for l in range(n_layers):
        c_inf = C_matrix[l, max_d] if l + max_d < n_layers else C_matrix[l, -1]
        for d in range(1, max_d + 1):
            G_matrix[l, d] = C_matrix[l, d] - c_inf

    # "Susceptibility" = sum of G(l, d) over d
    chi_corr = []
    for l in range(n_layers):
        chi_corr.append(np.sum(G_matrix[l, 1:]))

    # Find xi peak
    xi_arr = np.array(xi)
    xi_peak = np.argmax(xi_arr)

    layers = np.arange(n_layers)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Correlation function C(l, d=1)
    C1 = C_matrix[:, 1]
    axes[0, 0].plot(layers[:-1], C1[:-1], 'o-', color='#c0392b', markersize=4)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0, 0].set_xlabel('Layer $l$')
    axes[0, 0].set_ylabel('$C(l, d{=}1)$')
    axes[0, 0].set_title('(a) Nearest-Neighbor Correlation')
    axes[0, 0].legend()

    # (b) Correlation length profile
    axes[0, 1].plot(layers, xi_arr, 'o-', color='#8e44ad', markersize=5, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].scatter([xi_peak], [xi_arr[xi_peak]], s=150, marker='*',
                       color='#f39c12', zorder=10, label=f'Peak L{xi_peak}')
    axes[0, 1].set_xlabel('Layer $l$')
    axes[0, 1].set_ylabel('$\\xi$ (correlation length)')
    axes[0, 1].set_title(f'(b) Correlation Length (peak L{xi_peak})')
    axes[0, 1].legend()

    # (c) C(l, d) heatmap for selected layers
    selected = [5, 10, 15, 20, 25]
    for sl in selected:
        if sl < n_layers:
            axes[0, 2].plot(range(1, max_d + 1), C_matrix[sl, 1:], 'o-',
                           markersize=3, label=f'L{sl}')
    axes[0, 2].set_xlabel('Distance $d$')
    axes[0, 2].set_ylabel('$C(l, d)$')
    axes[0, 2].set_title('(c) Correlation Decay')
    axes[0, 2].legend(fontsize=7)

    # (d) Connected susceptibility
    axes[1, 0].plot(layers, chi_corr, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    chi_peak = np.argmax(chi_corr)
    axes[1, 0].scatter([chi_peak], [chi_corr[chi_peak]], s=150, marker='*',
                       color='#f39c12', zorder=10, label=f'Peak L{chi_peak}')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$\\chi_{corr}$')
    axes[1, 0].set_title(f'(d) Connected Susceptibility (peak L{chi_peak})')
    axes[1, 0].legend()

    # (e) Heatmap of C_matrix
    im = axes[1, 1].imshow(C_matrix[:, 1:].T, aspect='auto', cmap='inferno',
                            origin='lower', extent=[0, n_layers-1, 1, max_d])
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Layer $l$')
    axes[1, 1].set_ylabel('Distance $d$')
    axes[1, 1].set_title('(e) Correlation Heatmap')
    plt.colorbar(im, ax=axes[1, 1], shrink=0.7)

    # (f) Summary
    pre_xi = np.mean(xi_arr[:int(L0)])
    post_xi = np.mean(xi_arr[int(L0):])
    summary = (
        f"Correlation Length Analysis\n\n"
        f"xi peak: L{xi_peak}\n"
        f"chi peak: L{chi_peak}\n\n"
        f"xi pre-L0: {pre_xi:.2f}\n"
        f"xi post-L0: {post_xi:.2f}\n\n"
        f"xi at L0: {xi_arr[int(L0)]}\n"
        f"Max xi: {xi_arr.max()}\n\n"
        f"Divergence at L0: {'YES' if abs(xi_peak - L0) <= 3 else 'NO'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 115: Correlation Length ($\\xi$ peak L{xi_peak})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase115_correlation_length')
    plt.close()

    print(f"\n{'='*70}")
    print(f"xi peak: L{xi_peak}, chi peak: L{chi_peak}")
    print(f"xi pre-L0: {pre_xi:.2f}, post: {post_xi:.2f}")
    print(f"{'='*70}")

    save_results('phase115_correlation_length', {
        'experiment': 'Correlation Length Divergence',
        'xi': [int(v) for v in xi],
        'chi_corr': [float(v) for v in chi_corr],
        'C1': [float(v) for v in C1],
        'summary': {
            'xi_peak': int(xi_peak),
            'chi_peak': int(chi_peak),
            'pre_xi': float(pre_xi),
            'post_xi': float(post_xi),
        }
    })


if __name__ == '__main__':
    main()
