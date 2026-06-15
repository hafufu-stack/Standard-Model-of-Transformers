# -*- coding: utf-8 -*-
"""
Phase 199: Eigenvalue Spectrum & Random Matrix Theory
=======================================================
Random Matrix Theory (RMT) predicts universal spectral statistics.
If the weight matrices follow Marchenko-Pastur or Wigner semicircle
laws, the network has "random" structure. Deviations = learned structure.

Compare the eigenvalue spectrum of each layer's weight matrix to RMT
predictions. Excess eigenvalues beyond the Marchenko-Pastur edge
represent the "semantic information" encoded in weights.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def marchenko_pastur_edge(sigma, gamma):
    """Marchenko-Pastur upper edge: (sigma * (1 + sqrt(gamma)))^2"""
    return (sigma * (1 + np.sqrt(gamma))) ** 2


def main():
    print("=" * 70)
    print("Phase 199: Eigenvalue Spectrum & RMT")
    print("=" * 70)

    device = 'cpu'  # Need CPU for eigenvalue computation
    model, tok = load_model(device=device)
    n_transformer_layers = len(model.model.layers)
    L0 = 21

    # Analyze weight matrices at sampled layers
    sample_layers = [0, 4, 8, 12, 16, 20, 24, 27]
    sample_layers = [l for l in sample_layers if l < n_transformer_layers]

    mp_ratios = []  # Ratio of eigenvalues beyond MP edge
    spectral_data = {}

    for li in sample_layers:
        layer = model.model.layers[li]
        # Use the MLP down_proj weight as representative
        W = None
        for name in ['down_proj', 'gate_proj', 'up_proj']:
            if hasattr(layer.mlp, name):
                W = getattr(layer.mlp, name).weight.data.float().cpu().numpy()
                break
        if W is None:
            continue

        m, n = W.shape
        gamma = m / n if n > 0 else 1

        # Compute singular values (more stable than eigenvalues)
        # Eigenvalues of W^T W
        s = np.linalg.svd(W, compute_uv=False)
        eigenvalues = s ** 2 / n  # Normalize

        # Marchenko-Pastur prediction
        sigma_sq = np.var(W)
        mp_edge = marchenko_pastur_edge(np.sqrt(sigma_sq), gamma)

        # Fraction beyond MP edge (excess = learned info)
        n_beyond = np.sum(eigenvalues > mp_edge)
        ratio = n_beyond / len(eigenvalues)
        mp_ratios.append((li, ratio, mp_edge, len(eigenvalues)))

        spectral_data[li] = {
            'eigenvalues': eigenvalues[:100],  # Top 100
            'mp_edge': mp_edge,
            'sigma_sq': sigma_sq,
            'gamma': gamma,
            'beyond_ratio': ratio,
        }

        print(f"  Layer {li}: {n_beyond}/{len(eigenvalues)} beyond MP edge "
              f"({ratio:.1%}), gamma={gamma:.2f}")

    # === Visualization ===
    n_plots = min(len(sample_layers), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    plot_layers = sample_layers[:6]

    for idx, li in enumerate(plot_layers):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        if li in spectral_data:
            data = spectral_data[li]
            eigs = data['eigenvalues']
            mp_edge = data['mp_edge']

            # Histogram of eigenvalues
            ax.hist(eigs, bins=40, density=True, alpha=0.7,
                    color='#3498db' if li < L0 else '#e74c3c',
                    edgecolor='black', label='Empirical')
            ax.axvline(x=mp_edge, color='#f39c12', linewidth=2, linestyle='--',
                       label=f'MP edge={mp_edge:.4f}')

            ax.set_xlabel('Eigenvalue')
            ax.set_ylabel('Density')
            ax.set_title(f'Layer {li} ({data["beyond_ratio"]:.1%} beyond MP)')
            ax.legend(fontsize=7)

    fig.suptitle('Phase 199: Eigenvalue Spectrum vs Random Matrix Theory',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase199_rmt')
    plt.close()

    # Summary
    pre_L0 = [r for l, r, _, _ in mp_ratios if l < L0]
    post_L0 = [r for l, r, _, _ in mp_ratios if l >= L0]
    mean_pre = np.mean(pre_L0) if pre_L0 else 0
    mean_post = np.mean(post_L0) if post_L0 else 0

    print(f"\n{'=' * 70}")
    print(f"Pre-L0 beyond-MP ratio: {mean_pre:.3f}")
    print(f"Post-L0 beyond-MP ratio: {mean_post:.3f}")
    print(f"{'=' * 70}")

    save_results('phase199_rmt', {
        'experiment': 'Eigenvalue Spectrum & RMT',
        'mp_ratios': [(int(l), float(r)) for l, r, _, _ in mp_ratios],
        'summary': {
            'mean_pre_L0': float(mean_pre),
            'mean_post_L0': float(mean_post),
        }
    })


if __name__ == '__main__':
    main()
