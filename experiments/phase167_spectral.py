# -*- coding: utf-8 -*-
"""
Phase 167: Spectral Gap and Phase Transition
Measure the spectral gap (ratio of top-2 eigenvalues) of the
hidden state covariance matrix at each layer.
In statistical physics, the spectral gap closes at the critical point.
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
    "DNA encodes the instructions for all living organisms",
    "Thermodynamics governs the flow of energy and entropy",
    "The brain processes information through neural circuits",
    "Climate change is driven by greenhouse gas emissions",
]


def main():
    print("=" * 70)
    print("Phase 167: Spectral Gap Analysis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect hidden states across prompts at each layer
    # Then compute covariance matrix and its eigenvalues
    all_spectral_gap = [[] for _ in range(n_layers)]
    all_top_eig = [[] for _ in range(n_layers)]
    all_participation_ratio = [[] for _ in range(n_layers)]
    all_rank = [[] for _ in range(n_layers)]

    # Process each prompt
    hidden_per_layer = [[] for _ in range(n_layers)]

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(n_layers):
            h = out.hidden_states[li][0, -1, :].float().cpu()
            hidden_per_layer[li].append(h)

    # Compute spectral properties
    for li in range(n_layers):
        H = torch.stack(hidden_per_layer[li])  # (n_prompts, hidden_dim)

        # Center the data
        H_centered = H - H.mean(dim=0, keepdim=True)

        # Covariance matrix (use SVD for efficiency)
        try:
            U, S_vals, Vh = torch.linalg.svd(H_centered, full_matrices=False)
            eigenvalues = (S_vals ** 2) / (len(PROMPTS) - 1)
            eigenvalues = eigenvalues.numpy()

            # Spectral gap: ratio of 2nd to 1st eigenvalue
            if len(eigenvalues) >= 2 and eigenvalues[0] > 1e-10:
                gap = 1.0 - eigenvalues[1] / eigenvalues[0]
            else:
                gap = 0

            # Top eigenvalue (variance explained by PC1)
            total_var = eigenvalues.sum()
            pc1_frac = eigenvalues[0] / (total_var + 1e-10)

            # Participation ratio of eigenvalues
            if total_var > 1e-10:
                p = eigenvalues / total_var
                pr = 1.0 / (np.sum(p**2) + 1e-10)
            else:
                pr = 1

            # Effective rank
            if total_var > 1e-10:
                p_nonzero = eigenvalues[eigenvalues > 1e-10] / total_var
                eff_rank = np.exp(-np.sum(p_nonzero * np.log(p_nonzero + 1e-10)))
            else:
                eff_rank = 1

        except:
            gap = 0
            pc1_frac = 0
            pr = 1
            eff_rank = 1

        all_spectral_gap[li].append(gap)
        all_top_eig[li].append(pc1_frac)
        all_participation_ratio[li].append(pr)
        all_rank[li].append(eff_rank)

    avg_gap = [np.mean(v) if v else 0 for v in all_spectral_gap]
    avg_pc1 = [np.mean(v) if v else 0 for v in all_top_eig]
    avg_pr = [np.mean(v) if v else 0 for v in all_participation_ratio]
    avg_rank = [np.mean(v) if v else 0 for v in all_rank]

    layers = np.arange(n_layers)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Spectral gap
    axes[0,0].plot(layers, avg_gap, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Spectral Gap (1 - $\\lambda_2/\\lambda_1$)')
    axes[0,0].set_title('(a) Spectral Gap')
    axes[0,0].legend()

    # (b) PC1 fraction
    axes[0,1].plot(layers, avg_pc1, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('PC1 Variance Fraction')
    axes[0,1].set_title('(b) Dominance of PC1')

    # (c) Effective rank
    axes[0,2].plot(layers, avg_rank, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Effective Rank')
    axes[0,2].set_title('(c) Representation Dimensionality')

    # (d) Participation ratio
    axes[1,0].plot(layers, avg_pr, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Participation Ratio')
    axes[1,0].set_title('(d) Eigenvalue Distribution')

    # (e) Spectral gap gradient
    dg = np.gradient(avg_gap)
    dg_colors = ['#c0392b' if d < 0 else '#2980b9' for d in dg]
    axes[1,1].bar(layers, dg, color=dg_colors, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].axhline(y=0, color='black', linewidth=0.5)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('$d(gap)/dL$')
    axes[1,1].set_title('(e) Gap Gradient')

    # (f) Summary
    min_gap_layer = np.argmin(avg_gap[2:]) + 2
    pre_gap = np.mean(avg_gap[:20])
    post_gap = np.mean(avg_gap[20:])
    pre_rank = np.mean(avg_rank[:20])
    post_rank = np.mean(avg_rank[20:])
    summary = (
        f"Spectral Gap Analysis\n\n"
        f"Min gap layer: L{min_gap_layer}\n"
        f"  (gap={avg_gap[min_gap_layer]:.3f})\n\n"
        f"Pre-L0 gap: {pre_gap:.3f}\n"
        f"Post-L0 gap: {post_gap:.3f}\n\n"
        f"Pre-L0 rank: {pre_rank:.1f}\n"
        f"Post-L0 rank: {post_rank:.1f}\n\n"
        f"Gap {'CLOSES' if post_gap < pre_gap else 'OPENS'}\n"
        f"near phase transition"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 167: Spectral Gap and Phase Transition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase167_spectral')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Min gap: L{min_gap_layer} ({avg_gap[min_gap_layer]:.3f})")
    print(f"Pre-L0 gap: {pre_gap:.3f}, Post-L0 gap: {post_gap:.3f}")
    print(f"Pre-L0 rank: {pre_rank:.1f}, Post-L0 rank: {post_rank:.1f}")
    print(f"{'='*70}")

    save_results('phase167_spectral', {
        'experiment': 'Spectral Gap Analysis',
        'summary': {
            'min_gap_layer': int(min_gap_layer),
            'pre_gap': float(pre_gap),
            'post_gap': float(post_gap),
            'pre_rank': float(pre_rank),
            'post_rank': float(post_rank),
        }
    })


if __name__ == '__main__':
    main()
