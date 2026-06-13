# -*- coding: utf-8 -*-
"""
Phase 120: BKT Topological Defects
Observe "semantic vortices" in hidden state PCA space.
Measure winding numbers at each layer to detect topological defects
that annihilate at the phase transition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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


def compute_winding_number(vectors_2d):
    """Compute winding number from sequence of 2D vectors."""
    angles = np.arctan2(vectors_2d[:, 1], vectors_2d[:, 0])
    total_winding = 0
    for i in range(len(angles) - 1):
        delta = angles[i + 1] - angles[i]
        # Wrap to [-pi, pi]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        total_winding += delta
    return total_winding / (2 * np.pi)


def main():
    print("=" * 70)
    print("Phase 120: BKT Topological Defects")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Collect hidden states for all prompts at all layers
    all_hidden = []  # [prompt][layer] = vector
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = [out.hidden_states[li][0, -1, :].float().cpu().numpy() for li in range(n_layers)]
        all_hidden.append(hs)

    # At each layer, project all prompt vectors into PCA 2D space
    # and compute topological properties
    winding_numbers = []
    vortex_density = []
    angular_variance = []
    pca_explained = []

    for li in range(n_layers):
        vecs = np.array([h[li] for h in all_hidden])  # (n_prompts, d)

        # PCA to 2D
        pca = PCA(n_components=2)
        vecs_2d = pca.fit_transform(vecs)
        pca_explained.append(float(pca.explained_variance_ratio_.sum()))

        # Compute winding number (order the points by angle)
        angles = np.arctan2(vecs_2d[:, 1], vecs_2d[:, 0])
        sorted_idx = np.argsort(angles)
        sorted_vecs = vecs_2d[sorted_idx]

        wn = abs(compute_winding_number(np.vstack([sorted_vecs, sorted_vecs[0]])))
        winding_numbers.append(float(wn))

        # Angular variance (isotropy measure)
        ang_var = np.var(angles)
        angular_variance.append(float(ang_var))

        # "Vortex density" = number of angle reversals
        n_reversals = 0
        deltas = np.diff(angles[sorted_idx])
        for d in deltas:
            d_wrap = (d + np.pi) % (2 * np.pi) - np.pi
            if abs(d_wrap) > np.pi / 2:
                n_reversals += 1
        vortex_density.append(float(n_reversals / len(PROMPTS)))

    layers = np.arange(n_layers)

    # Phase transition signature: vortex density should drop at L0
    pre_vortex = np.mean(vortex_density[:int(L0)])
    post_vortex = np.mean(vortex_density[int(L0):])
    pre_winding = np.mean(winding_numbers[:int(L0)])
    post_winding = np.mean(winding_numbers[int(L0):])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Winding number profile
    axes[0, 0].plot(layers, winding_numbers, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('|Winding Number|')
    axes[0, 0].set_title('(a) Topological Winding')
    axes[0, 0].legend()

    # (b) Vortex density
    axes[0, 1].plot(layers, vortex_density, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Vortex Density')
    axes[0, 1].set_title('(b) Vortex Density')

    # (c) Angular variance
    axes[0, 2].plot(layers, angular_variance, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Angular Variance')
    axes[0, 2].set_title('(c) Angular Isotropy')

    # (d) PCA snapshots at 3 layers
    snapshot_layers = [5, int(L0), n_layers - 1]
    for i, sl in enumerate(snapshot_layers):
        vecs = np.array([h[sl] for h in all_hidden])
        pca = PCA(n_components=2)
        v2d = pca.fit_transform(vecs)
        axes[1, 0].scatter(v2d[:, 0], v2d[:, 1], s=40, alpha=0.7,
                           label=f'L{sl}')
        # Draw arrows
        for j in range(len(v2d)):
            axes[1, 0].annotate('', xy=v2d[j], xytext=(0, 0),
                               arrowprops=dict(arrowstyle='->', color=f'C{i}', alpha=0.3))
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('(d) PCA Snapshots')
    axes[1, 0].legend(fontsize=7)

    # (e) PCA explained variance
    axes[1, 1].plot(layers, pca_explained, 'o-', color='#e67e22', markersize=4)
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Explained Var (2 PCs)')
    axes[1, 1].set_title('(e) Dimensionality')

    # (f) Summary
    summary = (
        f"BKT Topological Analysis\n\n"
        f"Winding pre-L0: {pre_winding:.3f}\n"
        f"Winding post-L0: {post_winding:.3f}\n\n"
        f"Vortex pre-L0: {pre_vortex:.3f}\n"
        f"Vortex post-L0: {post_vortex:.3f}\n\n"
        f"Vortex annihilation:\n"
        f"{'YES' if post_vortex < pre_vortex * 0.7 else 'PARTIAL' if post_vortex < pre_vortex else 'NO'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 120: BKT Topological Defects',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase120_bkt')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Winding: pre={pre_winding:.3f}, post={post_winding:.3f}")
    print(f"Vortex: pre={pre_vortex:.3f}, post={post_vortex:.3f}")
    print(f"{'='*70}")

    save_results('phase120_bkt', {
        'experiment': 'BKT Topological Defects',
        'winding_numbers': winding_numbers,
        'vortex_density': vortex_density,
        'angular_variance': angular_variance,
        'summary': {
            'pre_winding': float(pre_winding),
            'post_winding': float(post_winding),
            'pre_vortex': float(pre_vortex),
            'post_vortex': float(post_vortex),
        }
    })


if __name__ == '__main__':
    main()
