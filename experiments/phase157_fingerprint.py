# -*- coding: utf-8 -*-
"""
Phase 157: Thermodynamic Fingerprinting
Each prompt produces a unique thermodynamic trajectory.
Can we use this trajectory as a "fingerprint" to distinguish
between prompts? Measure trajectory distances and clustering.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from utils import load_model, save_results, save_figure

PROMPTS = {
    'math1': "The integral of x squared equals",
    'math2': "The derivative of sin x is",
    'math3': "The eigenvalues of a matrix are",
    'phys1': "Black holes emit Hawking radiation because",
    'phys2': "The speed of light is constant in",
    'phys3': "Quantum entanglement connects distant particles",
    'bio1': "DNA stores genetic information using four",
    'bio2': "Mitochondria produce energy through cellular",
    'bio3': "Evolution occurs through natural selection of",
    'lang1': "Shakespeare wrote many famous plays including",
    'lang2': "The capital of Japan is Tokyo which",
    'lang3': "English is the most widely spoken language",
}


def main():
    print("=" * 70)
    print("Phase 157: Thermodynamic Fingerprinting")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect thermodynamic trajectories
    trajectories = {}
    for name, prompt in PROMPTS.items():
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_traj = []
        kT_traj = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_traj.append(S if not np.isnan(S) else 0)

            top_k = 50
            top_probs = torch.topk(probs, top_k).values
            log_probs = torch.log(top_probs + 1e-10).cpu().numpy()
            ranks = np.arange(1, top_k + 1, dtype=np.float64)
            if np.std(log_probs) > 0.01:
                slope = np.polyfit(ranks, log_probs, 1)[0]
                kT = -1.0 / (slope + 1e-10)
            else:
                kT = 0.1
            kT = max(0.01, min(kT, 50))
            kT_traj.append(float(kT))

        trajectories[name] = {
            'S': np.array(S_traj),
            'kT': np.array(kT_traj),
            'combined': np.concatenate([S_traj, kT_traj]),  # Full fingerprint
        }

    # Compute pairwise distances between fingerprints
    names = list(trajectories.keys())
    fingerprints = np.array([trajectories[n]['combined'] for n in names])
    dist_matrix = squareform(pdist(fingerprints, metric='euclidean'))

    # Categories
    categories = {'math': ['math1', 'math2', 'math3'],
                  'phys': ['phys1', 'phys2', 'phys3'],
                  'bio': ['bio1', 'bio2', 'bio3'],
                  'lang': ['lang1', 'lang2', 'lang3']}
    cat_map = {}
    for cat, members in categories.items():
        for m in members:
            cat_map[m] = cat

    # Within-category vs between-category distances
    within = []
    between = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                if cat_map[ni] == cat_map[nj]:
                    within.append(dist_matrix[i, j])
                else:
                    between.append(dist_matrix[i, j])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_cat = {'math': '#c0392b', 'phys': '#2980b9', 'bio': '#27ae60', 'lang': '#f39c12'}

    # (a) S trajectories colored by category
    for name, traj in trajectories.items():
        cat = cat_map[name]
        axes[0,0].plot(range(n_layers), traj['S'], '-', color=colors_cat[cat],
                      linewidth=1.5, alpha=0.7)
    # Legend
    for cat, color in colors_cat.items():
        axes[0,0].plot([], [], '-', color=color, linewidth=2, label=cat)
    axes[0,0].axvline(x=21.7, color='gray', linewidth=1, linestyle='--')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$S$')
    axes[0,0].set_title('(a) Entropy Fingerprints')
    axes[0,0].legend()

    # (b) Distance matrix
    im = axes[0,1].imshow(dist_matrix, cmap='viridis', aspect='auto')
    axes[0,1].set_xticks(range(len(names)))
    axes[0,1].set_xticklabels(names, fontsize=6, rotation=45)
    axes[0,1].set_yticks(range(len(names)))
    axes[0,1].set_yticklabels(names, fontsize=6)
    plt.colorbar(im, ax=axes[0,1], label='Distance')
    axes[0,1].set_title('(b) Fingerprint Distance Matrix')

    # (c) Dendrogram
    Z = linkage(pdist(fingerprints), method='ward')
    dendrogram(Z, labels=names, ax=axes[0,2], leaf_font_size=7)
    axes[0,2].set_title('(c) Hierarchical Clustering')

    # (d) Within vs between category distances
    axes[1,0].hist(within, bins=8, alpha=0.6, color='#27ae60',
                  label=f'Within ({np.mean(within):.2f})', edgecolor='black')
    axes[1,0].hist(between, bins=8, alpha=0.6, color='#c0392b',
                  label=f'Between ({np.mean(between):.2f})', edgecolor='black')
    axes[1,0].set_xlabel('Distance')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('(d) Within vs Between Category')
    axes[1,0].legend()

    # (e) 2D PCA projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(fingerprints)
    for i, name in enumerate(names):
        cat = cat_map[name]
        axes[1,1].scatter(proj[i, 0], proj[i, 1], c=colors_cat[cat], s=100,
                         edgecolors='black', zorder=5)
        axes[1,1].annotate(name, (proj[i, 0], proj[i, 1]),
                          xytext=(3, 3), textcoords='offset points', fontsize=7)
    axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    axes[1,1].set_title('(e) PCA of Fingerprints')

    # (f) Summary
    sep_ratio = np.mean(between) / (np.mean(within) + 1e-10)
    summary = (
        f"Thermodynamic Fingerprinting\n\n"
        f"12 prompts, 4 categories\n"
        f"Fingerprint: S + kT ({n_layers*2}D)\n\n"
        f"Within-cat distance: {np.mean(within):.2f}\n"
        f"Between-cat distance: {np.mean(between):.2f}\n"
        f"Separation ratio: {sep_ratio:.2f}x\n\n"
        f"Categories are\n"
        f"{'SEPARABLE' if sep_ratio > 1.2 else 'OVERLAPPING'}\n"
        f"in thermodynamic space"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 157: Thermodynamic Fingerprinting',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase157_fingerprint')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Separation ratio: {sep_ratio:.2f}x")
    print(f"Within: {np.mean(within):.2f}, Between: {np.mean(between):.2f}")
    print(f"{'='*70}")

    save_results('phase157_fingerprint', {
        'experiment': 'Thermodynamic Fingerprinting',
        'summary': {
            'separation_ratio': float(sep_ratio),
            'within_mean': float(np.mean(within)),
            'between_mean': float(np.mean(between)),
        }
    })


if __name__ == '__main__':
    main()
