# -*- coding: utf-8 -*-
"""
Phase 226: Dissipative Structure Detection
=============================================
Prigogine's theory: non-equilibrium systems can form organized
"dissipative structures" — stable patterns that exist only because
of continuous energy flow.

Test: Do transformer layers form stable thermodynamic patterns
(attractors in T-U-S space) that require continuous information flow?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
    "Superconductors have zero electrical resistance",
    "The brain contains billions of neurons",
    "Algorithms determine computational complexity",
    "Chemical reactions follow conservation of mass",
    "Stars form from collapsing molecular clouds",
]


def detect_structures(model, tok, device, model_name):
    """Detect dissipative structures in layer thermodynamics."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Collect multi-prompt profiles
    all_states = []  # [prompt][layer] = (T, U, S, P1)
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        states = []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = h.norm().item()
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            P1 = probs.max().item()
            T = S if not np.isnan(S) else 0
            states.append([T, U, S, P1])
        all_states.append(states)

    n = min(len(s) for s in all_states)

    # Mean and std of each variable at each layer
    var_names = ['T', 'U', 'S', 'P1']
    mean_profiles = {v: [] for v in var_names}
    std_profiles = {v: [] for v in var_names}
    for l in range(n):
        for vi, v in enumerate(var_names):
            vals = [all_states[p][l][vi] for p in range(len(PROMPTS))]
            mean_profiles[v].append(float(np.mean(vals)))
            std_profiles[v].append(float(np.std(vals)))

    # Dissipative structure detection:
    # 1. Coefficient of variation (CV) at each layer
    cv_profiles = {}
    for v in var_names:
        cv_profiles[v] = [std_profiles[v][l] / (abs(mean_profiles[v][l]) + 1e-10)
                         for l in range(n)]

    # 2. Clustering: do layers naturally cluster in (T,U,S) space?
    layer_points = np.array([[mean_profiles['T'][l], mean_profiles['U'][l],
                              mean_profiles['S'][l]] for l in range(n)])
    dist_matrix = squareform(pdist(layer_points))

    # 3. Phase identification via k-means
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(layer_points) > 2:
        Z = linkage(layer_points, method='ward')
        # Try 2 and 3 clusters
        labels_2 = fcluster(Z, t=2, criterion='maxclust')
        labels_3 = fcluster(Z, t=3, criterion='maxclust')

        # Silhouette-like score (simple)
        def cluster_quality(labels, points):
            from scipy.spatial.distance import cdist
            k = len(set(labels))
            if k <= 1:
                return 0
            intra = 0
            inter = 0
            for c in set(labels):
                mask = labels == c
                if mask.sum() > 1:
                    intra += np.mean(pdist(points[mask]))
                for c2 in set(labels):
                    if c2 > c:
                        mask2 = labels == c2
                        inter += np.mean(cdist(points[mask], points[mask2]))
            return float(inter / (intra + 1e-10))

        q2 = cluster_quality(labels_2, layer_points)
        q3 = cluster_quality(labels_3, layer_points)
    else:
        labels_2 = np.array([1] * n)
        labels_3 = np.array([1] * n)
        q2, q3 = 0, 0

    # 4. Stability: how stable are layer-level patterns across prompts?
    # Cross-prompt correlation of layer profiles
    prompt_corrs = []
    for i in range(len(PROMPTS)):
        for j in range(i+1, len(PROMPTS)):
            ti = [all_states[i][l][0] for l in range(n)]
            tj = [all_states[j][l][0] for l in range(n)]
            r, _ = np.corrcoef(ti, tj)[0, 1], 0
            if not np.isnan(r):
                prompt_corrs.append(r)
    mean_prompt_corr = float(np.mean(prompt_corrs)) if prompt_corrs else 0

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_profiles': mean_profiles,
        'std_profiles': std_profiles,
        'cv_profiles': cv_profiles,
        'cluster_labels_2': labels_2.tolist(),
        'cluster_labels_3': labels_3.tolist(),
        'cluster_quality_2': q2,
        'cluster_quality_3': q3,
        'mean_prompt_corr': mean_prompt_corr,
    }


def main():
    print("=" * 70)
    print("Phase 226: Dissipative Structure Detection")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = detect_structures(model, tok, device, size)
        results[size] = r
        print(f"  Cluster quality: k=2 -> {r['cluster_quality_2']:.3f}, k=3 -> {r['cluster_quality_3']:.3f}")
        print(f"  Prompt correlation: {r['mean_prompt_corr']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) T profile with cluster coloring
    for si, (size, r) in enumerate(results.items()):
        labels = r['cluster_labels_3']
        T = r['mean_profiles']['T']
        cluster_colors = ['#e74c3c', '#3498db', '#2ecc71']
        for l in range(len(T)):
            axes[0, 0].scatter(l, T[l], c=cluster_colors[labels[l]-1], s=30, alpha=0.7)
        axes[0, 0].plot(range(len(T)), T, '-', color=colors[size], alpha=0.3, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) T Profile (3 clusters)')
    axes[0, 0].legend(fontsize=8)

    # (b) CV profiles
    for size, r in results.items():
        for v in ['T', 'P1']:
            axes[0, 1].plot(range(len(r['cv_profiles'][v])), r['cv_profiles'][v],
                           '-' if v == 'T' else '--', color=colors[size], lw=1.5,
                           label=f'{size} {v}', alpha=0.8)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('CV')
    axes[0, 1].set_title('(b) Coefficient of Variation')
    axes[0, 1].legend(fontsize=7)

    # (c) U-T phase diagram with clusters
    for size, r in results.items():
        labels = r['cluster_labels_3']
        U = r['mean_profiles']['U']
        T = r['mean_profiles']['T']
        cluster_colors_map = ['#e74c3c', '#3498db', '#2ecc71']
        for l in range(len(U)):
            axes[0, 2].scatter(U[l], T[l], c=cluster_colors_map[labels[l]-1], s=40, alpha=0.7)
        axes[0, 2].plot(U, T, '-', color=colors[size], alpha=0.3)
    axes[0, 2].set_xlabel('U'); axes[0, 2].set_ylabel('T')
    axes[0, 2].set_title('(c) Phase Diagram + Clusters')

    # (d) Std profiles
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['std_profiles']['T'])), r['std_profiles']['T'],
                       '-', color=colors[size], lw=2, label=f'{size} T')
        axes[1, 0].plot(range(len(r['std_profiles']['P1'])), r['std_profiles']['P1'],
                       '--', color=colors[size], lw=1.5, alpha=0.6, label=f'{size} P1')
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('Std')
    axes[1, 0].set_title('(d) Cross-Prompt Variability')
    axes[1, 0].legend(fontsize=7)

    # (e) P1 with clusters
    for size, r in results.items():
        axes[1, 1].plot(range(len(r['mean_profiles']['P1'])), r['mean_profiles']['P1'],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('P1')
    axes[1, 1].set_title('(e) Order Parameter P1')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Dissipative Structures\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  k=2 quality = {r['cluster_quality_2']:.3f}\n"
        summary += f"  k=3 quality = {r['cluster_quality_3']:.3f}\n"
        summary += f"  Prompt corr = {r['mean_prompt_corr']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 226: Dissipative Structure Detection", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase226_dissipative')
    plt.close()
    save_results('phase226_dissipative', {'experiment': 'Dissipative Structures', 'results': results})


if __name__ == '__main__':
    main()
