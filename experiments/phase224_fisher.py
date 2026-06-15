# -*- coding: utf-8 -*-
"""
Phase 224: Fisher Information Metric
=======================================
Compute the Fisher information matrix at each layer.
The Fisher metric defines the natural geometry of the
probability manifold that the transformer traverses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
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


def compute_fisher(model, tok, device, model_name):
    """Compute Fisher information at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    K = 200  # Top-K tokens for Fisher computation

    all_fisher_trace = []
    all_fisher_det = []
    all_geodesic_dist = []
    all_T = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        fisher_trace_l, fisher_det_l, geo_l, T_l = [], [], [], []
        prev_probs = None

        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)

            # Top-K probabilities for tractable computation
            topk = probs.topk(K)
            p_topk = topk.values.cpu().numpy()
            p_topk = p_topk / (p_topk.sum() + 1e-10)  # Re-normalize

            # Fisher information: g_ij = sum_x (1/p(x)) * dp_i/dx * dp_j/dx
            # For categorical: F = diag(1/p) - trace is sum(1/p)
            fisher_trace = float(np.sum(1.0 / (p_topk + 1e-10)))
            fisher_trace_l.append(fisher_trace)

            # Fisher "determinant" proxy: product of 1/p for top-k
            log_det = float(np.sum(np.log(1.0 / (p_topk + 1e-10))))
            fisher_det_l.append(log_det)

            # Geodesic distance (Bhattacharyya) from previous layer
            if prev_probs is not None:
                # Hellinger-like geodesic: 2 * arccos(sum(sqrt(p*q)))
                overlap = np.sum(np.sqrt(p_topk * prev_probs + 1e-20))
                geo_dist = 2 * np.arccos(min(overlap, 1.0))
                geo_l.append(float(geo_dist))
            else:
                geo_l.append(0)
            prev_probs = p_topk.copy()

            T = -(probs.cpu().numpy() * np.log(probs.cpu().numpy() + 1e-10)).sum()
            T_l.append(float(T) if not np.isnan(T) else 0)

        all_fisher_trace.append(fisher_trace_l)
        all_fisher_det.append(fisher_det_l)
        all_geodesic_dist.append(geo_l)
        all_T.append(T_l)

    n = min(len(f) for f in all_fisher_trace)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]

    mean_fisher = avg(all_fisher_trace)
    mean_det = avg(all_fisher_det)
    mean_geo = avg(all_geodesic_dist)
    mean_T = avg(all_T)

    # Total geodesic length
    total_geodesic = sum(mean_geo)

    # Curvature proxy: d(geodesic_distance)/dl
    dgeo = [mean_geo[i+1] - mean_geo[i] for i in range(n-1)]

    # Fisher-temperature correlation
    from scipy.stats import pearsonr
    r_fisher_T, p_fisher_T = pearsonr(mean_fisher, mean_T)

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_fisher_trace': mean_fisher,
        'mean_log_det': mean_det,
        'mean_geodesic': mean_geo,
        'mean_T': mean_T,
        'total_geodesic': total_geodesic,
        'dgeo': [float(x) for x in dgeo],
        'r_fisher_T': float(r_fisher_T),
        'p_fisher_T': float(p_fisher_T),
    }


def main():
    print("=" * 70)
    print("Phase 224: Fisher Information Metric")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = compute_fisher(model, tok, device, size)
        results[size] = r
        print(f"  Total geodesic = {r['total_geodesic']:.4f}")
        print(f"  Fisher-T correlation: r={r['r_fisher_T']:.4f}, p={r['p_fisher_T']:.2e}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, r in results.items():
        c = colors[size]
        axes[0, 0].plot(range(len(r['mean_fisher_trace'])), r['mean_fisher_trace'],
                       '-', color=c, lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Tr(F)')
    axes[0, 0].set_title('(a) Fisher Information Trace')
    axes[0, 0].legend(fontsize=8)

    for size, r in results.items():
        axes[0, 1].plot(range(len(r['mean_geodesic'])), r['mean_geodesic'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Geodesic Distance')
    axes[0, 1].set_title('(b) Geodesic Step Size')
    axes[0, 1].legend(fontsize=8)

    for size, r in results.items():
        cum_geo = np.cumsum(r['mean_geodesic']).tolist()
        axes[0, 2].plot(range(len(cum_geo)), cum_geo, '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Cumulative Distance')
    axes[0, 2].set_title('(c) Geodesic Path Length')
    axes[0, 2].legend(fontsize=8)

    for size, r in results.items():
        axes[1, 0].scatter(r['mean_fisher_trace'], r['mean_T'], c=range(len(r['mean_T'])),
                          cmap='viridis', s=30, alpha=0.7)
        axes[1, 0].plot(r['mean_fisher_trace'], r['mean_T'], '-', color=colors[size], alpha=0.3)
    axes[1, 0].set_xlabel('Tr(F)'); axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title('(d) Fisher vs Temperature')

    for size, r in results.items():
        axes[1, 1].plot(range(len(r['mean_log_det'])), r['mean_log_det'],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('log det(F)')
    axes[1, 1].set_title('(e) Fisher Log-Determinant')
    axes[1, 1].legend(fontsize=8)

    summary = "Fisher Information\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Geodesic = {r['total_geodesic']:.3f}\n"
        summary += f"  r(F,T)   = {r['r_fisher_T']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 224: Fisher Information Metric", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase224_fisher')
    plt.close()
    save_results('phase224_fisher', {'experiment': 'Fisher Information', 'results': results})


if __name__ == '__main__':
    main()
