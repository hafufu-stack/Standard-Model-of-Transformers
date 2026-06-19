# -*- coding: utf-8 -*-
"""
Phase 313: Fisher Information Metric -- Information Geometry
=============================================================
The Fisher information metric defines a Riemannian geometry on the
space of probability distributions.
g_ij(theta) = E[d log p / d theta_i * d log p / d theta_j]
For transformers, compute the Fisher metric on hidden state space
to understand the curvature of information geometry.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def compute_fisher_metric(model, tok, prompt, device):
    """Compute Fisher information metric across layers."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    layer_fisher_trace = []
    layer_curvature = []
    layer_geodesic_dist = []

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0].float()  # (seq, D)

        # Convert to probability distribution (softmax-like)
        h_last = h[-1, :]  # last token
        p = torch.softmax(h_last, dim=0)

        # Fisher information: F = E[grad(log p)^2]
        # For discrete distribution: F_ii = 1/p_i
        # Fisher trace = sum(1/p_i) for p_i > threshold
        mask = p > 1e-8
        fisher_trace = float((1.0 / (p[mask])).sum().item())
        layer_fisher_trace.append(fisher_trace)

        # Information curvature from singular values
        _, s, _ = torch.linalg.svd(h, full_matrices=False)
        s = s.cpu().numpy()
        s_norm = s / (s[0] + 1e-10)

        # Curvature ~ how fast singular values decay
        if len(s_norm) > 1:
            log_s = np.log(s_norm[:10] + 1e-15)
            log_n = np.log(np.arange(1, len(log_s) + 1, dtype=float))
            slope, _, _, _, _ = stats.linregress(log_n, log_s)
            curvature = -slope  # higher = more curved
        else:
            curvature = 0
        layer_curvature.append(float(curvature))

    # Geodesic distances between consecutive layers
    for li in range(n_layers):
        h1 = out.hidden_states[li][0, -1, :].float()
        h2 = out.hidden_states[li + 1][0, -1, :].float()

        # Fisher-Rao distance between distributions
        p1 = torch.softmax(h1, dim=0)
        p2 = torch.softmax(h2, dim=0)

        # Bhattacharyya distance (related to Fisher-Rao)
        bc = float(torch.sqrt(p1 * p2).sum().item())
        d_fr = float(-np.log(bc + 1e-15))
        layer_geodesic_dist.append(d_fr)

    # Scalar curvature: sum of sectional curvatures
    R_scalar = float(np.sum(layer_curvature))

    return {
        'fisher_trace': [round(f, 2) for f in layer_fisher_trace],
        'curvature': [round(c, 4) for c in layer_curvature],
        'geodesic_dist': [round(d, 4) for d in layer_geodesic_dist],
        'R_scalar': round(R_scalar, 4),
        'mean_curvature': round(float(np.mean(layer_curvature)), 4),
        'total_geodesic': round(float(np.sum(layer_geodesic_dist)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 313: Fisher Information Metric")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        fi_data = []
        for prompt in PROMPTS:
            fi = compute_fisher_metric(model, tok, prompt, device)
            fi_data.append(fi)

        n = len(fi_data[0]['curvature'])
        avg_curv = [float(np.mean([f['curvature'][i] for f in fi_data])) for i in range(n)]
        nd = len(fi_data[0]['geodesic_dist'])
        avg_geo = [float(np.mean([f['geodesic_dist'][i] for f in fi_data])) for i in range(nd)]

        all_results[size] = {
            'n_layers': n - 1,
            'avg_curvature': [round(c, 4) for c in avg_curv],
            'avg_geodesic_dist': [round(d, 4) for d in avg_geo],
            'mean_curvature': round(float(np.mean(avg_curv)), 4),
            'total_geodesic': round(float(np.sum(avg_geo)), 4),
            'R_scalar': round(float(np.mean([f['R_scalar'] for f in fi_data])), 4),
        }
        print(f"  Mean curvature: {all_results[size]['mean_curvature']:.4f}")
        print(f"  Total geodesic: {all_results[size]['total_geodesic']:.4f}")
        print(f"  R (scalar): {all_results[size]['R_scalar']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_curvature'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Curvature')
    axes[0, 0].set_title('(a) Information Curvature', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_geodesic_dist'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Geodesic Distance')
    axes[0, 1].set_title('(b) Fisher-Rao Distance', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['R_scalar'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('R'); axes[0, 2].set_title('(c) Scalar Curvature', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # Normalized depth
    for size, data in all_results.items():
        n = len(data['avg_curvature'])
        x = np.linspace(0, 1, n)
        axes[1, 0].plot(x, data['avg_curvature'], '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Normalized Depth'); axes[1, 0].set_ylabel('Curvature')
    axes[1, 0].set_title('(d) Curvature vs Depth', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')

    txt = "INFORMATION GEOMETRY\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  R = {d['R_scalar']:.2f}\n"
        txt += f"  Geodesic = {d['total_geodesic']:.2f}\n\n"
    txt += "R > 0: positive curvature\n"
    txt += "R < 0: negative (hyperbolic)\n"
    txt += "R = 0: flat (Euclidean)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 313: Fisher Information Metric -- Information Geometry",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase313_fisher')
    plt.close()
    save_results('phase313_fisher', {'experiment': 'Fisher Information Metric', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
