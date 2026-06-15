# -*- coding: utf-8 -*-
"""
Phase 187: Information Geodesics
===================================
Each layer maps the token probability distribution p_l on a simplex.
The Fisher-Rao metric gives this simplex a natural Riemannian geometry,
where geodesics correspond to the "path of least statistical action."

KEY QUESTION: Does the transformer follow geodesics through
              information space, or does it take detours?

Geodesic efficiency = (direct Hellinger distance) / (path length)
If ~1, the model is a geodesic computer: minimum-action information processor.
If <<1, the detours may be the computational work itself.
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
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "DNA stores hereditary genetic information",
    "Entropy measures disorder in physical systems",
    "The Turing machine is a fundamental model of computation",
    "Superconductors carry current with zero resistance",
    "The human genome contains approximately three billion base pairs",
    "Artificial neural networks are inspired by biological neurons",
]


def hellinger_distance(p, q):
    """Hellinger distance: H(p,q) = sqrt(1 - sum(sqrt(p*q)))."""
    bc = torch.sum(torch.sqrt(p * q + 1e-20)).item()
    bc = min(bc, 1.0)
    return np.sqrt(max(0, 1 - bc))


def kl_divergence(p, q):
    """KL(p||q) = sum(p * log(p/q))."""
    return (p * torch.log((p + 1e-10) / (q + 1e-10))).sum().item()


def jensen_shannon(p, q):
    """Jensen-Shannon divergence (symmetric)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def main():
    print("=" * 70)
    print("Phase 187: Information Geodesics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_hell = []       # Hellinger distances between adjacent layers
    all_js = []         # Jensen-Shannon divergences
    all_path_len = []   # Cumulative path length
    all_geodesic = []   # Direct geodesic (first to last)
    all_efficiency = [] # Geodesic efficiency

    # Per-layer curvature
    all_curvature = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get probability distributions at each layer
        probs_all = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            probs_all.append(probs)

        # Hellinger and JS distances between adjacent layers
        hell_vals = []
        js_vals = []
        for i in range(n_layers - 1):
            h = hellinger_distance(probs_all[i], probs_all[i + 1])
            j = jensen_shannon(probs_all[i], probs_all[i + 1])
            hell_vals.append(h)
            js_vals.append(j if not np.isnan(j) else 0)

        all_hell.append(hell_vals)
        all_js.append(js_vals)

        # Path length (sum of Hellinger distances)
        path_length = sum(hell_vals)
        all_path_len.append(path_length)

        # Direct geodesic (first layer to last)
        geodesic = hellinger_distance(probs_all[0], probs_all[-1])
        all_geodesic.append(geodesic)

        # Geodesic efficiency
        efficiency = geodesic / (path_length + 1e-10)
        all_efficiency.append(efficiency)

        # Curvature: deviation from straight line at each point
        # kappa = |d2x/ds2| ~ angle between consecutive steps
        curvature = []
        for i in range(1, n_layers - 1):
            # Use triangle inequality deviation as curvature proxy
            d_prev = hell_vals[i - 1]
            d_next = hell_vals[i]  # Fixed: use i not i-1
            d_skip = hellinger_distance(probs_all[i - 1], probs_all[i + 1])
            # Curvature ~ how much shorter the direct path is
            kappa = (d_prev + d_next - d_skip) / (d_prev + d_next + 1e-10)
            curvature.append(kappa if not np.isnan(kappa) else 0)
        all_curvature.append(curvature)

    hell_mean = np.mean(all_hell, axis=0)
    js_mean = np.mean(all_js, axis=0)
    curv_mean = np.mean(all_curvature, axis=0)
    eff_mean = np.mean(all_efficiency)
    eff_std = np.std(all_efficiency)
    path_mean = np.mean(all_path_len)
    geo_mean = np.mean(all_geodesic)

    layers_t = np.arange(n_layers - 1) + 0.5
    layers_c = np.arange(1, n_layers - 1)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Hellinger distance profile
    axes[0, 0].plot(layers_t, hell_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Hellinger Distance')
    axes[0, 0].set_title('(a) Step Size in Information Space')
    axes[0, 0].legend(fontsize=8)

    # (b) Cumulative path length vs geodesic
    cum_path = np.cumsum(hell_mean)
    axes[0, 1].plot(layers_t, cum_path, 'o-', color='#c0392b', markersize=3, linewidth=2,
                    label='Actual path')
    axes[0, 1].axhline(y=geo_mean, color='#2980b9', linestyle='--', linewidth=2,
                        label=f'Geodesic ({geo_mean:.3f})')
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Cumulative Hellinger Distance')
    axes[0, 1].set_title(f'(b) Path vs Geodesic (eff={eff_mean:.3f})')
    axes[0, 1].legend(fontsize=8)

    # (c) Jensen-Shannon divergence
    axes[0, 2].plot(layers_t, js_mean, 's-', color='#2ecc71', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Jensen-Shannon Divergence')
    axes[0, 2].set_title('(c) Information Divergence per Step')

    # (d) Curvature profile
    colors_d = ['#e74c3c' if c > np.mean(curv_mean) else '#3498db' for c in curv_mean]
    axes[1, 0].bar(layers_c, curv_mean, color=colors_d, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Curvature $\\kappa$')
    axes[1, 0].set_title('(d) Path Curvature (red=high)')

    # (e) Efficiency per prompt
    axes[1, 1].hist(all_efficiency, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=eff_mean, color='#e74c3c', linewidth=2, linestyle='--',
                        label=f'Mean={eff_mean:.3f}')
    axes[1, 1].set_xlabel('Geodesic Efficiency')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('(e) Efficiency Distribution')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    # Pre-L0 vs post-L0 analysis
    pre_hell = np.mean(hell_mean[:L0])
    post_hell = np.mean(hell_mean[L0:])
    pre_curv = np.mean(curv_mean[:L0-1])
    post_curv = np.mean(curv_mean[L0-1:])

    summary = (
        f"Information Geodesics\n\n"
        f"Geodesic efficiency: {eff_mean:.3f} +/- {eff_std:.3f}\n"
        f"  (1.0 = perfect geodesic)\n\n"
        f"Path length: {path_mean:.3f}\n"
        f"Geodesic: {geo_mean:.3f}\n"
        f"Detour ratio: {path_mean/geo_mean:.2f}x\n\n"
        f"Pre-L0 step: {pre_hell:.4f}\n"
        f"Post-L0 step: {post_hell:.4f}\n\n"
        f"Pre-L0 curvature: {pre_curv:.4f}\n"
        f"Post-L0 curvature: {post_curv:.4f}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 187: Information Geodesics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase187_geodesics')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Geodesic efficiency: {eff_mean:.3f} +/- {eff_std:.3f}")
    print(f"Path length: {path_mean:.3f}, Geodesic: {geo_mean:.3f}")
    print(f"Detour ratio: {path_mean/geo_mean:.2f}x")
    print(f"Pre-L0 step: {pre_hell:.4f}, Post-L0: {post_hell:.4f}")
    print(f"{'=' * 70}")

    save_results('phase187_geodesics', {
        'experiment': 'Information Geodesics',
        'hellinger_mean': [float(x) for x in hell_mean],
        'js_mean': [float(x) for x in js_mean],
        'curvature_mean': [float(x) for x in curv_mean],
        'summary': {
            'geodesic_efficiency_mean': float(eff_mean),
            'geodesic_efficiency_std': float(eff_std),
            'path_length_mean': float(path_mean),
            'geodesic_mean': float(geo_mean),
            'detour_ratio': float(path_mean / geo_mean),
            'pre_L0_step': float(pre_hell),
            'post_L0_step': float(post_hell),
            'pre_L0_curvature': float(pre_curv),
            'post_L0_curvature': float(post_curv),
        }
    })


if __name__ == '__main__':
    main()
