# -*- coding: utf-8 -*-
"""
Phase 329: Emergent Spacetime -- Tensor Network Geometry
==========================================================
Does the transformer create an emergent spacetime?
In holography, tensor networks (MERA) give rise to AdS geometry.
Test if inter-layer correlations define a metric with negative curvature.
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


def measure_emergent_spacetime(model, tok, prompt, device):
    """Measure emergent spacetime geometry from correlations."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Build distance matrix from cosine similarity
    hiddens = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        hiddens.append(h)

    N = len(hiddens)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cos = torch.nn.functional.cosine_similarity(
                hiddens[i].unsqueeze(0), hiddens[j].unsqueeze(0)).item()
            dist_matrix[i, j] = np.arccos(np.clip(cos, -1, 1))

    # Check triangle inequality (metric space)
    violations = 0
    total_triples = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                total_triples += 1
                if (dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j] + 1e-8):
                    violations += 1
                if (dist_matrix[i, k] > dist_matrix[i, j] + dist_matrix[j, k] + 1e-8):
                    violations += 1
                if (dist_matrix[j, k] > dist_matrix[i, j] + dist_matrix[i, k] + 1e-8):
                    violations += 1

    metric_quality = 1.0 - violations / (3 * total_triples + 1e-10)

    # Curvature from distance matrix (Gromov hyperbolicity)
    # delta-hyperbolicity: max over 4-tuples
    deltas = []
    indices = list(range(min(N, 10)))  # limit for speed
    for i in indices:
        for j in indices:
            if j <= i: continue
            for k in indices:
                if k <= j: continue
                for l in indices:
                    if l <= k: continue
                    # Four-point condition
                    d_ij = dist_matrix[i, j]; d_kl = dist_matrix[k, l]
                    d_ik = dist_matrix[i, k]; d_jl = dist_matrix[j, l]
                    d_il = dist_matrix[i, l]; d_jk = dist_matrix[j, k]
                    sums = sorted([d_ij + d_kl, d_ik + d_jl, d_il + d_jk])
                    delta = (sums[2] - sums[1]) / 2
                    deltas.append(delta)

    gromov_delta = float(np.max(deltas)) if deltas else 0
    mean_delta = float(np.mean(deltas)) if deltas else 0

    # Effective dimension from distance scaling
    # d(i,j) ~ |i-j|^(1/d_eff) in d_eff dimensional space
    layer_dists = []
    layer_seps = []
    for sep in range(1, N):
        dists = [dist_matrix[i, i+sep] for i in range(N - sep)]
        layer_dists.append(float(np.mean(dists)))
        layer_seps.append(sep)

    if len(layer_seps) > 3:
        log_sep = np.log(layer_seps)
        log_dist = np.log(np.array(layer_dists) + 1e-10)
        slope, _, r, _, _ = stats.linregress(log_sep, log_dist)
        d_eff = 1.0 / (slope + 1e-10)
        scaling_r2 = r**2
    else:
        d_eff = 1.0
        scaling_r2 = 0

    return {
        'metric_quality': round(metric_quality, 4),
        'gromov_delta': round(gromov_delta, 4),
        'mean_delta': round(mean_delta, 4),
        'd_eff': round(float(d_eff), 4),
        'scaling_r2': round(float(scaling_r2), 4),
        'is_hyperbolic': gromov_delta < 0.5,
        'dist_profile': [round(d, 4) for d in layer_dists],
    }


def main():
    print("=" * 70)
    print("Phase 329: Emergent Spacetime")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        es_data = []
        for prompt in PROMPTS:
            e = measure_emergent_spacetime(model, tok, prompt, device)
            es_data.append(e)

        all_results[size] = {
            'metric_quality': round(float(np.mean([e['metric_quality'] for e in es_data])), 4),
            'gromov_delta': round(float(np.mean([e['gromov_delta'] for e in es_data])), 4),
            'mean_delta': round(float(np.mean([e['mean_delta'] for e in es_data])), 4),
            'd_eff': round(float(np.mean([e['d_eff'] for e in es_data])), 4),
            'scaling_r2': round(float(np.mean([e['scaling_r2'] for e in es_data])), 4),
            'is_hyperbolic': sum(1 for e in es_data if e['is_hyperbolic']) >= 3,
            'dist_profile': [round(float(np.mean([e['dist_profile'][i] for e in es_data])), 4)
                           for i in range(len(es_data[0]['dist_profile']))],
        }
        hyp = 'YES' if all_results[size]['is_hyperbolic'] else 'NO'
        print(f"  Metric quality: {all_results[size]['metric_quality']:.4f}")
        print(f"  Gromov delta: {all_results[size]['gromov_delta']:.4f}")
        print(f"  Hyperbolic: {hyp}")
        print(f"  d_eff: {all_results[size]['d_eff']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['dist_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Separation'); axes[0, 0].set_ylabel('Distance')
    axes[0, 0].set_title('(a) Distance vs Separation', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 1].bar(sizes, [all_results[s]['gromov_delta'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 1].axhline(0.5, color='gold', ls='--', lw=2, label='Hyp threshold')
    axes[0, 1].set_ylabel('delta'); axes[0, 1].set_title('(b) Gromov Hyperbolicity', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].bar(sizes, [all_results[s]['d_eff'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('d_eff'); axes[0, 2].set_title('(c) Effective Dimension', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['metric_quality'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Quality'); axes[1, 0].set_title('(d) Metric Quality', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "EMERGENT SPACETIME\n\n"
    for s in sizes:
        d = all_results[s]
        hyp = 'YES' if d['is_hyperbolic'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  delta = {d['gromov_delta']:.3f}\n"
        txt += f"  d_eff = {d['d_eff']:.2f}\n"
        txt += f"  hyp: {hyp}\n\n"
    txt += "delta<0.5: hyperbolic\n"
    txt += "(AdS-like geometry)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 329: Emergent Spacetime -- Tensor Network Geometry",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase329_spacetime')
    plt.close()
    save_results('phase329_spacetime', {'experiment': 'Emergent Spacetime', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
