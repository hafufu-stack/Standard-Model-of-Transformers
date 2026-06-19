# -*- coding: utf-8 -*-
"""
Phase 342: Quantum Complexity -- Circuit Depth
=====================================================
Quantum complexity measures the minimum circuit depth to prepare
a state. Test whether the Transformer's hidden states have
increasing complexity with layer depth, and whether complexity
growth is linear (Lloyd's bound).
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


def measure_complexity(model, tok, prompt, device):
    """Measure state complexity at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    h0 = hiddens[0]  # Reference state

    complexities = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Complexity proxy 1: geodesic distance from reference state
        # C = ||h(l) - h(0)|| (accumulated distance)
        dist = float(torch.norm(h - h0).item())

        # Complexity proxy 2: number of non-trivial components
        # (participation ratio of the difference vector)
        diff = h - h0
        if float(torch.norm(diff).item()) > 1e-10:
            p = (diff**2) / (torch.sum(diff**2) + 1e-30)
            pr = float(1.0 / (torch.sum(p**2).item() + 1e-30))
        else:
            pr = 0

        # Complexity proxy 3: SVD entropy of cumulative transformation
        complexities.append({
            'distance': round(float(dist), 4),
            'participation': round(float(pr), 2),
        })

    distances = [c['distance'] for c in complexities]
    participations = [c['participation'] for c in complexities]

    # Lloyd's bound: dC/dt <= 2*E/pi*hbar
    # Test linear growth: C(l) = a*l + b
    layers = np.arange(len(distances))
    slope, intercept, r, _, _ = stats.linregress(layers, distances)
    r2_linear = r**2

    # Test: C(l) = a*sqrt(l) (diffusive)
    sqrt_layers = np.sqrt(layers + 1)
    slope_sqrt, _, r_sqrt, _, _ = stats.linregress(sqrt_layers, distances)
    r2_sqrt = r_sqrt**2

    # Switchback time: does complexity saturate?
    if len(distances) > 5:
        late_slope = (distances[-1] - distances[-5]) / 5
        early_slope = (distances[5] - distances[0]) / 5 if distances[5] > distances[0] else 1
        saturation_ratio = late_slope / (early_slope + 1e-10)
    else:
        saturation_ratio = 1.0

    return {
        'distances': distances,
        'participations': participations,
        'r2_linear': round(float(r2_linear), 4),
        'r2_sqrt': round(float(r2_sqrt), 4),
        'growth_type': 'linear' if r2_linear > r2_sqrt else 'diffusive',
        'slope': round(float(slope), 4),
        'saturation_ratio': round(float(saturation_ratio), 4),
        'max_complexity': round(float(max(distances)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 342: Quantum Complexity")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        comp_data = []
        for prompt in PROMPTS:
            c = measure_complexity(model, tok, prompt, device)
            comp_data.append(c)

        n = len(comp_data[0]['distances'])
        linear_count = sum(1 for c in comp_data if c['growth_type'] == 'linear')
        all_results[size] = {
            'distances': [round(float(np.mean([c['distances'][i] for c in comp_data])), 4)
                         for i in range(n)],
            'participations': [round(float(np.mean([c['participations'][i] for c in comp_data])), 2)
                              for i in range(n)],
            'r2_linear': round(float(np.mean([c['r2_linear'] for c in comp_data])), 4),
            'r2_sqrt': round(float(np.mean([c['r2_sqrt'] for c in comp_data])), 4),
            'growth_type': 'linear' if linear_count >= 4 else 'diffusive',
            'slope': round(float(np.mean([c['slope'] for c in comp_data])), 4),
            'saturation_ratio': round(float(np.mean([c['saturation_ratio'] for c in comp_data])), 4),
            'max_complexity': round(float(np.mean([c['max_complexity'] for c in comp_data])), 4),
        }
        print(f"  Growth: {all_results[size]['growth_type']}")
        print(f"  R2 linear: {all_results[size]['r2_linear']:.4f}")
        print(f"  R2 sqrt: {all_results[size]['r2_sqrt']:.4f}")
        print(f"  Max C: {all_results[size]['max_complexity']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['distances'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Complexity')
    axes[0, 0].set_title('(a) Complexity Growth', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['participations'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Participation Ratio')
    axes[0, 1].set_title('(b) State Participation', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 2].bar(x - w/2, [all_results[s]['r2_linear'] for s in sizes], w,
                  label='Linear', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['r2_sqrt'] for s in sizes], w,
                  label='Diffusive', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_title('(c) Growth Fit', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(sizes, [all_results[s]['saturation_ratio'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].axhline(1.0, color='gold', ls='--', lw=2)
    axes[1, 0].set_title('(d) Saturation Ratio', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "QUANTUM COMPLEXITY\n\n"
    txt += "Lloyd: dC/dt <= 2E/pi\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  type = {d['growth_type']}\n"
        txt += f"  R2_lin = {d['r2_linear']:.3f}\n"
        txt += f"  R2_sqrt = {d['r2_sqrt']:.3f}\n"
        txt += f"  max C = {d['max_complexity']:.1f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 342: Quantum Complexity", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase342_complexity')
    plt.close()
    save_results('phase342_complexity', {'experiment': 'Quantum Complexity', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
