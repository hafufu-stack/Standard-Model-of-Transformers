# -*- coding: utf-8 -*-
"""
Phase 353: Calabi-Yau Compactification -- Internal Geometry
=====================================================
String theory requires extra dimensions compactified on Calabi-Yau
manifolds. The Euler number chi determines the number of generations.
Test whether the hidden state has internal compact dimensions with
CY-like properties.
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


def measure_calabi_yau(model, tok, prompt, device):
    """Test Calabi-Yau compactification properties."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # 1. Effective dimensionality: PCA to find "large" and "compact" dims
    # Stack all layers into a matrix
    H = torch.stack(hiddens)  # (n_layers+1, dim)
    try:
        U, S, V = torch.linalg.svd(H, full_matrices=False)
        total = float(torch.sum(S**2).item())
        cumsum = torch.cumsum(S**2, dim=0) / (total + 1e-10)

        # "Large" dimensions: those capturing 90% of variance
        d_large = int((cumsum < 0.90).sum().item()) + 1
        # "Compact" dimensions: the rest
        d_compact = len(S) - d_large
        # Effective compact dimensionality
        compact_frac = d_compact / len(S)
    except:
        d_large, d_compact, compact_frac = 0, 0, 0

    # 2. Euler characteristic proxy: alternating sum of Betti numbers
    # Betti_k = rank of k-th homology group
    # Proxy: use eigenvalue gaps as "holes"
    betti_numbers = []
    for li in range(n_layers + 1):
        h = hiddens[li].abs()
        sorted_h = torch.sort(h)[0].numpy()
        # Find significant gaps (potential "holes")
        gaps = np.diff(sorted_h)
        threshold = np.mean(gaps) + 2 * np.std(gaps)
        n_gaps = int(np.sum(gaps > threshold))
        betti_numbers.append(n_gaps)

    # Euler = sum(-1)^k * beta_k approximated as alternating sum
    euler = sum((-1)**i * b for i, b in enumerate(betti_numbers[:6]))

    # 3. Hodge numbers: h^{1,1} and h^{2,1} for CY 3-fold
    # Proxy: symmetric vs antisymmetric mode counts
    h11_proxy = []
    h21_proxy = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        # Symmetric modes: even dimensions
        sym = h[::2]
        antisym = h[1::2]
        h11_proxy.append(round(float(torch.var(sym).item()), 6))
        h21_proxy.append(round(float(torch.var(antisym).item()), 6))

    # 4. Ricci-flatness: CY has Ricci-flat metric
    # Test: is the "curvature" (second derivative) near zero?
    ricci_profile = []
    for li in range(1, n_layers):
        d2h = hiddens[li + 1] - 2 * hiddens[li] + hiddens[li - 1]
        ricci = float(torch.mean(d2h**2).item())
        ricci_profile.append(round(float(ricci), 6))

    avg_ricci = float(np.mean(ricci_profile)) if ricci_profile else 0

    return {
        'd_large': d_large,
        'd_compact': d_compact,
        'compact_frac': round(float(compact_frac), 4),
        'euler': euler,
        'betti_numbers': betti_numbers[:6],
        'h11_mean': round(float(np.mean(h11_proxy)), 6),
        'h21_mean': round(float(np.mean(h21_proxy)), 6),
        'avg_ricci': round(float(avg_ricci), 6),
        'ricci_profile': ricci_profile,
        'ricci_flat': avg_ricci < 0.1,
    }


def main():
    print("=" * 70)
    print("Phase 353: Calabi-Yau Compactification")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        cy_data = []
        for prompt in PROMPTS:
            c = measure_calabi_yau(model, tok, prompt, device)
            cy_data.append(c)

        n_r = len(cy_data[0]['ricci_profile'])
        all_results[size] = {
            'd_large': round(float(np.mean([c['d_large'] for c in cy_data])), 1),
            'd_compact': round(float(np.mean([c['d_compact'] for c in cy_data])), 1),
            'compact_frac': round(float(np.mean([c['compact_frac'] for c in cy_data])), 4),
            'euler': round(float(np.mean([c['euler'] for c in cy_data])), 1),
            'h11_mean': round(float(np.mean([c['h11_mean'] for c in cy_data])), 6),
            'h21_mean': round(float(np.mean([c['h21_mean'] for c in cy_data])), 6),
            'avg_ricci': round(float(np.mean([c['avg_ricci'] for c in cy_data])), 6),
            'ricci_profile': [round(float(np.mean([c['ricci_profile'][i] for c in cy_data])), 6)
                             for i in range(n_r)],
            'ricci_flat': sum(1 for c in cy_data if c['ricci_flat']) >= 4,
        }
        flat = 'YES' if all_results[size]['ricci_flat'] else 'NO'
        print(f"  Large dims: {all_results[size]['d_large']:.1f}")
        print(f"  Compact dims: {all_results[size]['d_compact']:.1f}")
        print(f"  Euler: {all_results[size]['euler']:.1f}")
        print(f"  Ricci flat: {flat}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[0, 0].bar(x - w/2, [all_results[s]['d_large'] for s in sizes], w, label='Large', color='#3498db')
    axes[0, 0].bar(x + w/2, [all_results[s]['d_compact'] for s in sizes], w, label='Compact', color='#e74c3c')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(sizes)
    axes[0, 0].set_title('(a) Large vs Compact Dims', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].bar(sizes, [all_results[s]['euler'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 1].set_title('(b) Euler Characteristic', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['ricci_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('|Ricci|^2')
    axes[0, 2].set_title('(c) Ricci Curvature', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].bar(x - w/2, [all_results[s]['h11_mean'] for s in sizes], w, label='h^{1,1}', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['h21_mean'] for s in sizes], w, label='h^{2,1}', color='#e74c3c')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_title('(d) Hodge Numbers (proxy)', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "CALABI-YAU\n\n"
    for s in sizes:
        d = all_results[s]
        flat = 'YES' if d['ricci_flat'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  d_L = {d['d_large']:.0f}\n"
        txt += f"  d_C = {d['d_compact']:.0f}\n"
        txt += f"  chi = {d['euler']:.0f}\n"
        txt += f"  Ricci flat: {flat}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 353: Calabi-Yau Compactification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase353_calabi_yau')
    plt.close()
    save_results('phase353_calabi_yau', {'experiment': 'Calabi-Yau', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
