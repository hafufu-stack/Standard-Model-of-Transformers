# -*- coding: utf-8 -*-
"""
Phase 328: Holographic Complexity -- Computational Depth
==========================================================
Holographic complexity: the minimum number of simple operations
to prepare a quantum state from a reference state.
In transformers: how does computational complexity grow with depth?
Test "complexity = action" (CA) and "complexity = volume" (CV) conjectures.
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
    """Measure holographic complexity."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    h0 = out.hidden_states[0][0, -1, :].float()

    # Complexity = distance from reference state (layer 0)
    # CV: complexity ~ volume of maximal slice
    cv_complexity = []
    # CA: complexity ~ action = integral of Lagrangian
    ca_complexity = []

    cumulative_action = 0.0
    for li in range(1, n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()

        # CV: Euclidean distance from reference
        dist = float((h - h0).norm().item())
        cv_complexity.append(dist)

        # CA: "action" = kinetic - potential
        h_prev = out.hidden_states[li - 1][0, -1, :].float()
        kinetic = float(((h - h_prev) ** 2).sum().item())
        potential = float((h ** 2).mean().item())
        action = kinetic - potential
        cumulative_action += action
        ca_complexity.append(float(cumulative_action))

    # Growth rate: dC/dl
    cv_growth = np.diff(cv_complexity) if len(cv_complexity) > 1 else [0]
    ca_growth = np.diff(ca_complexity) if len(ca_complexity) > 1 else [0]

    # Lloyd's bound: dC/dt <= 2*E / (pi*hbar)
    # Check if complexity growth saturates
    cv_arr = np.array(cv_complexity)
    layers = np.arange(1, len(cv_arr) + 1)

    # Linear vs sublinear growth
    slope, intercept, r, _, _ = stats.linregress(layers, cv_arr)
    cv_r2 = r**2

    # Log fit (sublinear)
    log_slope, log_int, log_r, _, _ = stats.linregress(np.log(layers), cv_arr)
    log_r2 = log_r**2

    growth_type = 'linear' if cv_r2 > log_r2 else 'sublinear'

    return {
        'cv_complexity': [round(c, 4) for c in cv_complexity],
        'ca_complexity': [round(c, 4) for c in ca_complexity],
        'cv_growth_rate': [round(g, 4) for g in cv_growth],
        'cv_r2_linear': round(float(cv_r2), 4),
        'cv_r2_log': round(float(log_r2), 4),
        'growth_type': growth_type,
        'final_cv': round(float(cv_complexity[-1]), 4) if cv_complexity else 0,
        'final_ca': round(float(ca_complexity[-1]), 4) if ca_complexity else 0,
    }


def main():
    print("=" * 70)
    print("Phase 328: Holographic Complexity")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        cx_data = []
        for prompt in PROMPTS:
            c = measure_complexity(model, tok, prompt, device)
            cx_data.append(c)

        n = len(cx_data[0]['cv_complexity'])
        avg_cv = [float(np.mean([c['cv_complexity'][i] for c in cx_data])) for i in range(n)]
        avg_ca = [float(np.mean([c['ca_complexity'][i] for c in cx_data])) for i in range(n)]

        all_results[size] = {
            'avg_cv_complexity': [round(c, 4) for c in avg_cv],
            'avg_ca_complexity': [round(c, 4) for c in avg_ca],
            'cv_r2_linear': round(float(np.mean([c['cv_r2_linear'] for c in cx_data])), 4),
            'cv_r2_log': round(float(np.mean([c['cv_r2_log'] for c in cx_data])), 4),
            'growth_type': max(set([c['growth_type'] for c in cx_data]),
                              key=[c['growth_type'] for c in cx_data].count),
            'final_cv': round(float(np.mean([c['final_cv'] for c in cx_data])), 4),
        }
        print(f"  Growth: {all_results[size]['growth_type']}")
        print(f"  CV R2 (linear): {all_results[size]['cv_r2_linear']:.4f}")
        print(f"  Final CV: {all_results[size]['final_cv']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_cv_complexity'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('CV Complexity')
    axes[0, 0].set_title('(a) CV Complexity', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_ca_complexity'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('CA Complexity')
    axes[0, 1].set_title('(b) CA Complexity', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.35
    axes[0, 2].bar(x - w/2, [all_results[s]['cv_r2_linear'] for s in sizes], w,
                  label='Linear', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['cv_r2_log'] for s in sizes], w,
                  label='Log', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_ylabel('R2'); axes[0, 2].set_title('(c) Growth Fit', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "HOLOGRAPHIC COMPLEXITY\n\n"
    txt += "CV: C ~ dist(h_l, h_0)\n"
    txt += "CA: C ~ integral(action)\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}: {d['growth_type']}\n"
        txt += f"  R2_lin={d['cv_r2_linear']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 328: Holographic Complexity", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase328_complexity')
    plt.close()
    save_results('phase328_complexity', {'experiment': 'Holographic Complexity', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
