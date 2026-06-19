# -*- coding: utf-8 -*-
"""
Phase 310: Spontaneous Symmetry Breaking
==========================================
Does the transformer spontaneously break any symmetry?
- Test permutation symmetry of hidden dimensions
- Test layer translation symmetry
- Look for order parameters that distinguish phases
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


def measure_symmetry_breaking(model, tok, prompt, device):
    """Detect spontaneous symmetry breaking."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    D = model.config.hidden_size

    # 1. Permutation symmetry of dimensions
    # If symmetric: all dimensions should have equal variance
    # Broken: some dimensions dominate
    dim_gini = []  # Gini coefficient of dimension activations
    dim_kurtosis = []

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        h_abs = np.abs(h)
        h_sorted = np.sort(h_abs)
        n = len(h_sorted)
        # Gini coefficient
        cumsum = np.cumsum(h_sorted)
        gini = 1 - 2 * np.sum(cumsum) / (n * cumsum[-1] + 1e-10) + 1/n
        dim_gini.append(float(gini))

        # Kurtosis of activations (excess kurtosis)
        kurt = float(stats.kurtosis(h))
        dim_kurtosis.append(kurt)

    # 2. Layer translation symmetry
    # If symmetric: all layers should look "the same" statistically
    # Broken: layers develop distinct characters
    layer_means = []
    layer_stds = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        layer_means.append(float(np.mean(h)))
        layer_stds.append(float(np.std(h)))

    # CV of means and stds across layers
    mean_cv = float(np.std(layer_means) / (np.abs(np.mean(layer_means)) + 1e-10))
    std_cv = float(np.std(layer_stds) / (np.mean(layer_stds) + 1e-10))

    # 3. Order parameter: magnetization analogue
    # M = <h> / std(h) normalized
    magnetization = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        M = float(np.mean(h) / (np.std(h) + 1e-10))
        magnetization.append(M)

    return {
        'dim_gini': [round(g, 4) for g in dim_gini],
        'dim_kurtosis': [round(k, 4) for k in dim_kurtosis],
        'magnetization': [round(m, 4) for m in magnetization],
        'mean_cv': round(mean_cv, 4),
        'std_cv': round(std_cv, 4),
        'gini_increase': round(float(dim_gini[-1] - dim_gini[0]), 4),
        'kurt_increase': round(float(dim_kurtosis[-1] - dim_kurtosis[0]), 4),
    }


def main():
    print("=" * 70)
    print("Phase 310: Spontaneous Symmetry Breaking")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ssb_data = []
        for prompt in PROMPTS:
            s = measure_symmetry_breaking(model, tok, prompt, device)
            ssb_data.append(s)

        n = len(ssb_data[0]['dim_gini'])
        avg_gini = [float(np.mean([s['dim_gini'][i] for s in ssb_data])) for i in range(n)]
        avg_kurt = [float(np.mean([s['dim_kurtosis'][i] for s in ssb_data])) for i in range(n)]
        avg_mag = [float(np.mean([s['magnetization'][i] for s in ssb_data])) for i in range(n)]

        gini_increase = float(avg_gini[-1] - avg_gini[0])
        ssb_detected = gini_increase > 0.05  # significant increase

        all_results[size] = {
            'n_layers': n - 1,
            'avg_gini': [round(g, 4) for g in avg_gini],
            'avg_kurtosis': [round(k, 4) for k in avg_kurt],
            'avg_magnetization': [round(m, 4) for m in avg_mag],
            'gini_increase': round(gini_increase, 4),
            'ssb_detected': ssb_detected,
            'mean_cv': round(float(np.mean([s['mean_cv'] for s in ssb_data])), 4),
            'std_cv': round(float(np.mean([s['std_cv'] for s in ssb_data])), 4),
        }
        print(f"  Gini increase: {gini_increase:.4f} ({'SSB!' if ssb_detected else 'no SSB'})")
        print(f"  Mean CV: {all_results[size]['mean_cv']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_gini'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Gini Coefficient')
    axes[0, 0].set_title('(a) Dimension Inequality (Gini)', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_kurtosis'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Kurtosis')
    axes[0, 1].set_title('(b) Activation Kurtosis', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_magnetization'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].axhline(0, color='gold', ls='--', lw=1)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Magnetization M')
    axes[0, 2].set_title('(c) Order Parameter (Magnetization)', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].bar(sizes, [all_results[s]['gini_increase'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].axhline(0.05, color='gold', ls='--', label='SSB threshold')
    axes[1, 0].set_ylabel('Gini Increase'); axes[1, 0].set_title('(d) SSB Measure', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')

    txt = "SYMMETRY BREAKING\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Gini change: {d['gini_increase']:.3f}\n"
        txt += f"  SSB: {'YES' if d['ssb_detected'] else 'NO'}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 310: Spontaneous Symmetry Breaking", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase310_ssb')
    plt.close()

    save_results('phase310_ssb', {'experiment': 'Spontaneous Symmetry Breaking', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
