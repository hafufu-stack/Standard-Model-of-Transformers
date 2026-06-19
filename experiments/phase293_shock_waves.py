# -*- coding: utf-8 -*-
"""
Phase 293: Transonic Shock Waves
==================================
1.5B is supersonic (Mach=1.14). Supersonic flow creates shock waves.
Detect shock-like discontinuities in hidden state norms.
A shock = sudden jump in norm/temperature between adjacent layers.
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
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
    "The chemical composition of water molecules is",
    "Artificial intelligence will transform how we live and work",
]


def detect_shocks(hidden_states, threshold_sigma=2.0):
    """Detect shock-like discontinuities in norm profile."""
    norms = []
    temps = []
    for h in hidden_states:
        v = h[0, -1, :].float().cpu()
        norms.append(v.norm().item())
        temps.append(v.std().item())

    # Layer-to-layer jumps
    norm_jumps = [abs(norms[i+1] - norms[i]) for i in range(len(norms)-1)]
    temp_jumps = [abs(temps[i+1] - temps[i]) for i in range(len(temps)-1)]

    # Shock = jump > mean + threshold * std
    mean_nj = np.mean(norm_jumps)
    std_nj = np.std(norm_jumps)
    mean_tj = np.mean(temp_jumps)
    std_tj = np.std(temp_jumps)

    shock_layers_norm = [i for i, j in enumerate(norm_jumps)
                        if j > mean_nj + threshold_sigma * std_nj]
    shock_layers_temp = [i for i, j in enumerate(temp_jumps)
                        if j > mean_tj + threshold_sigma * std_tj]

    # Compression ratio at each shock
    compressions = []
    for sl in shock_layers_norm:
        if norms[sl] > 0:
            ratio = norms[sl+1] / norms[sl]
            compressions.append(round(ratio, 4))

    return {
        'norms': [round(n, 4) for n in norms],
        'temps': [round(t, 4) for t in temps],
        'norm_jumps': [round(j, 4) for j in norm_jumps],
        'temp_jumps': [round(j, 4) for j in temp_jumps],
        'shock_layers_norm': shock_layers_norm,
        'shock_layers_temp': shock_layers_temp,
        'compressions': compressions,
        'n_shocks': len(shock_layers_norm),
        'max_jump_norm': round(float(max(norm_jumps)), 4) if norm_jumps else 0,
        'max_jump_layer': int(np.argmax(norm_jumps)) if norm_jumps else 0,
    }


def main():
    print("=" * 70)
    print("Phase 293: Transonic Shock Waves")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B', '7B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        shocks_all = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            shock_data = detect_shocks(out.hidden_states)
            shocks_all.append(shock_data)

        # Aggregate
        avg_n_shocks = float(np.mean([s['n_shocks'] for s in shocks_all]))
        all_shock_layers = []
        for s in shocks_all:
            all_shock_layers.extend(s['shock_layers_norm'])
        shock_histogram = {}
        for sl in all_shock_layers:
            shock_histogram[sl] = shock_histogram.get(sl, 0) + 1

        # Average norm profile
        n_layers = len(shocks_all[0]['norms'])
        avg_norms = [float(np.mean([s['norms'][i] for s in shocks_all]))
                    for i in range(n_layers)]
        avg_jumps = [float(np.mean([s['norm_jumps'][i] for s in shocks_all]))
                    for i in range(n_layers - 1)]

        all_results[size] = {
            'n_layers': n_layers - 1,
            'avg_n_shocks': round(avg_n_shocks, 2),
            'shock_histogram': shock_histogram,
            'avg_norms': [round(n, 4) for n in avg_norms],
            'avg_jumps': [round(j, 4) for j in avg_jumps],
            'max_jump_layer': int(np.argmax(avg_jumps)),
            'max_jump_value': round(float(max(avg_jumps)), 4),
        }
        print(f"  Avg shocks: {avg_n_shocks:.1f}")
        print(f"  Max jump at layer {all_results[size]['max_jump_layer']}: "
              f"{all_results[size]['max_jump_value']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) Norm profiles
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_norms'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Hidden State Norm')
    axes[0, 0].set_title('(a) Norm Profiles', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Jump magnitudes (shock detection)
    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_jumps'], '-', color=colors[size], lw=2, label=size)
        sl = data['max_jump_layer']
        axes[0, 1].axvline(sl, color=colors[size], ls='--', alpha=0.3)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('|Norm Jump|')
    axes[0, 1].set_title('(b) Shock Detection (Norm Jumps)', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Shock frequency histogram
    for size, data in all_results.items():
        if data['shock_histogram']:
            layers = sorted(data['shock_histogram'].keys())
            counts = [data['shock_histogram'][l] for l in layers]
            axes[0, 2].bar([l + {'0.5B': -0.3, '1.5B': 0, '7B': 0.3}[size] for l in layers],
                          counts, width=0.25, color=colors[size], alpha=0.7, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Shock Frequency')
    axes[0, 2].set_title('(c) Shock Layer Histogram', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Number of shocks vs model size
    sizes = list(all_results.keys())
    n_shocks = [all_results[s]['avg_n_shocks'] for s in sizes]
    axes[1, 0].bar(sizes, n_shocks, color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Avg Number of Shocks')
    axes[1, 0].set_title('(d) Shock Count vs Scale', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) Normalized jump profile
    for size, data in all_results.items():
        n = len(data['avg_jumps'])
        x_norm = np.linspace(0, 1, n)
        axes[1, 1].plot(x_norm, data['avg_jumps'], '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Normalized Depth')
    axes[1, 1].set_ylabel('|Norm Jump|')
    axes[1, 1].set_title('(e) Normalized Shock Profile', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "TRANSONIC SHOCK WAVES\n\n"
    for size, data in all_results.items():
        txt += f"{size}:\n"
        txt += f"  Avg shocks: {data['avg_n_shocks']:.1f}\n"
        txt += f"  Max jump: L{data['max_jump_layer']} "
        txt += f"({data['max_jump_value']:.4f})\n\n"
    txt += "Subsonic: smooth flow\n"
    txt += "Supersonic: shock waves\n"
    txt += "at layer boundaries"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 293: Transonic Shock Waves in Transformer Flow",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase293_shock_waves')
    plt.close()

    save_results('phase293_shock_waves', {
        'experiment': 'Transonic Shock Waves',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
