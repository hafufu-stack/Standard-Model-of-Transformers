# -*- coding: utf-8 -*-
"""
Phase 341: Quantum Channel Capacity
=====================================================
Each Transformer layer acts as a quantum channel. The capacity
Q = max I(A;B) determines the maximum information throughput.
Test whether layers have finite, measurable channel capacity
and how it varies with depth.
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


def measure_channel_capacity(model, tok, prompt, device):
    """Measure channel capacity of each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]

    capacities = []
    fidelities = []
    noise_levels = []

    for li in range(n_layers):
        h_in = hiddens[li]
        h_out = hiddens[li + 1]
        dim = h_in.shape[0]

        # Channel capacity proxy: mutual information I(in; out)
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        # Proxy: use correlation-based MI estimate
        cos = float(torch.nn.functional.cosine_similarity(
            h_in.unsqueeze(0), h_out.unsqueeze(0)).item())

        # Capacity ~ log(1 + SNR) where SNR is signal-to-noise ratio
        signal = float(torch.norm(h_out).item())
        noise = float(torch.norm(h_out - h_in).item())
        snr = signal**2 / (noise**2 + 1e-10)
        capacity = 0.5 * np.log2(1 + snr)  # Shannon capacity

        # Fidelity: |<in|out>|^2 / (||in|| * ||out||)
        fidelity = cos**2

        capacities.append(round(float(capacity), 4))
        fidelities.append(round(float(fidelity), 4))
        noise_levels.append(round(float(noise / (signal + 1e-10)), 4))

    # Cumulative capacity: total bits transmitted
    total_capacity = float(np.sum(capacities))

    # Bottleneck: minimum capacity layer
    bottleneck_layer = int(np.argmin(capacities))
    bottleneck_cap = float(capacities[bottleneck_layer])

    # Capacity scaling: does C ~ log(d)?
    layers = np.arange(len(capacities))
    if len(layers) > 3:
        slope, _, r, _, _ = stats.linregress(layers, capacities)
        r2_linear = r**2
    else:
        slope, r2_linear = 0, 0

    return {
        'capacities': capacities,
        'fidelities': fidelities,
        'noise_levels': noise_levels,
        'total_capacity': round(total_capacity, 2),
        'bottleneck_layer': bottleneck_layer,
        'bottleneck_cap': round(bottleneck_cap, 4),
        'capacity_slope': round(float(slope), 4),
        'r2_linear': round(float(r2_linear), 4),
        'avg_fidelity': round(float(np.mean(fidelities)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 341: Quantum Channel Capacity")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        cap_data = []
        for prompt in PROMPTS:
            c = measure_channel_capacity(model, tok, prompt, device)
            cap_data.append(c)

        n = len(cap_data[0]['capacities'])
        all_results[size] = {
            'capacities': [round(float(np.mean([c['capacities'][i] for c in cap_data])), 4)
                          for i in range(n)],
            'fidelities': [round(float(np.mean([c['fidelities'][i] for c in cap_data])), 4)
                          for i in range(n)],
            'noise_levels': [round(float(np.mean([c['noise_levels'][i] for c in cap_data])), 4)
                            for i in range(n)],
            'total_capacity': round(float(np.mean([c['total_capacity'] for c in cap_data])), 2),
            'bottleneck_layer': int(np.median([c['bottleneck_layer'] for c in cap_data])),
            'bottleneck_cap': round(float(np.mean([c['bottleneck_cap'] for c in cap_data])), 4),
            'avg_fidelity': round(float(np.mean([c['avg_fidelity'] for c in cap_data])), 4),
        }
        print(f"  Total capacity: {all_results[size]['total_capacity']:.2f} bits")
        print(f"  Bottleneck: L{all_results[size]['bottleneck_layer']} ({all_results[size]['bottleneck_cap']:.4f})")
        print(f"  Avg fidelity: {all_results[size]['avg_fidelity']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['capacities'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('C (bits)')
    axes[0, 0].set_title('(a) Channel Capacity', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['fidelities'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Fidelity')
    axes[0, 1].set_title('(b) Channel Fidelity', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['noise_levels'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Noise/Signal')
    axes[0, 2].set_title('(c) Noise Levels', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].bar(sizes, [all_results[s]['total_capacity'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Total bits')
    axes[1, 0].set_title('(d) Total Capacity', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "CHANNEL CAPACITY\n\n"
    txt += "C = 0.5*log2(1+SNR)\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  total = {d['total_capacity']:.1f} bits\n"
        txt += f"  bottleneck = L{d['bottleneck_layer']}\n"
        txt += f"  fidelity = {d['avg_fidelity']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 341: Quantum Channel Capacity", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase341_channel')
    plt.close()
    save_results('phase341_channel', {'experiment': 'Channel Capacity', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
