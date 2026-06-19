# -*- coding: utf-8 -*-
"""
Phase 307: Partition Function -- Free Energy Landscape
========================================================
The partition function Z = sum_i exp(-E_i/T) encodes all thermodynamic info.
Free energy F = -T * log(Z).
Map the free energy landscape across layers and models.
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


def compute_partition_function(model, tok, prompt, device):
    """Compute partition function and free energy at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    layer_Z = []
    layer_F = []
    layer_S = []  # Entropy S = -dF/dT
    layer_E = []  # Energy E = F + TS

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()

        # Temperature
        T = float(np.std(h))

        # Energy levels: squared components (sorted)
        E_i = np.sort(h ** 2)[::-1]

        # Partition function
        beta = 1.0 / (T + 1e-10)
        # Clip to prevent overflow
        boltz = np.exp(-beta * E_i)
        boltz = np.clip(boltz, 1e-300, 1e300)
        Z = float(np.sum(boltz))

        # Free energy F = -T * log(Z)
        F = -T * np.log(Z + 1e-300)

        # Internal energy E = <E> = sum(E_i * p_i)
        p_i = boltz / (Z + 1e-300)
        E_avg = float(np.sum(E_i * p_i))

        # Entropy S = (E - F) / T
        S = (E_avg - F) / (T + 1e-10)

        layer_Z.append(float(Z))
        layer_F.append(float(F))
        layer_S.append(float(S))
        layer_E.append(float(E_avg))

    # Free energy barrier: max(F) - min(F)
    F_arr = np.array(layer_F)
    F_barrier = float(np.max(F_arr) - np.min(F_arr))

    # Specific heat from dE/dT (numerical)
    E_arr = np.array(layer_E)
    T_list = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
        T_list.append(float(np.std(h)))
    T_arr = np.array(T_list)

    return {
        'layer_Z': [round(z, 4) for z in layer_Z],
        'layer_F': [round(f, 4) for f in layer_F],
        'layer_S': [round(s, 4) for s in layer_S],
        'layer_E': [round(e, 4) for e in layer_E],
        'F_barrier': round(F_barrier, 4),
        'mean_F': round(float(np.mean(layer_F)), 4),
        'mean_S': round(float(np.mean(layer_S)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 307: Partition Function -- Free Energy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        pf_data = []
        for prompt in PROMPTS:
            pf = compute_partition_function(model, tok, prompt, device)
            pf_data.append(pf)

        n = len(pf_data[0]['layer_F'])
        avg_F = [float(np.mean([p['layer_F'][i] for p in pf_data])) for i in range(n)]
        avg_S = [float(np.mean([p['layer_S'][i] for p in pf_data])) for i in range(n)]
        avg_E = [float(np.mean([p['layer_E'][i] for p in pf_data])) for i in range(n)]
        avg_Z = [float(np.mean([p['layer_Z'][i] for p in pf_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n - 1,
            'avg_F': [round(f, 4) for f in avg_F],
            'avg_S': [round(s, 4) for s in avg_S],
            'avg_E': [round(e, 4) for e in avg_E],
            'avg_Z': [round(z, 4) for z in avg_Z],
            'F_barrier': round(float(np.mean([p['F_barrier'] for p in pf_data])), 4),
            'mean_F': round(float(np.mean(avg_F)), 4),
            'mean_S': round(float(np.mean(avg_S)), 4),
        }
        print(f"  Mean Free Energy: {all_results[size]['mean_F']:.4f}")
        print(f"  Mean Entropy: {all_results[size]['mean_S']:.4f}")
        print(f"  F barrier: {all_results[size]['F_barrier']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Free energy
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_F'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Free Energy F')
    axes[0, 0].set_title('(a) Free Energy Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Entropy
    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_S'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Entropy S')
    axes[0, 1].set_title('(b) Entropy Profile', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Internal energy
    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_E'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Internal Energy E')
    axes[0, 2].set_title('(c) Internal Energy', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) F vs S (Legendre structure)
    for size, data in all_results.items():
        axes[1, 0].plot(data['avg_S'], data['avg_F'], 'o-', color=colors[size],
                       lw=1.5, markersize=3, label=size)
    axes[1, 0].set_xlabel('Entropy S')
    axes[1, 0].set_ylabel('Free Energy F')
    axes[1, 0].set_title('(d) F vs S (Legendre)', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Partition function
    for size, data in all_results.items():
        axes[1, 1].semilogy(data['avg_Z'], '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Z (log scale)')
    axes[1, 1].set_title('(e) Partition Function', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    sizes = list(all_results.keys())
    txt = "PARTITION FUNCTION\n\n"
    txt += "F = -T log Z\n"
    txt += "S = (E - F) / T\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  <F> = {d['mean_F']:.2f}\n"
        txt += f"  <S> = {d['mean_S']:.2f}\n"
        txt += f"  F barrier = {d['F_barrier']:.2f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 307: Partition Function and Free Energy Landscape",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase307_partition')
    plt.close()

    save_results('phase307_partition', {
        'experiment': 'Partition Function - Free Energy',
        'results': {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
