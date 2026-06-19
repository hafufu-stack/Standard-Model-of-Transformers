# -*- coding: utf-8 -*-
"""
Phase 324: Deconfinement Transition -- Temperature-Driven Phase Change
=======================================================================
In QCD, there is a deconfinement transition at high temperature where
quarks become free (quark-gluon plasma).
Test: do transformers show a deconfinement transition as
the hidden state "temperature" changes across layers?
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


def measure_deconfinement(model, tok, prompt, device):
    """Detect deconfinement transition."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Temperature and confinement measure at each layer
    layer_T = []
    layer_polyakov = []  # Polyakov loop = order parameter for deconfinement

    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        T = float(h.std().item())
        layer_T.append(T)

        # Polyakov loop analogue:
        # P = trace of product of link variables along "time" direction
        # Here: participation ratio as confinement measure
        h_sq = h ** 2
        p = h_sq / (h_sq.sum() + 1e-10)
        PR = float(1.0 / (p ** 2).sum().item())
        D = len(h)
        polyakov = PR / D  # normalized
        layer_polyakov.append(polyakov)

    # Detect transition: sharp change in Polyakov loop
    polyakov_arr = np.array(layer_polyakov)
    polyakov_grad = np.abs(np.diff(polyakov_arr))
    transition_layer = int(np.argmax(polyakov_grad))
    transition_T = layer_T[transition_layer]

    # Phase classification
    confined_PR = np.mean(polyakov_arr[:n_layers//3])
    deconfined_PR = np.mean(polyakov_arr[2*n_layers//3:])
    has_transition = abs(deconfined_PR - confined_PR) / (confined_PR + 1e-10) > 0.1

    return {
        'layer_T': [round(t, 4) for t in layer_T],
        'layer_polyakov': [round(p, 4) for p in layer_polyakov],
        'transition_layer': transition_layer,
        'transition_T': round(transition_T, 4),
        'confined_PR': round(float(confined_PR), 4),
        'deconfined_PR': round(float(deconfined_PR), 4),
        'has_transition': has_transition,
    }


def main():
    print("=" * 70)
    print("Phase 324: Deconfinement Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        dec_data = []
        for prompt in PROMPTS:
            d = measure_deconfinement(model, tok, prompt, device)
            dec_data.append(d)

        n = len(dec_data[0]['layer_polyakov'])
        avg_poly = [float(np.mean([d['layer_polyakov'][i] for d in dec_data])) for i in range(n)]
        avg_T = [float(np.mean([d['layer_T'][i] for d in dec_data])) for i in range(n)]

        all_results[size] = {
            'avg_polyakov': [round(p, 4) for p in avg_poly],
            'avg_temperature': [round(t, 4) for t in avg_T],
            'avg_transition_layer': round(float(np.mean([d['transition_layer'] for d in dec_data])), 1),
            'avg_transition_T': round(float(np.mean([d['transition_T'] for d in dec_data])), 4),
            'confined_PR': round(float(np.mean([d['confined_PR'] for d in dec_data])), 4),
            'deconfined_PR': round(float(np.mean([d['deconfined_PR'] for d in dec_data])), 4),
            'has_transition': sum(1 for d in dec_data if d['has_transition']) >= 3,
        }
        tr = 'YES' if all_results[size]['has_transition'] else 'NO'
        print(f"  Transition: {tr} at L{all_results[size]['avg_transition_layer']:.0f}")
        print(f"  T_c = {all_results[size]['avg_transition_T']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_polyakov'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Polyakov Loop')
    axes[0, 0].set_title('(a) Polyakov Loop (Order Param)', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_temperature'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Temperature')
    axes[0, 1].set_title('(b) Temperature Profile', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Polyakov vs T scatter
    for size, data in all_results.items():
        axes[0, 2].scatter(data['avg_temperature'], data['avg_polyakov'],
                          color=colors[size], s=20, label=size)
    axes[0, 2].set_xlabel('T'); axes[0, 2].set_ylabel('Polyakov')
    axes[0, 2].set_title('(c) Polyakov vs T', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "DECONFINEMENT\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        tr = 'YES' if d['has_transition'] else 'NO'
        txt += f"  Transition: {tr}\n"
        txt += f"  L_c = {d['avg_transition_layer']:.0f}\n"
        txt += f"  T_c = {d['avg_transition_T']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 324: Deconfinement Transition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase324_deconfinement')
    plt.close()
    save_results('phase324_deconfinement', {'experiment': 'Deconfinement Transition', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
