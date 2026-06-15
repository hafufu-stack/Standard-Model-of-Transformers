# -*- coding: utf-8 -*-
"""
Phase 242: Thermodynamic Work and Heat
========================================
Decompose layer-to-layer energy changes into work (dW) and heat (dQ).
First law: dU = dQ - dW
dW = change in "useful" output (information gain toward prediction)
dQ = dissipated energy (entropy production * T)
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
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Plate tectonics shapes the surface of the Earth",
    "The brain contains billions of neurons",
    "Chemical reactions follow conservation of mass",
    "Stars form from collapsing molecular clouds",
    "Algorithms determine computational complexity",
    "Proteins fold into specific three dimensional shapes",
]


def work_heat_decomp(model, tok, device, model_name):
    """Decompose energy changes into work and heat."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_dU, all_dW, all_dQ = [], [], []
    all_efficiency = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Measure at each layer
        layer_data = []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T = float(S) if not np.isnan(S) else 0
            layer_data.append({'U': U, 'T': T, 'P1': P1, 'S': S})

        # Decompose
        dU_l, dW_l, dQ_l, eff_l = [], [], [], []
        for i in range(len(layer_data) - 1):
            dU = layer_data[i+1]['U'] - layer_data[i]['U']
            dS = layer_data[i+1]['S'] - layer_data[i]['S']
            T_avg = (layer_data[i]['T'] + layer_data[i+1]['T']) / 2

            # Work = useful information gain = -dT (reduction in uncertainty)
            dW = -(layer_data[i+1]['T'] - layer_data[i]['T'])
            # Heat = T * dS (dissipation)
            dQ = T_avg * dS if not np.isnan(T_avg * dS) else 0

            # Efficiency = |dW| / (|dW| + |dQ|) if both positive
            if abs(dW) + abs(dQ) > 1e-10:
                eff = abs(dW) / (abs(dW) + abs(dQ))
            else:
                eff = 0

            dU_l.append(dU); dW_l.append(dW); dQ_l.append(float(dQ))
            eff_l.append(float(eff))

        all_dU.append(dU_l); all_dW.append(dW_l); all_dQ.append(dQ_l)
        all_efficiency.append(eff_l)

    n = min(len(x) for x in all_dU)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_dU = avg(all_dU)
    mean_dW = avg(all_dW)
    mean_dQ = avg(all_dQ)
    mean_eff = avg(all_efficiency)

    # Cumulative work and heat
    cum_W = np.cumsum(mean_dW)
    cum_Q = np.cumsum(mean_dQ)

    # Overall efficiency
    total_W = sum(max(w, 0) for w in mean_dW)
    total_Q = sum(abs(q) for q in mean_dQ)
    overall_eff = total_W / (total_W + total_Q + 1e-10)

    # Carnot efficiency: eta_C = 1 - T_cold/T_hot
    # Use initial T as T_hot and final T as T_cold (in the useful direction)
    T_layers = [np.mean([all_dU[p][0] for p in range(len(PROMPTS))])]  # placeholder

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_dU': mean_dU,
        'mean_dW': mean_dW,
        'mean_dQ': mean_dQ,
        'mean_eff': mean_eff,
        'cum_W': cum_W.tolist(),
        'cum_Q': cum_Q.tolist(),
        'overall_eff': overall_eff,
        'total_W': total_W,
        'total_Q': total_Q,
    }


def main():
    print("=" * 70)
    print("Phase 242: Thermodynamic Work and Heat")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = work_heat_decomp(model, tok, device, size)
        results[size] = r
        print(f"  Overall efficiency: {r['overall_eff']:.3f}")
        print(f"  Total work: {r['total_W']:.2f}, Total heat: {r['total_Q']:.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) dW and dQ per layer
    for size, r in results.items():
        c = colors[size]
        layers = range(len(r['mean_dW']))
        axes[0, 0].plot(layers, r['mean_dW'], '-', color=c, lw=2, label=f'dW ({size})')
        axes[0, 0].plot(layers, r['mean_dQ'], '--', color=c, lw=1.5, label=f'dQ ({size})')
    axes[0, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 0].set_xlabel('Layer Transition')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('(a) Work (dW) and Heat (dQ)')
    axes[0, 0].legend(fontsize=7)

    # (b) Cumulative work and heat
    for size, r in results.items():
        c = colors[size]
        axes[0, 1].plot(range(len(r['cum_W'])), r['cum_W'], '-', color=c, lw=2, label=f'Cum W ({size})')
        axes[0, 1].plot(range(len(r['cum_Q'])), r['cum_Q'], '--', color=c, lw=1.5, label=f'Cum Q ({size})')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Cumulative Energy')
    axes[0, 1].set_title('(b) Cumulative Work & Heat')
    axes[0, 1].legend(fontsize=7)

    # (c) Layer efficiency
    for size, r in results.items():
        c = colors[size]
        axes[0, 2].plot(range(len(r['mean_eff'])), r['mean_eff'], '-o', color=c,
                       lw=2, markersize=3, label=size)
    axes[0, 2].set_xlabel('Layer Transition')
    axes[0, 2].set_ylabel('Efficiency')
    axes[0, 2].set_title('(c) Layer-wise Efficiency')
    axes[0, 2].legend(fontsize=8)

    # (d) dU decomposition
    for size, r in results.items():
        c = colors[size]
        axes[1, 0].fill_between(range(len(r['mean_dW'])), 0, r['mean_dW'],
                                alpha=0.3, color=c, label=f'dW ({size})')
        axes[1, 0].fill_between(range(len(r['mean_dQ'])), 0, r['mean_dQ'],
                                alpha=0.15, color=c, hatch='//')
    axes[1, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 0].set_xlabel('Layer Transition')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('(d) Energy Decomposition')
    axes[1, 0].legend(fontsize=7)

    # (e) Efficiency comparison
    sizes = list(results.keys())
    effs = [results[s]['overall_eff'] for s in sizes]
    axes[1, 1].bar(range(len(sizes)), effs, color=[colors[s] for s in sizes], alpha=0.8)
    axes[1, 1].set_xticks(range(len(sizes)))
    axes[1, 1].set_xticklabels(sizes)
    axes[1, 1].set_ylabel('Overall Efficiency')
    axes[1, 1].set_title('(e) Overall Efficiency')
    for i, e in enumerate(effs):
        axes[1, 1].text(i, e + 0.01, f'{e:.3f}', ha='center', fontsize=10)

    # (f) Summary
    summary = "WORK-HEAT DECOMPOSITION\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Total W = {r['total_W']:.2f}\n"
        summary += f"  Total Q = {r['total_Q']:.2f}\n"
        summary += f"  Efficiency = {r['overall_eff']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 242: Thermodynamic Work and Heat Decomposition",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase242_work_heat')
    plt.close()
    save_results('phase242_work_heat', {'experiment': 'Work-Heat', 'results': results})


if __name__ == '__main__':
    main()
