# -*- coding: utf-8 -*-
"""
Phase 243: Thermodynamic Carnot Cycle
======================================
Can we construct a Carnot-like cycle in the transformer's state space?
A cycle of: isothermal expansion -> adiabatic expansion -> isothermal compression -> adiabatic compression
Mapped to: easy prompt layers -> transition -> hard prompt layers -> transition
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

# Two contrasting prompts for the cycle
HOT_PROMPTS = [  # High temperature (uncertain)
    "Purple elephants calculated the square root of happiness while",
    "Yesterday tomorrow forgot to remember the color of silence and",
    "Seven abstract thoughts collided in a vacuum creating new dimensions of",
    "The moon decided to become a professional dancer and performed",
    "Silence tasted exactly like the sound of growing uncertainty in",
]

COLD_PROMPTS = [  # Low temperature (certain)
    "The capital of France is Paris and the capital of Germany is",
    "Water freezes at zero degrees Celsius and boils at one hundred",
    "The square root of one hundred is ten and the square root of",
    "One plus one equals two and two plus two equals four and",
    "The sun rises in the east and sets in the west every single",
]


def carnot_analysis(model, tok, device, model_name):
    """Construct and analyze thermodynamic cycles."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    def get_trajectory(prompts):
        all_T, all_U, all_S = [], [], []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            T_l, U_l, S_l = [], [], []
            for hs in out.hidden_states:
                h = hs[0, -1, :].float()
                U_l.append(h.norm().item())
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                p1 = float(probs.max().item())
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_l.append(float(S) if not np.isnan(S) else 0)
                S_l.append(float(S) if not np.isnan(S) else 0)
            all_T.append(T_l); all_U.append(U_l); all_S.append(S_l)
        n = min(len(t) for t in all_T)
        avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]
        return avg(all_T), avg(all_U), avg(all_S)

    hot_T, hot_U, hot_S = get_trajectory(HOT_PROMPTS)
    cold_T, cold_U, cold_S = get_trajectory(COLD_PROMPTS)

    # Construct cycle in T-S space:
    # 1. Hot path: layers 0 -> N (high T isothermal-like)
    # 2. Transition: hot final -> cold final (adiabatic-like)
    # 3. Cold path: layers N -> 0 (low T isothermal-like, reversed)
    # 4. Transition: cold initial -> hot initial (adiabatic-like)

    cycle_T = hot_T + [cold_T[-1]] + cold_T[::-1] + [hot_T[0]]
    cycle_S = hot_S + [cold_S[-1]] + cold_S[::-1] + [hot_S[0]]
    cycle_U = hot_U + [cold_U[-1]] + cold_U[::-1] + [hot_U[0]]

    # Area enclosed = net work
    # Shoelace formula for area
    area = 0
    for i in range(len(cycle_T) - 1):
        area += (cycle_S[i] * cycle_T[i+1] - cycle_S[i+1] * cycle_T[i])
    area = abs(area) / 2

    # Efficiency estimates
    T_hot = max(hot_T)
    T_cold = min(cold_T)
    carnot_eff = 1 - T_cold / (T_hot + 1e-10) if T_hot > T_cold else 0

    # Delta between hot and cold
    delta_T_final = hot_T[-1] - cold_T[-1]
    delta_T_initial = hot_T[0] - cold_T[0]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'hot_T': hot_T, 'hot_U': hot_U, 'hot_S': hot_S,
        'cold_T': cold_T, 'cold_U': cold_U, 'cold_S': cold_S,
        'cycle_T': cycle_T, 'cycle_S': cycle_S,
        'cycle_area': area,
        'T_hot': T_hot, 'T_cold': T_cold,
        'carnot_eff': carnot_eff,
        'delta_T_final': delta_T_final,
    }


def main():
    print("=" * 70)
    print("Phase 243: Thermodynamic Carnot Cycle")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = carnot_analysis(model, tok, device, size)
        results[size] = r
        print(f"  T_hot={r['T_hot']:.2f}, T_cold={r['T_cold']:.2f}")
        print(f"  Carnot eff={r['carnot_eff']:.3f}")
        print(f"  Cycle area={r['cycle_area']:.2f}")
        print(f"  Delta T_final (hot-cold)={r['delta_T_final']:.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) T-S diagram with cycle (1.5B)
    r15 = results[list(results.keys())[-1]]
    axes[0, 0].plot(r15['hot_S'], r15['hot_T'], 'r-', lw=2, label='Hot path')
    axes[0, 0].plot(r15['cold_S'], r15['cold_T'], 'b-', lw=2, label='Cold path')
    # Connecting lines
    axes[0, 0].plot([r15['hot_S'][-1], r15['cold_S'][-1]],
                   [r15['hot_T'][-1], r15['cold_T'][-1]], 'g--', lw=1.5, label='Transition')
    axes[0, 0].plot([r15['cold_S'][0], r15['hot_S'][0]],
                   [r15['cold_T'][0], r15['hot_T'][0]], 'g--', lw=1.5)
    axes[0, 0].set_xlabel('S (entropy)')
    axes[0, 0].set_ylabel('T (temperature)')
    axes[0, 0].set_title(f'(a) T-S Cycle (area={r15["cycle_area"]:.1f})')
    axes[0, 0].legend(fontsize=7)

    # (b) Hot vs Cold T profiles
    for size, r in results.items():
        n_hot = len(r['hot_T'])
        n_cold = len(r['cold_T'])
        axes[0, 1].plot(np.linspace(0, 1, n_hot), r['hot_T'], '-',
                       color='red', alpha=0.5 if '0.5' in size else 1, lw=2,
                       label=f'Hot ({size})')
        axes[0, 1].plot(np.linspace(0, 1, n_cold), r['cold_T'], '-',
                       color='blue', alpha=0.5 if '0.5' in size else 1, lw=2,
                       label=f'Cold ({size})')
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('Temperature')
    axes[0, 1].set_title('(b) Hot vs Cold Profiles')
    axes[0, 1].legend(fontsize=7)

    # (c) U profiles
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['hot_U'])), r['hot_U'], '-',
                       color='red', alpha=0.5 if '0.5' in size else 1, lw=2)
        axes[0, 2].plot(range(len(r['cold_U'])), r['cold_U'], '-',
                       color='blue', alpha=0.5 if '0.5' in size else 1, lw=2)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('U (internal energy)')
    axes[0, 2].set_title('(c) Energy Profiles')

    # (d) Cycle in T-U space
    axes[1, 0].plot(r15['hot_U'], r15['hot_T'], 'r-', lw=2, label='Hot')
    axes[1, 0].plot(r15['cold_U'], r15['cold_T'], 'b-', lw=2, label='Cold')
    axes[1, 0].plot([r15['hot_U'][-1], r15['cold_U'][-1]],
                   [r15['hot_T'][-1], r15['cold_T'][-1]], 'g--', lw=1.5)
    axes[1, 0].plot([r15['cold_U'][0], r15['hot_U'][0]],
                   [r15['cold_T'][0], r15['hot_T'][0]], 'g--', lw=1.5)
    axes[1, 0].set_xlabel('U'); axes[1, 0].set_ylabel('T')
    axes[1, 0].set_title('(d) T-U Cycle')
    axes[1, 0].legend(fontsize=7)

    # (e) Efficiency comparison
    sizes = list(results.keys())
    effs = [results[s]['carnot_eff'] for s in sizes]
    areas = [results[s]['cycle_area'] for s in sizes]
    axes[1, 1].bar(range(len(sizes)), effs, color=[colors[s] for s in sizes], alpha=0.8)
    axes[1, 1].set_xticks(range(len(sizes)))
    axes[1, 1].set_xticklabels(sizes)
    axes[1, 1].set_ylabel('Carnot Efficiency')
    axes[1, 1].set_title('(e) Carnot Efficiency')
    for i, e in enumerate(effs):
        axes[1, 1].text(i, e + 0.01, f'{e:.3f}', ha='center', fontsize=10)

    # (f) Summary
    summary = "CARNOT CYCLE\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  T_hot={r['T_hot']:.2f}, T_cold={r['T_cold']:.2f}\n"
        summary += f"  eta_C={r['carnot_eff']:.3f}\n"
        summary += f"  Area={r['cycle_area']:.1f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 243: Thermodynamic Carnot Cycle",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase243_carnot')
    plt.close()
    save_results('phase243_carnot', {'experiment': 'Carnot Cycle', 'results': results})


if __name__ == '__main__':
    main()
