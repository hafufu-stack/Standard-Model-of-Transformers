# -*- coding: utf-8 -*-
"""
Phase 220: Entropy Production Rate
=====================================
Measure entropy production rate dS_irr/dl at each layer.
Test the Maximum Entropy Production Principle (MEPP):
does the system maximize its entropy production rate?
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
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
    "Superconductors have zero electrical resistance",
    "The brain contains billions of neurons",
    "Algorithms determine computational complexity",
    "Chemical reactions follow conservation of mass",
    "Stars form from collapsing molecular clouds",
]


def measure_entropy_production(model, tok, device, model_name):
    """Measure entropy production rate at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_S = []   # Logit entropy at each layer
    all_S_h = [] # Hidden state entropy
    all_U = []   # Internal energy
    all_T = []   # Temperature

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_list, Sh_list, U_list, T_list = [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())

            # Hidden state entropy (from squared components as probabilities)
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S_h = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()
            Sh_list.append(S_h)

            # Logit entropy
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_list.append(S if not np.isnan(S) else 0)
            T_list.append(S if not np.isnan(S) else 0)

        all_S.append(S_list)
        all_S_h.append(Sh_list)
        all_U.append(U_list)
        all_T.append(T_list)

    n = min(len(s) for s in all_S)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_S = avg(all_S)
    mean_Sh = avg(all_S_h)
    mean_U = avg(all_U)
    mean_T = avg(all_T)

    # Entropy production rate: dS/dl
    dS = [mean_S[i+1] - mean_S[i] for i in range(n-1)]
    dSh = [mean_Sh[i+1] - mean_Sh[i] for i in range(n-1)]
    dU = [mean_U[i+1] - mean_U[i] for i in range(n-1)]

    # Irreversible entropy production: dS_irr = dS - dQ/T
    # where dQ = -dU (heat released), T = logit entropy
    dS_irr = []
    for i in range(n-1):
        T = (mean_T[i] + mean_T[i+1]) / 2
        dQ = -dU[i]  # Heat released = -work
        dS_rev = dQ / T if T > 0.01 else 0
        dS_irr.append(dS[i] - dS_rev)

    # Cumulative entropy production
    cum_S_irr = np.cumsum(dS_irr).tolist()

    # MEPP test: is dS_irr maximized at each layer compared to random permutations?
    total_irr = sum(dS_irr)
    n_perm = 100
    perm_totals = []
    for _ in range(n_perm):
        perm = np.random.permutation(dS_irr)
        perm_totals.append(sum(perm))  # Same sum, but test ordering

    # Actual test: is the actual ordering producing more entropy than random orderings?
    # Compare cumulative paths
    actual_area = sum(cum_S_irr)
    perm_areas = []
    for _ in range(n_perm):
        perm = list(np.random.permutation(dS_irr))
        cum_perm = np.cumsum(perm).tolist()
        perm_areas.append(sum(cum_perm))

    mepp_percentile = float(np.mean([1 if actual_area > pa else 0 for pa in perm_areas]))

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_S': mean_S,
        'mean_Sh': mean_Sh,
        'mean_U': mean_U,
        'dS': [float(x) for x in dS],
        'dSh': [float(x) for x in dSh],
        'dS_irr': [float(x) for x in dS_irr],
        'cum_S_irr': cum_S_irr,
        'total_irr': total_irr,
        'mepp_percentile': mepp_percentile,
    }


def main():
    print("=" * 70)
    print("Phase 220: Entropy Production Rate")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_entropy_production(model, tok, device, size)
        results[size] = r
        print(f"  Total S_irr={r['total_irr']:.4f}")
        print(f"  MEPP percentile={r['mepp_percentile']:.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Logit entropy profile
    for size, r in results.items():
        axes[0, 0].plot(range(len(r['mean_S'])), r['mean_S'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('S (logit entropy)')
    axes[0, 0].set_title('(a) Entropy Profile')
    axes[0, 0].legend(fontsize=8)

    # (b) dS/dl
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['dS'])), r['dS'], '-', color=colors[size], lw=2, label=f'{size} logit')
        axes[0, 1].plot(range(len(r['dSh'])), r['dSh'], '--', color=colors[size], lw=1, alpha=0.5, label=f'{size} hidden')
    axes[0, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('dS/dLayer')
    axes[0, 1].set_title('(b) Entropy Production Rate')
    axes[0, 1].legend(fontsize=7)

    # (c) Irreversible entropy production
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['dS_irr'])), r['dS_irr'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].axhline(y=0, color='red', ls='--', alpha=0.5, label='2nd law')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('dS_irr/dLayer')
    axes[0, 2].set_title('(c) Irreversible Entropy Production')
    axes[0, 2].legend(fontsize=8)

    # (d) Cumulative irreversible entropy
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['cum_S_irr'])), r['cum_S_irr'], '-', color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Cumulative S_irr')
    axes[1, 0].set_title('(d) Cumulative Irreversible Entropy')
    axes[1, 0].legend(fontsize=8)

    # (e) Hidden state entropy
    for size, r in results.items():
        axes[1, 1].plot(range(len(r['mean_Sh'])), r['mean_Sh'], '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('S_hidden')
    axes[1, 1].set_title('(e) Hidden State Entropy')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "Entropy Production\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Total S_irr = {r['total_irr']:.4f}\n"
        summary += f"  MEPP rank   = {r['mepp_percentile']:.0%}\n\n"
    summary += "2nd Law: dS_irr >= 0?"
    for size, r in results.items():
        violations = sum(1 for x in r['dS_irr'] if x < 0)
        summary += f"\n  {size}: {violations}/{len(r['dS_irr'])} violations"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 220: Entropy Production Rate", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase220_entropy_production')
    plt.close()
    save_results('phase220_entropy_production', {'experiment': 'Entropy Production Rate', 'results': results})


if __name__ == '__main__':
    main()
