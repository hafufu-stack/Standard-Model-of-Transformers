# -*- coding: utf-8 -*-
"""
Phase 228: Thermodynamic Uncertainty Relations
=================================================
Test the Thermodynamic Uncertainty Relation (TUR):
  Var(J) / <J>^2 >= 2 / S_irr
where J is a current (information flow) and S_irr is
irreversible entropy production.
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


def measure_tur(model, tok, device, model_name):
    """Measure TUR at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Collect per-prompt currents
    all_dT = []  # Temperature current
    all_dU = []  # Energy current
    all_dS = []  # Entropy current
    all_dP1 = [] # Order parameter current

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_l, U_l, S_l, P1_l = [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            P1_l.append(float(probs.max().item()))
            T_l.append(float(S) if not np.isnan(S) else 0)
            S_l.append(float(S) if not np.isnan(S) else 0)

        n = len(T_l)
        all_dT.append([T_l[i+1] - T_l[i] for i in range(n-1)])
        all_dU.append([U_l[i+1] - U_l[i] for i in range(n-1)])
        all_dS.append([S_l[i+1] - S_l[i] for i in range(n-1)])
        all_dP1.append([P1_l[i+1] - P1_l[i] for i in range(n-1)])

    n_trans = min(len(d) for d in all_dT)

    # TUR per layer: Var(J) / <J>^2 >= 2 / S_irr
    currents = {'dT': all_dT, 'dU': all_dU, 'dS': all_dS, 'dP1': all_dP1}
    tur_results = {}

    for curr_name, curr_data in currents.items():
        tur_lhs = []  # Var(J) / <J>^2
        tur_rhs = []  # 2 / S_irr (approximated)
        tur_ratio = []

        for l in range(n_trans):
            J_samples = [curr_data[p][l] for p in range(len(PROMPTS))]
            mean_J = np.mean(J_samples)
            var_J = np.var(J_samples)

            # S_irr at this layer (from dS data)
            dS_samples = [all_dS[p][l] for p in range(len(PROMPTS))]
            S_irr = abs(np.mean(dS_samples))

            if abs(mean_J) > 1e-10:
                lhs = var_J / (mean_J ** 2)
            else:
                lhs = float('inf')

            if S_irr > 1e-10:
                rhs = 2.0 / S_irr
            else:
                rhs = float('inf')

            tur_lhs.append(min(lhs, 1e6))
            tur_rhs.append(min(rhs, 1e6))

            if rhs > 0 and rhs < 1e6:
                tur_ratio.append(lhs / rhs)
            else:
                tur_ratio.append(1)

        tur_results[curr_name] = {
            'lhs': tur_lhs,
            'rhs': tur_rhs,
            'ratio': tur_ratio,
            'satisfied': sum(1 for r in tur_ratio if r >= 1) / len(tur_ratio) if tur_ratio else 0,
        }

    return {
        'model': model_name,
        'n_layers': n_layers,
        'tur_results': tur_results,
    }


def main():
    print("=" * 70)
    print("Phase 228: Thermodynamic Uncertainty Relations")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_tur(model, tok, device, size)
        results[size] = r
        for curr, tr in r['tur_results'].items():
            print(f"  {curr}: TUR satisfied {tr['satisfied']:.0%}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    curr_names = ['dT', 'dU', 'dS', 'dP1']

    # (a-d) TUR ratio per current
    for ci, curr in enumerate(curr_names):
        ax = axes[ci // 3, ci % 3]
        for size, r in results.items():
            tr = r['tur_results'][curr]
            ax.plot(range(len(tr['ratio'])), tr['ratio'],
                   '-o', color=colors[size], lw=1.5, markersize=3, label=size)
        ax.axhline(y=1, color='green', ls='--', alpha=0.7, label='TUR bound')
        ax.set_xlabel('Layer')
        ax.set_ylabel('LHS/RHS')
        ax.set_title(f'({chr(97+ci)}) TUR: {curr}')
        ax.legend(fontsize=7)
        ax.set_ylim(-0.5, 10)

    # (e) Summary bar chart
    ax = axes[1, 1]
    curr_labels = curr_names
    x = np.arange(len(curr_labels))
    width = 0.35
    for si, (size, r) in enumerate(results.items()):
        sats = [r['tur_results'][c]['satisfied'] for c in curr_labels]
        ax.bar(x + si * width, sats, width, label=size, color=colors[size], alpha=0.7)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(curr_labels)
    ax.set_ylabel('Fraction Satisfied')
    ax.set_title('(e) TUR Satisfaction Rate')
    ax.legend(fontsize=8)
    ax.axhline(y=1, color='green', ls='--', alpha=0.3)

    # (f) Summary text
    summary = "TUR Analysis\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        for curr, tr in r['tur_results'].items():
            summary += f"  {curr}: {tr['satisfied']:.0%} satisfied\n"
        summary += "\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 228: Thermodynamic Uncertainty Relations", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase228_tur')
    plt.close()
    save_results('phase228_tur', {'experiment': 'TUR', 'results': results})


if __name__ == '__main__':
    main()
