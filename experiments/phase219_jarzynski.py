# -*- coding: utf-8 -*-
"""
Phase 219: Jarzynski Equality Test
=====================================
Test if the Jarzynski equality holds for transformer forward passes.
In non-equilibrium stat mech: <exp(-W/kT)> = exp(-dF/kT)
Here W = work done on hidden state by each layer, T = logit entropy.
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


def measure_jarzynski(model, tok, device, model_name):
    """Measure Jarzynski equality components layer-by-layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Collect per-prompt, per-layer: W (work), T (temperature), dF (free energy change)
    all_W = []  # Work = dU per layer
    all_T = []  # Temperature
    all_F = []  # Free energy F = U - TS

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        W_list, T_list, F_list = [], [], []
        prev_U = None
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = h.norm().item()
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T = S  # T ~ entropy in our framework
            F = U - T * S if T > 0 else U  # F = U - TS

            if prev_U is not None:
                W_list.append(U - prev_U)  # Work = delta U
            T_list.append(T)
            F_list.append(F)
            prev_U = U

        all_W.append(W_list)
        all_T.append(T_list)
        all_F.append(F_list)

    n_transitions = min(len(w) for w in all_W)
    n_states = min(len(t) for t in all_T)

    # Layer-by-layer Jarzynski test
    jarzynski_lhs = []  # <exp(-W/T)>
    jarzynski_rhs = []  # exp(-dF/T)
    jarzynski_ratio = []

    for l in range(n_transitions):
        W_samples = [all_W[p][l] for p in range(len(PROMPTS))]
        T_mean = np.mean([all_T[p][l+1] for p in range(len(PROMPTS))])
        dF = np.mean([all_F[p][l+1] - all_F[p][l] for p in range(len(PROMPTS))])

        if T_mean > 0.01:
            # LHS: <exp(-W/T)>
            exp_terms = [np.exp(-w / T_mean) for w in W_samples]
            lhs = float(np.mean(exp_terms))
            # RHS: exp(-dF/T)
            rhs = float(np.exp(-dF / T_mean))
            jarzynski_lhs.append(lhs)
            jarzynski_rhs.append(rhs)
            jarzynski_ratio.append(lhs / rhs if rhs > 0 else float('inf'))
        else:
            jarzynski_lhs.append(0)
            jarzynski_rhs.append(0)
            jarzynski_ratio.append(1)

    # Overall Jarzynski test: cumulative work
    cum_W = [sum(all_W[p][l] for l in range(n_transitions)) for p in range(len(PROMPTS))]
    T_final = np.mean([all_T[p][-1] for p in range(len(PROMPTS))])
    dF_total = np.mean([all_F[p][-1] - all_F[p][0] for p in range(len(PROMPTS))])

    if T_final > 0.01:
        global_lhs = float(np.mean([np.exp(-w / T_final) for w in cum_W]))
        global_rhs = float(np.exp(-dF_total / T_final))
    else:
        global_lhs, global_rhs = 0, 0

    # Crooks-like forward/reverse asymmetry
    W_forward = [float(np.mean([all_W[p][l] for p in range(len(PROMPTS))])) for l in range(n_transitions)]
    W_reverse = list(reversed(W_forward))
    crooks_asymmetry = [W_forward[i] + W_reverse[i] for i in range(n_transitions)]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'jarzynski_lhs': jarzynski_lhs,
        'jarzynski_rhs': jarzynski_rhs,
        'jarzynski_ratio': jarzynski_ratio,
        'global_lhs': global_lhs,
        'global_rhs': global_rhs,
        'global_ratio': global_lhs / global_rhs if global_rhs > 0 else float('inf'),
        'W_forward': W_forward,
        'crooks_asymmetry': crooks_asymmetry,
        'dF_total': dF_total,
        'T_final': T_final,
    }


def main():
    print("=" * 70)
    print("Phase 219: Jarzynski Equality Test")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_jarzynski(model, tok, device, size)
        results[size] = r
        print(f"  Global Jarzynski: LHS={r['global_lhs']:.4f}, RHS={r['global_rhs']:.4f}, "
              f"ratio={r['global_ratio']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, r in results.items():
        c = colors[size]
        n = len(r['jarzynski_lhs'])
        # (a) LHS vs RHS layer by layer
        axes[0, 0].plot(range(n), r['jarzynski_lhs'], '-o', color=c, lw=1.5,
                       markersize=4, label=f'{size} LHS', alpha=0.8)
        axes[0, 0].plot(range(n), r['jarzynski_rhs'], '--s', color=c, lw=1.5,
                       markersize=4, alpha=0.5, label=f'{size} RHS')
    axes[0, 0].set_xlabel('Layer Transition')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('(a) Jarzynski: LHS vs RHS')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_yscale('symlog', linthresh=1)

    # (b) Jarzynski ratio
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['jarzynski_ratio'])), r['jarzynski_ratio'],
                       '-o', color=colors[size], lw=2, markersize=4, label=size)
    axes[0, 1].axhline(y=1, color='green', ls='--', alpha=0.7, label='Equality')
    axes[0, 1].set_xlabel('Layer Transition')
    axes[0, 1].set_ylabel('LHS / RHS')
    axes[0, 1].set_title('(b) Jarzynski Ratio (1 = equality)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_ylim(-0.5, 5)

    # (c) Work profile
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['W_forward'])), r['W_forward'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 2].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 2].set_xlabel('Layer Transition')
    axes[0, 2].set_ylabel('Work W = dU')
    axes[0, 2].set_title('(c) Work Profile')
    axes[0, 2].legend(fontsize=8)

    # (d) Crooks asymmetry
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['crooks_asymmetry'])), r['crooks_asymmetry'],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 0].axhline(y=0, color='green', ls='--', alpha=0.7, label='Symmetry')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('W_fwd + W_rev')
    axes[1, 0].set_title('(d) Crooks Asymmetry')
    axes[1, 0].legend(fontsize=8)

    # (e) Global summary
    labels = list(results.keys())
    lhs_vals = [results[s]['global_lhs'] for s in labels]
    rhs_vals = [results[s]['global_rhs'] for s in labels]
    x = np.arange(len(labels))
    axes[1, 1].bar(x - 0.15, lhs_vals, 0.3, label='<exp(-W/T)>', color='steelblue')
    axes[1, 1].bar(x + 0.15, rhs_vals, 0.3, label='exp(-dF/T)', color='coral')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('(e) Global Jarzynski')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary text
    summary = "Jarzynski Equality\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  <exp(-W/T)> = {r['global_lhs']:.4f}\n"
        summary += f"  exp(-dF/T)  = {r['global_rhs']:.4f}\n"
        summary += f"  Ratio       = {r['global_ratio']:.4f}\n\n"
    verdict = "HOLDS" if all(0.5 < results[s]['global_ratio'] < 2.0 for s in results) else "VIOLATED"
    summary += f"Verdict: {verdict}"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 219: Jarzynski Equality Test", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase219_jarzynski')
    plt.close()
    save_results('phase219_jarzynski', {'experiment': 'Jarzynski Equality', 'results': results})


if __name__ == '__main__':
    main()
