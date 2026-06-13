# -*- coding: utf-8 -*-
"""
Phase 118: Token-Position Dependent Phase Transition
Does the eta transition point L0 vary across token positions?
Early tokens (near BOS) vs late tokens (long context).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure

LONG_PROMPTS = [
    "The fundamental theorem of calculus states that differentiation and integration are inverse operations which means that the derivative of the integral of a function returns the original function and this relationship has profound implications for both pure mathematics and applied physics because it allows us to compute areas under curves and solve differential equations that model real world phenomena ranging from planetary motion to population growth and the heat equation describes how temperature distributions evolve",
    "In quantum mechanics the wave function provides a complete description of the quantum state of a physical system and the probability of finding a particle at a given position is determined by the square of the absolute value of the wave function and this interpretation was first proposed by Max Born in nineteen twenty six and it fundamentally changed our understanding of the microscopic world by introducing inherent randomness into the laws of physics which was deeply unsettling",
    "Neural networks are computational models inspired by the biological neural networks found in animal brains and they consist of interconnected nodes organized in layers where each connection has an associated weight that is adjusted during training through a process called backpropagation which computes the gradient of the loss function with respect to each weight and uses gradient descent to minimize the error between predicted and actual outputs enabling the network to learn complex patterns",
]


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


def main():
    print("=" * 70)
    print("Phase 118: Token-Position Dependent Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Sample token positions: early, mid, late
    position_results = {}

    for prompt in LONG_PROMPTS:
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=200).to(device)
        seq_len = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Sample positions
        positions = {
            'early (pos 2-5)': list(range(2, min(6, seq_len))),
            'mid (pos 20-30)': list(range(20, min(31, seq_len))),
            'late (pos 50+)': list(range(50, min(70, seq_len))),
        }

        for pos_name, pos_list in positions.items():
            for pos in pos_list:
                if pos >= seq_len:
                    continue

                # Compute eta at each effective depth for this token position
                T_vals = []
                for li in range(n_layers):
                    hs = out.hidden_states[li]
                    with torch.no_grad():
                        normed = model.model.norm(hs[:, pos:pos+1, :])
                        logits = model.lm_head(normed).squeeze().float()
                    probs = torch.softmax(logits, dim=-1)
                    T = -(probs * torch.log(probs + 1e-10)).sum().item()
                    T_vals.append(T if not np.isnan(T) else 0)

                if pos_name not in position_results:
                    position_results[pos_name] = []
                position_results[pos_name].append(T_vals)

    # Compute eta profiles for each position group
    eta_profiles = {}
    for pos_name, T_lists in position_results.items():
        profile = []
        for L in range(4, n_layers):
            etas = []
            for T_vals in T_lists:
                T_subset = T_vals[:L+1]
                if len(T_subset) >= 4:
                    T_hot = max(T_subset)
                    T_cold = min(T_subset[len(T_subset)//2:])
                    if T_hot > 0.01:
                        etas.append(1.0 - T_cold / T_hot)
            if etas:
                profile.append({
                    'L': L,
                    'eta': float(np.mean(etas)),
                    'std': float(np.std(etas)),
                })
        eta_profiles[pos_name] = profile

    # Fit sigmoid to each
    fit_results = {}
    for pos_name, profile in eta_profiles.items():
        Ls = np.array([r['L'] for r in profile])
        etas = np.array([r['eta'] for r in profile])
        try:
            popt, _ = curve_fit(sigmoid, Ls, etas,
                                p0=[20, 0.5, np.min(etas), np.max(etas)],
                                maxfev=10000)
            L0_fit = popt[0]
            sig_pred = sigmoid(Ls, *popt)
            ss_res = np.sum((etas - sig_pred)**2)
            ss_tot = np.sum((etas - np.mean(etas))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        except:
            L0_fit = 20.0
            r2 = 0.0

        fit_results[pos_name] = {'L0': float(L0_fit), 'r2': float(r2)}
        print(f"  {pos_name}: L0={L0_fit:.1f}, R2={r2:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'early (pos 2-5)': '#27ae60', 'mid (pos 20-30)': '#2980b9',
              'late (pos 50+)': '#c0392b'}

    # (a) Eta profiles
    for pos_name, profile in eta_profiles.items():
        Ls = [r['L'] for r in profile]
        etas = [r['eta'] for r in profile]
        stds = [r['std'] for r in profile]
        color = colors.get(pos_name, 'gray')
        L0 = fit_results[pos_name]['L0']
        axes[0].plot(Ls, etas, 'o-', color=color, markersize=3,
                    label=f'{pos_name} ($L_0={L0:.1f}$)')
        axes[0].fill_between(Ls, np.array(etas)-np.array(stds),
                             np.array(etas)+np.array(stds), alpha=0.1, color=color)
    axes[0].set_xlabel('Effective Layer Count')
    axes[0].set_ylabel('$\\eta$')
    axes[0].set_title('(a) Eta by Token Position')
    axes[0].legend(fontsize=7)

    # (b) L0 comparison
    pos_names = list(fit_results.keys())
    L0s = [fit_results[p]['L0'] for p in pos_names]
    bar_colors = [colors.get(p, 'gray') for p in pos_names]
    axes[1].bar(range(len(pos_names)), L0s, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(pos_names)))
    axes[1].set_xticklabels([p.split('(')[0].strip() for p in pos_names], fontsize=9)
    axes[1].set_ylabel('$L_0$')
    mean_L0 = np.mean(L0s)
    axes[1].axhline(y=mean_L0, color='black', linestyle='--',
                    label=f'Mean={mean_L0:.1f}')
    axes[1].set_title('(b) Transition Point by Position')
    axes[1].legend()

    # (c) Summary
    cv = np.std(L0s) / (np.mean(L0s) + 1e-10)
    summary = (
        f"Token Position Analysis\n\n"
        + "\n".join(f"{p}: L0={fit_results[p]['L0']:.1f} (R2={fit_results[p]['r2']:.3f})"
                    for p in pos_names)
        + f"\n\nMean L0: {mean_L0:.1f} +/- {np.std(L0s):.1f}\n"
        f"CV: {cv:.3f}\n\n"
        f"{'POSITION-INDEPENDENT' if cv < 0.1 else 'POSITION-DEPENDENT'}"
    )
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('(c) Summary')

    fig.suptitle(f'Phase 118: Token-Position Dependent L0 (CV={cv:.3f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase118_position')
    plt.close()

    print(f"\n{'='*70}")
    for p in pos_names:
        print(f"  {p}: L0={fit_results[p]['L0']:.1f}")
    print(f"  CV={cv:.3f}")
    print(f"{'='*70}")

    save_results('phase118_position', {
        'experiment': 'Token-Position Dependent Transition',
        'fit_results': fit_results,
        'summary': {
            'mean_L0': float(mean_L0),
            'std_L0': float(np.std(L0s)),
            'cv': float(cv),
        }
    })


if __name__ == '__main__':
    main()
