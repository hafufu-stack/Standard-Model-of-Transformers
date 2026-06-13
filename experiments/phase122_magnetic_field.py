# -*- coding: utf-8 -*-
"""
Phase 122: External Magnetic Field h (System Prompt as Magnetic Field)
Test if a strong system prompt shifts the Landau a(L) coefficients
and smooths the phase transition into a crossover.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


CONDITIONS = {
    'no_prompt': {
        'prefix': '',
        'prompts': [
            "The theory of general relativity",
            "Photosynthesis converts sunlight into",
            "Machine learning algorithms identify",
            "The periodic table organizes elements",
            "Quantum entanglement allows particles",
            "Climate models predict temperature",
        ]
    },
    'cot_prompt': {
        'prefix': "Think step by step and reason carefully about each claim. ",
        'prompts': [
            "The theory of general relativity",
            "Photosynthesis converts sunlight into",
            "Machine learning algorithms identify",
            "The periodic table organizes elements",
            "Quantum entanglement allows particles",
            "Climate models predict temperature",
        ]
    },
    'creative_prompt': {
        'prefix': "Be wildly creative, imaginative, and use vivid metaphors. ",
        'prompts': [
            "The theory of general relativity",
            "Photosynthesis converts sunlight into",
            "Machine learning algorithms identify",
            "The periodic table organizes elements",
            "Quantum entanglement allows particles",
            "Climate models predict temperature",
        ]
    },
}


def main():
    print("=" * 70)
    print("Phase 122: External Magnetic Field (System Prompt)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    condition_results = {}

    for cond_name, cond in CONDITIONS.items():
        print(f"\n--- {cond_name} ---")

        all_etas = []
        for prompt in cond['prompts']:
            full = cond['prefix'] + prompt
            inp = tok(full, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            T_vals = []
            for li in range(n_layers):
                hs = out.hidden_states[li]
                with torch.no_grad():
                    normed = model.model.norm(hs[:, -1:, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_vals.append(T if not np.isnan(T) else 0)

            etas = []
            for L in range(n_layers):
                T_subset = T_vals[:L+1]
                if len(T_subset) >= 4:
                    T_hot = max(T_subset)
                    T_cold = min(T_subset[len(T_subset)//2:])
                    if T_hot > 0.01:
                        etas.append(1.0 - T_cold / T_hot)
                    else:
                        etas.append(0.0)
                else:
                    etas.append(0.0)
            all_etas.append(etas)

        avg_etas = np.mean(all_etas, axis=0)

        # Fit sigmoid
        Ls = np.arange(4, n_layers)
        eta_fit = avg_etas[4:]
        try:
            popt, _ = curve_fit(sigmoid, Ls, eta_fit,
                                p0=[20, 0.5, np.min(eta_fit), np.max(eta_fit)],
                                maxfev=10000)
            L0_fit = popt[0]
            k_fit = popt[1]
            sig_pred = sigmoid(Ls, *popt)
            ss_res = np.sum((eta_fit - sig_pred)**2)
            ss_tot = np.sum((eta_fit - np.mean(eta_fit))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        except:
            L0_fit = 20.0
            k_fit = 0.5
            r2 = 0.0

        # Transition sharpness = k (steeper = sharper transition)
        condition_results[cond_name] = {
            'L0': float(L0_fit),
            'k': float(k_fit),
            'r2': float(r2),
            'eta_profile': [float(v) for v in avg_etas],
        }
        print(f"  L0={L0_fit:.1f}, k={k_fit:.3f}, R2={r2:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'no_prompt': '#2980b9', 'cot_prompt': '#c0392b', 'creative_prompt': '#27ae60'}

    # (a) Eta profiles
    for cond_name, res in condition_results.items():
        axes[0].plot(range(n_layers), res['eta_profile'], 'o-', markersize=3,
                    color=colors[cond_name],
                    label=f'{cond_name} ($L_0$={res["L0"]:.1f})')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('$\\eta$')
    axes[0].set_title('(a) $\\eta$ Under Different Fields')
    axes[0].legend(fontsize=8)

    # (b) L0 and k comparison
    conds = list(condition_results.keys())
    L0s = [condition_results[c]['L0'] for c in conds]
    ks = [condition_results[c]['k'] for c in conds]

    x_pos = np.arange(len(conds))
    bar_c = [colors[c] for c in conds]
    axes[1].bar(x_pos - 0.15, L0s, 0.3, color=bar_c, alpha=0.8, label='$L_0$')
    ax2 = axes[1].twinx()
    ax2.bar(x_pos + 0.15, ks, 0.3, color=bar_c, alpha=0.4, label='$k$ (sharpness)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=8)
    axes[1].set_ylabel('$L_0$')
    ax2.set_ylabel('$k$ (transition sharpness)')
    axes[1].set_title('(b) $L_0$ and Sharpness')

    # (c) Summary
    L0_shift = condition_results['cot_prompt']['L0'] - condition_results['no_prompt']['L0']
    k_change = condition_results['cot_prompt']['k'] / (condition_results['no_prompt']['k'] + 1e-10)
    creative_shift = condition_results['creative_prompt']['L0'] - condition_results['no_prompt']['L0']

    summary = (
        f"External Field Analysis\n\n"
        + "\n".join(f"{c}: L0={r['L0']:.1f}, k={r['k']:.3f}"
                    for c, r in condition_results.items())
        + f"\n\nCoT field shift: {L0_shift:+.1f} layers\n"
        f"CoT sharpness: {k_change:.2f}x\n"
        f"Creative shift: {creative_shift:+.1f} layers\n\n"
        f"Crossover: {'YES' if abs(k_change - 1) > 0.3 else 'NO'}"
    )
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('(c) Summary')

    fig.suptitle(f'Phase 122: System Prompt as Magnetic Field (CoT shift: {L0_shift:+.1f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase122_magnetic_field')
    plt.close()

    print(f"\n{'='*70}")
    for c, r in condition_results.items():
        print(f"  {c}: L0={r['L0']:.1f}, k={r['k']:.3f}")
    print(f"  CoT shift: {L0_shift:+.1f} layers")
    print(f"{'='*70}")

    save_results('phase122_magnetic_field', {
        'experiment': 'External Magnetic Field',
        'conditions': condition_results,
        'summary': {
            'L0_shift_cot': float(L0_shift),
            'k_change_cot': float(k_change),
            'L0_shift_creative': float(creative_shift),
        }
    })


if __name__ == '__main__':
    main()
