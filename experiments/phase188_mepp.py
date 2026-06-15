# -*- coding: utf-8 -*-
"""
Phase 188: Maximum Entropy Production Principle (MEPP)
=======================================================
Prigogine's principle: far-from-equilibrium systems organize to
MAXIMIZE entropy production rate under given constraints.

KEY QUESTION: Is eta=0.813 the efficiency at which entropy
              production is maximized?

We sweep noise injection (sigma) to artificially change the operating
point of the "transformer engine" and measure:
  - sigma_irr (entropy production rate)
  - eta (Carnot efficiency)
  - confidence (computational quality)

If MEPP holds, there should be a PEAK in sigma_irr near eta~0.813.
This would explain WHY the universal constants take their specific values.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure, make_safe_noise_hook

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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def measure_with_noise(model, tok, prompt, device, sigma, inject_layers=None):
    """Measure thermodynamics with noise injected at specific layers."""
    n_total = len(model.model.layers)
    if inject_layers is None:
        inject_layers = range(n_total)

    # Register noise hooks
    handles = []
    if sigma > 0:
        for li in inject_layers:
            h = model.model.layers[li].register_forward_hook(make_safe_noise_hook(sigma))
            handles.append(h)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in handles:
        h.remove()

    n_layers = len(out.hidden_states)
    U_vals, T_vals, S_vals = [], [], []

    for li in range(n_layers):
        hs = out.hidden_states[li]
        h = hs[0, -1, :].float()
        U = h.norm().item()

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()

        U_vals.append(U if not np.isnan(U) else 0)
        T_vals.append(T if not np.isnan(T) else 0)
        S_vals.append(S if not np.isnan(S) else 0)

    # Compute eta
    T_hot = np.mean(T_vals[:3])
    T_cold = np.mean(T_vals[-3:])
    eta = 1 - T_cold / (T_hot + 1e-10)

    # Compute sigma_irr (total entropy production)
    sigma_irr = 0
    for i in range(n_layers - 1):
        dS = S_vals[i + 1] - S_vals[i]
        T_avg = (T_vals[i] + T_vals[i + 1]) / 2 + 1e-10
        Q = T_avg * dS
        sigma_i = dS - Q / (T_avg + 1e-10)
        if not np.isnan(sigma_i):
            sigma_irr += abs(sigma_i)

    # Confidence
    final_logits = out.logits[0, -1, :].float()
    probs_final = torch.softmax(final_logits, dim=-1)
    conf = probs_final.max().item()

    return {
        'eta': float(eta), 'sigma_irr': float(sigma_irr),
        'conf': float(conf), 'T_hot': float(T_hot), 'T_cold': float(T_cold),
    }


def main():
    print("=" * 70)
    print("Phase 188: Maximum Entropy Production Principle")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Sweep noise levels to change operating point
    sigmas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    results_all = {}

    for sigma in sigmas:
        print(f"\n--- sigma = {sigma} ---")
        etas, sigmas_irr, confs = [], [], []

        for prompt in PROMPTS:
            r = measure_with_noise(model, tok, prompt, device, sigma)
            etas.append(r['eta'])
            sigmas_irr.append(r['sigma_irr'])
            confs.append(r['conf'])

        results_all[sigma] = {
            'mean_eta': float(np.mean(etas)),
            'mean_sigma_irr': float(np.mean(sigmas_irr)),
            'mean_conf': float(np.mean(confs)),
            'std_eta': float(np.std(etas)),
            'std_sigma_irr': float(np.std(sigmas_irr)),
        }
        print(f"  eta={np.mean(etas):.4f}, sigma_irr={np.mean(sigmas_irr):.4f}, conf={np.mean(confs):.4f}")

    # === Find MEPP peak ===
    sigma_list = sorted(results_all.keys())
    etas_list = [results_all[s]['mean_eta'] for s in sigma_list]
    sirr_list = [results_all[s]['mean_sigma_irr'] for s in sigma_list]
    conf_list = [results_all[s]['mean_conf'] for s in sigma_list]

    # Peak sigma_irr
    peak_idx = np.argmax(sirr_list)
    peak_sigma = sigma_list[peak_idx]
    peak_eta = etas_list[peak_idx]
    peak_sirr = sirr_list[peak_idx]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) sigma_irr vs noise sigma
    axes[0, 0].plot(sigma_list, sirr_list, 'o-', color='#e74c3c', markersize=8, linewidth=2)
    axes[0, 0].axvline(x=peak_sigma, color='#f39c12', linestyle='--', linewidth=2,
                        label=f'Peak at sigma={peak_sigma}')
    axes[0, 0].set_xlabel('Noise $\\sigma$')
    axes[0, 0].set_ylabel('$\\sigma_{irr}$ (Entropy Production)')
    axes[0, 0].set_title('(a) Entropy Production vs Noise')
    axes[0, 0].set_xscale('symlog', linthresh=0.001)
    axes[0, 0].legend(fontsize=8)

    # (b) eta vs noise sigma
    axes[0, 1].plot(sigma_list, etas_list, 's-', color='#3498db', markersize=8, linewidth=2)
    axes[0, 1].axhline(y=0.813, color='black', linestyle='--', alpha=0.3, label='$\\eta=0.813$')
    axes[0, 1].axvline(x=peak_sigma, color='#f39c12', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Noise $\\sigma$')
    axes[0, 1].set_ylabel('$\\eta$')
    axes[0, 1].set_title('(b) Efficiency vs Noise')
    axes[0, 1].set_xscale('symlog', linthresh=0.001)
    axes[0, 1].legend(fontsize=8)

    # (c) Confidence vs noise sigma
    axes[0, 2].plot(sigma_list, conf_list, '^-', color='#2ecc71', markersize=8, linewidth=2)
    axes[0, 2].axvline(x=peak_sigma, color='#f39c12', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('Noise $\\sigma$')
    axes[0, 2].set_ylabel('Confidence')
    axes[0, 2].set_title('(c) Computational Quality vs Noise')
    axes[0, 2].set_xscale('symlog', linthresh=0.001)

    # (d) sigma_irr vs eta (THE key plot)
    axes[1, 0].scatter(etas_list, sirr_list, c=sigma_list, cmap='hot_r', s=100,
                        edgecolors='black', zorder=5)
    axes[1, 0].scatter([peak_eta], [peak_sirr], s=200, marker='*', c='#f39c12',
                        edgecolors='black', zorder=10, label=f'MEPP peak ($\\eta$={peak_eta:.3f})')
    axes[1, 0].axvline(x=0.813, color='gray', linestyle='--', alpha=0.3, label='$\\eta=0.813$ (S-Qubit)')
    axes[1, 0].set_xlabel('Carnot Efficiency $\\eta$')
    axes[1, 0].set_ylabel('$\\sigma_{irr}$')
    axes[1, 0].set_title('(d) MEPP: $\\sigma_{irr}$ vs $\\eta$')
    axes[1, 0].legend(fontsize=7)

    # (e) sigma_irr * conf (useful work weighted by quality)
    useful = [s * c for s, c in zip(sirr_list, conf_list)]
    axes[1, 1].plot(sigma_list, useful, 'D-', color='#8e44ad', markersize=8, linewidth=2)
    peak_useful_idx = np.argmax(useful)
    axes[1, 1].axvline(x=sigma_list[peak_useful_idx], color='#f39c12', linestyle='--', linewidth=2,
                        label=f'Peak at sigma={sigma_list[peak_useful_idx]}')
    axes[1, 1].set_xlabel('Noise $\\sigma$')
    axes[1, 1].set_ylabel('$\\sigma_{irr} \\times$ Confidence')
    axes[1, 1].set_title('(e) Useful Entropy Production')
    axes[1, 1].set_xscale('symlog', linthresh=0.001)
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = (
        f"Maximum Entropy Production\n\n"
        f"MEPP PEAK:\n"
        f"  noise sigma = {peak_sigma}\n"
        f"  eta at peak = {peak_eta:.4f}\n"
        f"  sigma_irr = {peak_sirr:.4f}\n\n"
        f"BASELINE (sigma=0):\n"
        f"  eta = {etas_list[0]:.4f}\n"
        f"  sigma_irr = {sirr_list[0]:.4f}\n\n"
        f"eta=0.813 at peak: {'YES' if abs(peak_eta - 0.813) < 0.1 else 'NO'}\n"
        f"MEPP explains eta: {'POSSIBLE' if abs(peak_eta - 0.813) < 0.15 else 'UNLIKELY'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 188: Maximum Entropy Production Principle', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase188_mepp')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"MEPP peak: sigma={peak_sigma}, eta={peak_eta:.4f}, sigma_irr={peak_sirr:.4f}")
    print(f"Baseline:  sigma=0, eta={etas_list[0]:.4f}, sigma_irr={sirr_list[0]:.4f}")
    print(f"eta=0.813 at peak: {'YES' if abs(peak_eta - 0.813) < 0.1 else 'NO'}")
    print(f"{'=' * 70}")

    save_results('phase188_mepp', {
        'experiment': 'Maximum Entropy Production Principle',
        'results': {str(k): v for k, v in results_all.items()},
        'peak': {
            'sigma': float(peak_sigma),
            'eta': float(peak_eta),
            'sigma_irr': float(peak_sirr),
        }
    })


if __name__ == '__main__':
    main()
