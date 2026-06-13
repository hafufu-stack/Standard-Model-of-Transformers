# -*- coding: utf-8 -*-
"""
Phase 116: Landau Free Energy F(eta)
In Landau theory, the free energy near a 2nd order transition is:
  F(eta) = a(L-L0)*eta^2 + b*eta^4
This predicts a parabolic shape that flattens and develops two minima
at the transition. Test by computing F(eta) at each layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "The discovery of antibiotics revolutionized medicine",
]


def main():
    print("=" * 70)
    print("Phase 116: Landau Free Energy F(eta)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Collect eta at each layer for each prompt
    all_etas = []  # per-prompt eta profile

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
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

        # Compute eta at each effective depth
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

    all_etas = np.array(all_etas)

    # At each layer, compute the "Landau free energy" F(eta)
    # Use the probability distribution P(eta) at each layer
    # F(eta) = -log P(eta) (Boltzmann relation)

    # Bin eta values and compute histogram at each layer
    n_bins = 20
    eta_bins = np.linspace(0, 1, n_bins + 1)
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])

    # Select representative layers: pre, near, post transition
    test_layers = [8, 14, 18, 20, 22, 25]
    F_profiles = {}

    for ll in test_layers:
        if ll < n_layers:
            eta_vals = all_etas[:, ll]
            hist, _ = np.histogram(eta_vals, bins=eta_bins, density=True)
            hist = hist + 1e-10  # avoid log(0)
            F = -np.log(hist)
            F = F - F.min()  # shift so min is 0
            F_profiles[ll] = F

    # Fit Landau form F = a*eta^2 + b*eta^4 + c to each layer
    def landau(eta, a, b, c, d):
        return a * (eta - d)**2 + b * (eta - d)**4 + c

    fit_params = {}
    for ll, F in F_profiles.items():
        try:
            popt, _ = curve_fit(landau, eta_centers, F,
                                p0=[1, 1, 0, np.mean(all_etas[:, ll])],
                                maxfev=10000)
            fit_params[ll] = popt
        except:
            fit_params[ll] = None

    # Compute "a" coefficient vs layer (should change sign at L0)
    a_coeffs = []
    for ll in range(4, n_layers):
        eta_vals = all_etas[:, ll]
        hist, _ = np.histogram(eta_vals, bins=eta_bins, density=True)
        hist = hist + 1e-10
        F = -np.log(hist)
        F = F - F.min()
        try:
            popt, _ = curve_fit(landau, eta_centers, F,
                                p0=[1, 1, 0, np.mean(eta_vals)],
                                maxfev=5000)
            a_coeffs.append({'layer': ll, 'a': float(popt[0]), 'b': float(popt[1])})
        except:
            a_coeffs.append({'layer': ll, 'a': 0, 'b': 0})

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) F(eta) at different layers
    colors_l = plt.cm.coolwarm(np.linspace(0, 1, len(test_layers)))
    for i, (ll, F) in enumerate(F_profiles.items()):
        axes[0, 0].plot(eta_centers, F, 'o-', color=colors_l[i], markersize=3,
                       label=f'L{ll}')
        if fit_params.get(ll) is not None:
            eta_fit = np.linspace(0, 1, 100)
            axes[0, 0].plot(eta_fit, landau(eta_fit, *fit_params[ll]),
                           '--', color=colors_l[i], alpha=0.5)
    axes[0, 0].set_xlabel('$\\eta$')
    axes[0, 0].set_ylabel('$F(\\eta) = -\\ln P(\\eta)$')
    axes[0, 0].set_title('(a) Landau Free Energy')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_ylim(0, 8)

    # (b) "a" coefficient vs layer
    a_layers = [r['layer'] for r in a_coeffs]
    a_vals = [r['a'] for r in a_coeffs]
    colors_a = ['#c0392b' if a > 0 else '#2980b9' for a in a_vals]
    axes[0, 1].bar(a_layers, a_vals, color=colors_a, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$a$ coefficient')
    axes[0, 1].set_title('(b) Landau $a(L)$ (sign change = transition)')
    axes[0, 1].legend()

    # (c) Eta distribution at selected layers
    for i, ll in enumerate(test_layers):
        if ll < n_layers:
            axes[0, 2].hist(all_etas[:, ll], bins=10, alpha=0.5,
                           color=colors_l[i], label=f'L{ll}',
                           density=True)
    axes[0, 2].set_xlabel('$\\eta$')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('(c) Eta Distributions')
    axes[0, 2].legend(fontsize=7)

    # (d) Mean and std of eta vs layer
    eta_mean = np.mean(all_etas, axis=0)
    eta_std = np.std(all_etas, axis=0)
    axes[1, 0].plot(range(n_layers), eta_mean, 'o-', color='#c0392b', markersize=3)
    axes[1, 0].fill_between(range(n_layers), eta_mean - eta_std, eta_mean + eta_std,
                             alpha=0.2, color='#c0392b')
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$\\eta \\pm \\sigma$')
    axes[1, 0].set_title('(d) Order Parameter Profile')

    # (e) "b" coefficient vs layer
    b_vals = [r['b'] for r in a_coeffs]
    axes[1, 1].plot(a_layers, b_vals, 'o-', color='#27ae60', markersize=4)
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('$b$ coefficient')
    axes[1, 1].set_title('(e) Quartic Term $b(L)$')

    # (f) Summary
    # Check if a changes sign near L0
    a_sign_changes = []
    for i in range(1, len(a_vals)):
        if a_vals[i] * a_vals[i-1] < 0:
            a_sign_changes.append(a_layers[i])
    
    closest_sign = min(a_sign_changes, key=lambda x: abs(x - L0)) if a_sign_changes else -1
    
    summary = (
        f"Landau Theory Analysis\n\n"
        f"a(L) sign changes: {a_sign_changes}\n"
        f"Closest to L0: L{closest_sign}\n\n"
        f"Landau fit quality:\n"
        + "\n".join(f"  L{ll}: a={fit_params[ll][0]:.2f}, b={fit_params[ll][1]:.2f}"
                    for ll in test_layers if fit_params.get(ll) is not None)
        + f"\n\nLandau valid: {'YES' if abs(closest_sign - L0) <= 5 else 'NO'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 116: Landau Free Energy Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase116_landau')
    plt.close()

    print(f"\n{'='*70}")
    print(f"a sign changes: {a_sign_changes}")
    print(f"Closest to L0: L{closest_sign}")
    print(f"{'='*70}")

    save_results('phase116_landau', {
        'experiment': 'Landau Free Energy',
        'a_coeffs': a_coeffs,
        'summary': {
            'a_sign_changes': a_sign_changes,
            'closest_sign': closest_sign,
        }
    })


if __name__ == '__main__':
    main()
