# -*- coding: utf-8 -*-
"""
Phase 247: Landau Free Energy Functional
==========================================
Construct the Landau free energy F(phi) = a*phi^2 + b*phi^4
where phi = P1 (order parameter). 
Fit Landau coefficients at each layer and track how they change.
Phase transition = sign change of 'a' coefficient.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize, stats
from utils import load_model, save_results, save_figure

# Need many prompts to build distribution of P1 at each layer
PROMPTS = [
    "The capital of France is", "Water freezes at zero degrees",
    "One plus one equals", "The sky is blue because",
    "Shakespeare wrote many plays including",
    "Quantum mechanics describes atomic behavior",
    "The speed of light is constant",
    "DNA stores genetic information using",
    "Neural networks learn from data through",
    "Evolution works through natural selection",
    "The periodic table organizes elements by",
    "Black holes have strong gravitational pull",
    "Photosynthesis converts light into energy",
    "Stars form from molecular gas clouds",
    "The brain processes information through neurons",
    "Entropy measures disorder in a system",
    "General relativity describes spacetime curvature",
    "Machine learning finds patterns in data",
    "Protein folding determines biological function",
    "The Higgs boson gives particles mass",
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
    "Seven abstract thoughts collided creating new",
    "Silence tasted exactly like growing uncertainty",
    "The moon decided to become a dancer",
    "Random words create unpredictable sequences when",
    "Colorless green ideas sleep furiously in",
    "The square circle computed its own radius",
    "Time traveled backward through forgotten memories of",
    "Abstract nonsense generates high entropy distributions for",
]


def landau_analysis(model, tok, device, model_name):
    """Construct Landau free energy at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Collect P1 distribution at each layer across prompts
    P1_by_layer = [[] for _ in range(n_layers + 1)]
    T_by_layer = [[] for _ in range(n_layers + 1)]
    
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for li, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            P1_by_layer[li].append(P1)
            T_by_layer[li].append(float(S) if not np.isnan(S) else 0)

    # Fit Landau free energy F(phi) = a*phi^2 + b*phi^4
    # We use the histogram of P1 to estimate the effective potential
    landau_coeffs = []
    for li in range(len(P1_by_layer)):
        p1_vals = np.array(P1_by_layer[li])
        if len(p1_vals) < 5:
            landau_coeffs.append({'a': 0, 'b': 0, 'phi_min': 0})
            continue
        
        # Effective potential from probability distribution
        # P(phi) ~ exp(-F(phi)/kT) => F(phi) ~ -kT * ln(P(phi))
        # Use histogram
        hist, edges = np.histogram(p1_vals, bins=10, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Avoid log(0)
        mask = hist > 0
        if mask.sum() < 4:
            landau_coeffs.append({'a': 0, 'b': 0, 'phi_min': 0})
            continue
        
        F_eff = -np.log(hist[mask] + 1e-10)
        phi = centers[mask]
        
        # Fit F = a*phi^2 + b*phi^4 + c
        try:
            def landau_model(x, a, b, c):
                return a * x**2 + b * x**4 + c
            popt, _ = optimize.curve_fit(landau_model, phi, F_eff, 
                                         p0=[1, 1, 0], maxfev=5000)
            a_coeff, b_coeff = popt[0], popt[1]
            # Minimum: phi_min = sqrt(-a/(2b)) if a < 0 and b > 0
            if a_coeff < 0 and b_coeff > 0:
                phi_min = float(np.sqrt(-a_coeff / (2 * b_coeff)))
            else:
                phi_min = 0.0
        except Exception:
            a_coeff, b_coeff, phi_min = 0, 0, 0

        landau_coeffs.append({
            'a': float(a_coeff), 'b': float(b_coeff),
            'phi_min': phi_min,
            'layer': li,
        })

    # Find phase transition: where 'a' changes sign
    a_vals = [lc['a'] for lc in landau_coeffs]
    transition_layers = []
    for i in range(len(a_vals) - 1):
        if a_vals[i] * a_vals[i+1] < 0:
            transition_layers.append(i)

    # Mean T at each layer
    mean_T = [float(np.mean(T_by_layer[li])) for li in range(len(T_by_layer))]
    mean_P1 = [float(np.mean(P1_by_layer[li])) for li in range(len(P1_by_layer))]
    std_P1 = [float(np.std(P1_by_layer[li])) for li in range(len(P1_by_layer))]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'landau_coeffs': landau_coeffs,
        'transition_layers': transition_layers,
        'a_vals': a_vals,
        'mean_T': mean_T,
        'mean_P1': mean_P1,
        'std_P1': std_P1,
    }


def main():
    print("=" * 70)
    print("Phase 247: Landau Free Energy Functional")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = landau_analysis(model, tok, device, size)
        results[size] = r
        print(f"  Transition layers: {r['transition_layers']}")
        print(f"  a-coefficients range: {min(r['a_vals']):.3f} to {max(r['a_vals']):.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Landau 'a' coefficient vs layer
    for size, r in results.items():
        axes[0, 0].plot(range(len(r['a_vals'])), r['a_vals'], '-o',
                       color=colors[size], lw=2, markersize=3, label=size)
        for tl in r['transition_layers']:
            axes[0, 0].axvline(x=tl, color=colors[size], ls='--', alpha=0.5)
    axes[0, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Landau a')
    axes[0, 0].set_title('(a) Landau a-coefficient')
    axes[0, 0].legend(fontsize=8)

    # (b) Landau 'b' coefficient vs layer
    for size, r in results.items():
        b_vals = [lc['b'] for lc in r['landau_coeffs']]
        axes[0, 1].plot(range(len(b_vals)), b_vals, '-o',
                       color=colors[size], lw=2, markersize=3, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Landau b')
    axes[0, 1].set_title('(b) Landau b-coefficient')
    axes[0, 1].legend(fontsize=8)

    # (c) phi_min (order parameter equilibrium value)
    for size, r in results.items():
        phi_min = [lc['phi_min'] for lc in r['landau_coeffs']]
        axes[0, 2].plot(range(len(phi_min)), phi_min, '-o',
                       color=colors[size], lw=2, markersize=3, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('phi_min')
    axes[0, 2].set_title('(c) Equilibrium Order Parameter')
    axes[0, 2].legend(fontsize=8)

    # (d) P1 distribution width (susceptibility proxy)
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['std_P1'])), r['std_P1'], '-o',
                       color=colors[size], lw=2, markersize=3, label=size)
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('std(P1)')
    axes[1, 0].set_title('(d) P1 Fluctuation (Susceptibility)')
    axes[1, 0].legend(fontsize=8)

    # (e) Mean P1 profile
    for size, r in results.items():
        axes[1, 1].plot(range(len(r['mean_P1'])), r['mean_P1'], '-',
                       color=colors[size], lw=2, label=f'P1 ({size})')
        axes[1, 1].fill_between(range(len(r['mean_P1'])),
                                np.array(r['mean_P1']) - np.array(r['std_P1']),
                                np.array(r['mean_P1']) + np.array(r['std_P1']),
                                color=colors[size], alpha=0.2)
    axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('P1')
    axes[1, 1].set_title('(e) P1 Profile with Fluctuations')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "LANDAU FREE ENERGY\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Transitions at layers: {r['transition_layers']}\n"
        summary += f"  a range: [{min(r['a_vals']):.2f}, {max(r['a_vals']):.2f}]\n"
        summary += f"  Final P1: {r['mean_P1'][-1]:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 247: Landau Free Energy Functional",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase247_landau')
    plt.close()
    save_results('phase247_landau', {'experiment': 'Landau Free Energy', 'results': results})


if __name__ == '__main__':
    main()
