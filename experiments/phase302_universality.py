# -*- coding: utf-8 -*-
"""
Phase 302: Universality Class Identification
==============================================
What universality class does the transformer phase transition belong to?
Test against known classes:
- Ising (Z2 symmetry): critical exponents nu=1, beta=1/8, gamma=7/4
- Mean-field: nu=1/2, beta=1/2, gamma=1
- XY model: nu=0.67, beta=0.35
Measure critical exponents from the T-PR phase transition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
    "The chemical composition of water molecules is",
    "Artificial intelligence will transform how we live and work",
    "The periodic table organizes chemical elements by their properties",
    "Quantum computing uses qubits that can exist in multiple states",
]


def measure_critical_exponents(model, tok, prompts, device):
    """Extract critical exponents from the transformer phase transition."""
    all_T = []
    all_PR = []
    all_P1 = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(len(out.hidden_states)):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            T = float(np.std(h))
            h_sq = h ** 2
            h_p = h_sq / (np.sum(h_sq) + 1e-15)
            PR = float(1.0 / (np.sum(h_p ** 2) + 1e-15))
            P1 = float(np.max(h_p))
            all_T.append(T)
            all_PR.append(PR)
            all_P1.append(P1)

    T_arr = np.array(all_T)
    PR_arr = np.array(all_PR)
    P1_arr = np.array(all_P1)

    # Find critical temperature (where PR derivative is maximum)
    # Sort by T
    sort_idx = np.argsort(T_arr)
    T_sorted = T_arr[sort_idx]
    PR_sorted = PR_arr[sort_idx]
    P1_sorted = P1_arr[sort_idx]

    # Smooth for derivative
    from scipy.ndimage import uniform_filter1d
    PR_smooth = uniform_filter1d(PR_sorted, size=max(5, len(PR_sorted)//20))
    dPR_dT = np.gradient(PR_smooth, T_sorted)
    T_c_idx = np.argmax(np.abs(dPR_dT))
    T_c = float(T_sorted[T_c_idx])

    # Reduced temperature t = (T - T_c) / T_c
    t = (T_arr - T_c) / (T_c + 1e-10)

    # Order parameter M = P1 (or could be 1/PR)
    # Near T_c: M ~ |t|^beta
    # Susceptibility chi ~ |t|^(-gamma_crit)
    # Correlation length xi ~ |t|^(-nu)

    # Measure beta: M vs |t| for T < T_c
    below_mask = (t < 0) & (t > -0.9)
    if below_mask.sum() > 5:
        log_t_below = np.log(np.abs(t[below_mask]) + 1e-10)
        log_M_below = np.log(P1_arr[below_mask] + 1e-10)
        slope_beta, _, r_beta, _, _ = stats.linregress(log_t_below, log_M_below)
        beta_exp = float(slope_beta)
        r2_beta = float(r_beta**2)
    else:
        beta_exp = 0
        r2_beta = 0

    # Measure gamma_crit: chi ~ 1/(PR) as susceptibility
    # chi diverges at T_c
    chi = 1.0 / (PR_arr + 1)
    above_mask = (t > 0.01) & (t < 0.9)
    if above_mask.sum() > 5:
        log_t_above = np.log(np.abs(t[above_mask]) + 1e-10)
        log_chi = np.log(chi[above_mask] + 1e-10)
        slope_gamma, _, r_gamma, _, _ = stats.linregress(log_t_above, log_chi)
        gamma_exp = float(-slope_gamma)
        r2_gamma = float(r_gamma**2)
    else:
        gamma_exp = 0
        r2_gamma = 0

    # Classify universality class
    classes = {
        'Mean-field': {'beta': 0.5, 'gamma': 1.0},
        '2D Ising': {'beta': 0.125, 'gamma': 1.75},
        '3D Ising': {'beta': 0.326, 'gamma': 1.237},
        '3D XY': {'beta': 0.348, 'gamma': 1.316},
        '3D Heisenberg': {'beta': 0.365, 'gamma': 1.386},
    }

    best_class = 'Unknown'
    best_dist = float('inf')
    for name, exps in classes.items():
        dist = abs(beta_exp - exps['beta']) + abs(gamma_exp - exps['gamma'])
        if dist < best_dist:
            best_dist = dist
            best_class = name

    return {
        'T_c': round(T_c, 4),
        'beta': round(beta_exp, 4),
        'gamma_crit': round(gamma_exp, 4),
        'r2_beta': round(r2_beta, 4),
        'r2_gamma': round(r2_gamma, 4),
        'best_class': best_class,
        'best_dist': round(best_dist, 4),
        'n_points': len(T_arr),
    }


def main():
    print("=" * 70)
    print("Phase 302: Universality Class Identification")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        result = measure_critical_exponents(model, tok, PROMPTS, device)
        all_results[size] = result

        print(f"  T_c = {result['T_c']:.4f}")
        print(f"  beta = {result['beta']:.4f} (R2={result['r2_beta']:.4f})")
        print(f"  gamma = {result['gamma_crit']:.4f} (R2={result['r2_gamma']:.4f})")
        print(f"  Best match: {result['best_class']} (dist={result['best_dist']:.3f})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Beta exponents comparison
    classes = {
        'Mean-field': {'beta': 0.5, 'gamma': 1.0},
        '2D Ising': {'beta': 0.125, 'gamma': 1.75},
        '3D Ising': {'beta': 0.326, 'gamma': 1.237},
        '3D XY': {'beta': 0.348, 'gamma': 1.316},
        '3D Heisenberg': {'beta': 0.365, 'gamma': 1.386},
    }
    class_names = list(classes.keys())
    betas = [classes[c]['beta'] for c in class_names]
    axes[0, 0].barh(class_names, betas, color='lightgrey', alpha=0.7, label='Theory')
    for size, data in all_results.items():
        axes[0, 0].axvline(data['beta'], color=colors[size], ls='--', lw=2, label=f'{size}: {data["beta"]:.3f}')
    axes[0, 0].set_xlabel('beta exponent')
    axes[0, 0].set_title('(a) Order Parameter Exponent beta', fontweight='bold')
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    # (b) Gamma exponents
    gammas_class = [classes[c]['gamma'] for c in class_names]
    axes[0, 1].barh(class_names, gammas_class, color='lightgrey', alpha=0.7, label='Theory')
    for size, data in all_results.items():
        axes[0, 1].axvline(data['gamma_crit'], color=colors[size], ls='--', lw=2, label=f'{size}: {data["gamma_crit"]:.3f}')
    axes[0, 1].set_xlabel('gamma exponent')
    axes[0, 1].set_title('(b) Susceptibility Exponent gamma', fontweight='bold')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    # (c) Phase space: beta vs gamma
    for name, exps in classes.items():
        axes[0, 2].plot(exps['beta'], exps['gamma'], 'o', markersize=8, label=name)
    for size, data in all_results.items():
        axes[0, 2].plot(data['beta'], data['gamma_crit'], '*', markersize=15,
                       color=colors[size], label=f'{size} (measured)')
    axes[0, 2].set_xlabel('beta')
    axes[0, 2].set_ylabel('gamma')
    axes[0, 2].set_title('(c) Universality Class Map', fontweight='bold')
    axes[0, 2].legend(fontsize=7); axes[0, 2].grid(alpha=0.3)

    # (d) Best class match
    sizes = list(all_results.keys())
    best_classes = [all_results[s]['best_class'] for s in sizes]
    dists = [all_results[s]['best_dist'] for s in sizes]
    axes[1, 0].bar(sizes, dists, color=[colors[s] for s in sizes])
    for i, s in enumerate(sizes):
        axes[1, 0].text(i, dists[i] + 0.01, best_classes[i], ha='center', fontsize=8)
    axes[1, 0].set_ylabel('Distance to Nearest Class')
    axes[1, 0].set_title('(d) Best Match', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) R2 values
    x = np.arange(len(sizes))
    w = 0.35
    axes[1, 1].bar(x - w/2, [all_results[s]['r2_beta'] for s in sizes], w,
                  label='beta R2', color='#3498db')
    axes[1, 1].bar(x + w/2, [all_results[s]['r2_gamma'] for s in sizes], w,
                  label='gamma R2', color='#e74c3c')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(sizes)
    axes[1, 1].set_ylabel('R2')
    axes[1, 1].set_title('(e) Fit Quality', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "UNIVERSALITY CLASS\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  beta = {d['beta']:.3f}\n"
        txt += f"  gamma = {d['gamma_crit']:.3f}\n"
        txt += f"  Class: {d['best_class']}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 302: Universality Class Identification",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase302_universality')
    plt.close()

    save_results('phase302_universality', {
        'experiment': 'Universality Class Identification',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
