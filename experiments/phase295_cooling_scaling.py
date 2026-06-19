# -*- coding: utf-8 -*-
"""
Phase 295: Cooling Exponent Scaling Law
=========================================
gamma scales: 0.5B=0.15, 1.5B=0.51, 7B=0.79
Does gamma -> 1.0 as model grows? gamma=1 = Newton's cooling law.
This would mean large models are "Newtonian" coolers.
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
    "The laws of thermodynamics govern all energy transformations",
    "The chemical composition of water molecules is",
]


def measure_cooling_exponent(model, tok, prompts, device):
    """Measure cooling exponent: T(l) ~ l^(-gamma)"""
    all_temps = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        temps = []
        for h in out.hidden_states:
            t = h[0, -1, :].float().std().item()
            temps.append(t)
        all_temps.append(temps)

    # Average temperature profile
    n = len(all_temps[0])
    avg_temps = [float(np.mean([t[i] for t in all_temps])) for i in range(n)]

    # Fit: log(T) = -gamma * log(l) + const
    layers = np.arange(1, n)
    log_l = np.log(layers)
    log_t = np.log([avg_temps[i] for i in range(1, n)])

    slope, intercept, r, p, se = stats.linregress(log_l, log_t)
    gamma = -slope  # gamma = -d(log T)/d(log l)

    # Also fit exponential: T(l) = T0 * exp(-alpha * l)
    try:
        def exp_model(l, T0, alpha):
            return T0 * np.exp(-alpha * l)
        popt, pcov = optimize.curve_fit(exp_model, layers, [avg_temps[i] for i in range(1, n)],
                                        p0=[avg_temps[1], 0.01], maxfev=5000)
        T0_exp, alpha_exp = popt
        residuals = np.array([avg_temps[i] for i in range(1, n)]) - exp_model(layers, *popt)
        r2_exp = 1 - np.sum(residuals**2) / np.sum((np.array([avg_temps[i] for i in range(1, n)]) - np.mean([avg_temps[i] for i in range(1, n)]))**2)
    except:
        T0_exp, alpha_exp, r2_exp = 0, 0, 0

    return {
        'avg_temps': [round(t, 4) for t in avg_temps],
        'gamma_power': round(float(gamma), 4),
        'R2_power': round(float(r**2), 4),
        'T0_exp': round(float(T0_exp), 4),
        'alpha_exp': round(float(alpha_exp), 4),
        'R2_exp': round(float(r2_exp), 4),
        'better_fit': 'power' if r**2 > r2_exp else 'exponential',
    }


def main():
    print("=" * 70)
    print("Phase 295: Cooling Exponent Scaling Law")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}
    param_counts = {'0.5B': 5e8, '1.5B': 1.5e9, '7B': 7e9}

    for size in ['0.5B', '1.5B', '7B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        result = measure_cooling_exponent(model, tok, PROMPTS, device)
        result['n_params'] = param_counts[size]
        all_results[size] = result

        print(f"  gamma (power law) = {result['gamma_power']:.4f} (R2={result['R2_power']:.4f})")
        print(f"  alpha (exponential) = {result['alpha_exp']:.4f} (R2={result['R2_exp']:.4f})")
        print(f"  Better fit: {result['better_fit']}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Scaling Analysis ===
    sizes = list(all_results.keys())
    gammas = [all_results[s]['gamma_power'] for s in sizes]
    params = [all_results[s]['n_params'] for s in sizes]

    # Fit: gamma ~ log(N)
    log_params = np.log(params)
    slope_g, int_g, r_g, p_g, _ = stats.linregress(log_params, gammas)

    # Predict gamma(70B)
    gamma_70b = int_g + slope_g * np.log(7e10)

    scaling = {
        'gamma_vs_logN_slope': round(float(slope_g), 4),
        'R2': round(float(r_g**2), 4),
        'predicted_gamma_70B': round(float(gamma_70b), 4),
        'converges_to_1': gamma_70b > 0.9,
    }
    print(f"\n--- Scaling ---")
    print(f"  gamma ~ {slope_g:.4f} * log(N) + {int_g:.4f}")
    print(f"  R2 = {r_g**2:.4f}")
    print(f"  Predicted gamma(70B) = {gamma_70b:.4f}")
    print(f"  Converges to 1.0: {gamma_70b > 0.9}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) Temperature profiles
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_temps'], 'o-', color=colors[size], lw=2, markersize=3, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Temperature (std)')
    axes[0, 0].set_title('(a) Cooling Curves', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Log-log: T vs l
    for size, data in all_results.items():
        n = len(data['avg_temps'])
        axes[0, 1].loglog(range(1, n), data['avg_temps'][1:], 'o-', color=colors[size],
                         lw=2, markersize=3, label=f"{size} (g={data['gamma_power']:.2f})")
    axes[0, 1].set_xlabel('Layer (log)'); axes[0, 1].set_ylabel('Temperature (log)')
    axes[0, 1].set_title('(b) Power Law: T ~ l^(-gamma)', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) gamma vs model size
    axes[0, 2].semilogx(params, gammas, 'o-', color='#9b59b6', lw=2, markersize=10)
    axes[0, 2].axhline(1.0, color='gold', ls='--', lw=2, label='gamma=1 (Newton)')
    # Extrapolation
    extrap_params = np.logspace(np.log10(5e8), np.log10(7e10), 50)
    extrap_gamma = int_g + slope_g * np.log(extrap_params)
    axes[0, 2].semilogx(extrap_params, extrap_gamma, '--', color='grey', alpha=0.5)
    for i, s in enumerate(sizes):
        axes[0, 2].annotate(f'{s}\ng={gammas[i]:.2f}', (params[i], gammas[i]),
                           textcoords="offset points", xytext=(10, 5), fontsize=8)
    axes[0, 2].set_xlabel('Parameters')
    axes[0, 2].set_ylabel('Cooling Exponent gamma')
    axes[0, 2].set_title('(c) gamma Scaling', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Power vs exponential R2
    x = np.arange(len(sizes))
    w = 0.35
    axes[1, 0].bar(x - w/2, [all_results[s]['R2_power'] for s in sizes], w,
                  label='Power Law', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['R2_exp'] for s in sizes], w,
                  label='Exponential', color='#e74c3c')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_ylabel('R2'); axes[1, 0].set_title('(d) Fit Quality', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Residuals
    for size, data in all_results.items():
        n = len(data['avg_temps'])
        layers = np.arange(1, n)
        pred_power = data['avg_temps'][1] * (layers / 1.0) ** (-data['gamma_power'])
        resid = np.array(data['avg_temps'][1:]) - pred_power
        axes[1, 1].plot(layers, resid, '-', color=colors[size], lw=1.5, label=size)
    axes[1, 1].axhline(0, color='black', ls='-', lw=0.5)
    axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('(e) Power Law Residuals', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "COOLING EXPONENT SCALING\n\n"
    for s in sizes:
        txt += f"{s}: gamma = {all_results[s]['gamma_power']:.3f}\n"
    txt += f"\ngamma ~ {slope_g:.3f}*log(N){int_g:+.3f}\n"
    txt += f"R2 = {r_g**2:.4f}\n\n"
    txt += f"Predicted gamma(70B) = {gamma_70b:.3f}\n"
    txt += f"gamma -> 1.0: {'YES' if gamma_70b > 0.9 else 'NO'}\n\n"
    txt += "gamma=1: Newton cooling\n"
    txt += "gamma<1: anomalous cooling"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 295: Cooling Exponent Scaling -- Does gamma -> 1?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase295_cooling_scaling')
    plt.close()

    save_results('phase295_cooling_scaling', {
        'experiment': 'Cooling Exponent Scaling Law',
        'results': all_results,
        'scaling': scaling,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
