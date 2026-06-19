# -*- coding: utf-8 -*-
"""
Phase 280: Bekenstein-Hawking Bound
=====================================
Test if Hawking temperature T_H scales as 1/sqrt(N_params) or 1/N_params.
Phase 277 found T_H(0.5B)=0.33, T_H(1.5B)=0.45.
Add 7B-Instruct as a third data point.

Bekenstein-Hawking: T_H = hbar*c^3 / (8*pi*G*M*k_B) ~ 1/M
If M ~ N_params, then T_H ~ 1/N_params.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, load_any_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The fundamental laws of physics state that",
    "In the beginning of the universe",
    "The principle of conservation of energy implies",
    "Quantum field theory predicts that",
]

# Model configs: (name, load_fn_args, n_params_approx)
MODELS = [
    ('0.5B', '0.5B', 0.5e9),
    ('1.5B', '1.5B', 1.5e9),
]


def measure_hawking_temp(model, tok, prompts, device):
    """Measure temperature at each layer, fit exponential decay to extract T_H."""
    all_layer_temps = []
    for prompt in prompts:
        results, _ = measure_full_thermodynamics(model, tok, prompt, device)
        temps = [r['T'] for r in results]
        all_layer_temps.append(temps)

    # Average across prompts
    n_layers = min(len(t) for t in all_layer_temps)
    avg_temps = [np.mean([t[i] for t in all_layer_temps]) for i in range(n_layers)]

    # Fit T(l) = T0 * exp(-gamma * l/L) + T_H
    layers = np.arange(n_layers)
    norm_layers = layers / (n_layers - 1)

    # Simple fit: T_H = minimum temperature in latter half
    latter_half = avg_temps[n_layers//2:]
    T_H = float(np.min(latter_half))
    T0 = float(avg_temps[0])

    # Exponential fit
    from scipy.optimize import curve_fit
    def exp_decay(x, t0, gamma, th):
        return t0 * np.exp(-gamma * x) + th

    try:
        popt, _ = curve_fit(exp_decay, norm_layers, avg_temps,
                           p0=[T0, 1.0, T_H], maxfev=5000,
                           bounds=([0, 0, 0], [20, 50, 10]))
        T0_fit, gamma_fit, TH_fit = popt
    except Exception:
        T0_fit, gamma_fit, TH_fit = T0, 1.0, T_H

    return {
        'T0': round(float(T0_fit), 4),
        'gamma': round(float(gamma_fit), 4),
        'T_H': round(float(TH_fit), 4),
        'layer_temps': [round(float(t), 4) for t in avg_temps],
        'n_layers': n_layers,
    }


def main():
    print("=" * 70)
    print("Phase 280: Bekenstein-Hawking Bound")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for name, size, n_params in MODELS:
        print(f"\n=== {name} ({n_params/1e9:.1f}B params) ===")
        model, tok = load_model(device, size=size)
        r = measure_hawking_temp(model, tok, PROMPTS, device)
        r['n_params'] = n_params
        all_results[name] = r
        print(f"  T_H = {r['T_H']:.4f}, T0 = {r['T0']:.4f}, gamma = {r['gamma']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Also try 7B-Instruct
    try:
        print("\n=== 7B-Instruct ===")
        model, tok = load_any_model('Qwen/Qwen2.5-7B-Instruct', device=device)
        r = measure_hawking_temp(model, tok, PROMPTS, device)
        r['n_params'] = 7.0e9
        all_results['7B'] = r
        print(f"  T_H = {r['T_H']:.4f}, T0 = {r['T0']:.4f}, gamma = {r['gamma']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"  7B-Instruct skipped: {e}")

    # === Bekenstein-Hawking Scaling Analysis ===
    names = list(all_results.keys())
    n_params_arr = np.array([all_results[n]['n_params'] for n in names])
    th_arr = np.array([all_results[n]['T_H'] for n in names])

    scaling = {}
    if len(names) >= 2:
        # Test T_H ~ 1/N
        log_n = np.log(n_params_arr)
        log_th = np.log(th_arr + 1e-10)
        slope, intercept, r_val, p_val, _ = stats.linregress(log_n, log_th)
        scaling['power_law_exponent'] = round(float(slope), 4)
        scaling['R2'] = round(float(r_val**2), 4)
        scaling['p_value'] = round(float(p_val), 6)
        # BH predicts slope = -1
        scaling['BH_consistent'] = abs(slope + 1) < 0.5
        print(f"\n  Scaling: T_H ~ N^{slope:.3f} (BH predicts N^-1)")
        print(f"  R2={r_val**2:.4f}, p={p_val:.6f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) Temperature profiles
    for name, r in all_results.items():
        axes[0, 0].plot(r['layer_temps'], '-', color=colors.get(name, '#999'),
                       lw=2, label=f"{name} (T_H={r['T_H']:.3f})")
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) Temperature Profiles', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) T_H vs N_params (log-log)
    axes[0, 1].scatter([r['n_params']/1e9 for r in all_results.values()],
                      [r['T_H'] for r in all_results.values()],
                      c=[colors.get(n, '#999') for n in names], s=100, zorder=5)
    for n in names:
        axes[0, 1].annotate(n, (all_results[n]['n_params']/1e9, all_results[n]['T_H']),
                           textcoords="offset points", xytext=(10, 5))
    axes[0, 1].set_xlabel('N_params (Billions)')
    axes[0, 1].set_ylabel('Hawking Temperature T_H')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('(b) T_H vs Model Size', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # (c) Log-log with fit
    if len(names) >= 2:
        axes[0, 2].scatter(log_n, log_th, c=[colors.get(n, '#999') for n in names],
                          s=100, zorder=5)
        x_fit = np.linspace(min(log_n), max(log_n), 50)
        axes[0, 2].plot(x_fit, slope * x_fit + intercept, 'k--',
                       label=f"slope={slope:.2f} (BH=-1)")
        axes[0, 2].set_xlabel('log(N_params)')
        axes[0, 2].set_ylabel('log(T_H)')
        axes[0, 2].set_title('(c) Power Law Fit', fontweight='bold')
        axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Gamma (decay rate) vs size
    axes[1, 0].bar(names, [all_results[n]['gamma'] for n in names],
                  color=[colors.get(n, '#999') for n in names])
    axes[1, 0].set_ylabel('Decay Rate gamma')
    axes[1, 0].set_title('(d) Cooling Rate vs Model Size', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) T0 (initial temp) vs size
    axes[1, 1].bar(names, [all_results[n]['T0'] for n in names],
                  color=[colors.get(n, '#999') for n in names])
    axes[1, 1].set_ylabel('Initial Temperature T0')
    axes[1, 1].set_title('(e) Initial Temperature vs Size', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "BEKENSTEIN-HAWKING BOUND\n"
    txt += "BH: T_H ~ 1/M (slope = -1)\n\n"
    for n in names:
        txt += f"{n}: T_H = {all_results[n]['T_H']:.4f}\n"
    if scaling:
        txt += f"\nMeasured slope: {scaling['power_law_exponent']:.3f}\n"
        txt += f"R2: {scaling['R2']:.4f}\n"
        txt += f"BH consistent: {scaling['BH_consistent']}\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 280: Bekenstein-Hawking Bound",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase280_bekenstein_hawking')
    plt.close()

    save_results('phase280_bekenstein_hawking', {
        'experiment': 'Bekenstein-Hawking Bound',
        'results': all_results,
        'scaling': scaling,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
