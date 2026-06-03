# -*- coding: utf-8 -*-
"""
Phase 31: dU/dT Scaling Law (Opus Original)
=============================================
Quantify how dU/dT scales with model dimension d.
0.5B (d=896) vs 1.5B (d=1536): determine the power law exponent.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, measure_full_thermodynamics, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 31: dU/dT Scaling Law")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        "The fundamental laws of physics govern all matter",
        "Neural networks learn representations from data",
        "Stars form from collapsing clouds of gas and dust",
        "The gradient descent algorithm minimizes loss functions",
        "Water freezes at zero degrees Celsius",
        "The brain processes information through neurons",
        "Mathematics is the language of nature",
        "Quantum entanglement connects distant particles",
    ]

    import gc
    model_data = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n--- Model: Qwen2.5-{size} ---")
        model, tok = load_model(device=device, size=size)
        d = model.config.hidden_size
        n_layers = model.config.num_hidden_layers

        all_dUdT = []
        all_PRT = []
        for p in prompts:
            thermo, _ = measure_full_thermodynamics(model, tok, p, device)
            U = np.array([t['U'] for t in thermo])
            T = np.array([t['T'] for t in thermo])
            PRT = np.array([t['PRT'] for t in thermo])

            try:
                valid = np.isfinite(T[1:]) & np.isfinite(U[1:])
                slope = np.polyfit(T[1:][valid], U[1:][valid], 1)[0]
            except Exception:
                slope = 0
            all_dUdT.append(slope)
            all_PRT.append(PRT[-1])

        avg_dUdT = np.mean(all_dUdT)
        std_dUdT = np.std(all_dUdT)
        avg_PRT = np.mean(all_PRT)

        model_data[size] = {
            'd': d, 'n_layers': n_layers,
            'dUdT_mean': avg_dUdT, 'dUdT_std': std_dUdT,
            'PRT_mean': avg_PRT,
            'dUdT_all': all_dUdT,
        }
        print(f"  d={d}, L={n_layers}")
        print(f"  dU/dT = {avg_dUdT:.2f} +/- {std_dUdT:.2f}")
        print(f"  PRT = {avg_PRT:.1f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute scaling exponents
    d_vals = [model_data[s]['d'] for s in ['0.5B', '1.5B']]
    dUdT_vals = [abs(model_data[s]['dUdT_mean']) for s in ['0.5B', '1.5B']]
    PRT_vals = [model_data[s]['PRT_mean'] for s in ['0.5B', '1.5B']]

    # Power law: |dU/dT| ~ d^alpha
    log_d = np.log(d_vals)
    log_dUdT = np.log(np.array(dUdT_vals) + 1e-10)
    alpha_dUdT = (log_dUdT[1] - log_dUdT[0]) / (log_d[1] - log_d[0])

    log_PRT = np.log(np.array(PRT_vals) + 1e-10)
    alpha_PRT = (log_PRT[1] - log_PRT[0]) / (log_d[1] - log_d[0])

    print(f"\n--- Scaling Exponents ---")
    print(f"  |dU/dT| ~ d^{alpha_dUdT:.2f}")
    print(f"  PRT ~ d^{alpha_PRT:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    sizes_label = ['0.5B', '1.5B']
    ax.bar(sizes_label, dUdT_vals, color=['#3498db', '#e74c3c'], alpha=0.8)
    for i, v in enumerate(dUdT_vals):
        ax.text(i, v+0.5, f'{v:.1f}', ha='center', fontsize=11)
    ax.set_ylabel('|dU/dT|')
    ax.set_title(f'(a) |dU/dT| Scaling (exponent={alpha_dUdT:.2f})')

    ax = axes[1]
    ax.bar(sizes_label, PRT_vals, color=['#3498db', '#e74c3c'], alpha=0.8)
    for i, v in enumerate(PRT_vals):
        ax.text(i, v+0.5, f'{v:.1f}', ha='center', fontsize=11)
    ax.set_ylabel('PR x T')
    ax.set_title(f'(b) PRT Scaling (exponent={alpha_PRT:.2f})')

    ax = axes[2]
    ax.loglog(d_vals, dUdT_vals, 'o-', color='#e74c3c', ms=8, label=f'|dU/dT| ~ d^{{{alpha_dUdT:.2f}}}')
    ax.loglog(d_vals, PRT_vals, 's-', color='#3498db', ms=8, label=f'PRT ~ d^{{{alpha_PRT:.2f}}}')
    ax.set_xlabel('Model Dimension d')
    ax.set_ylabel('Value')
    ax.set_title('(c) Log-Log Scaling')
    ax.legend()

    fig.suptitle(
        f"Phase 31: dU/dT Scaling Law\n"
        f"|dU/dT| ~ d^{alpha_dUdT:.2f} | PRT ~ d^{alpha_PRT:.2f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase31_scaling_law")
    plt.close()

    verdict = (f"|dU/dT| scales as d^{alpha_dUdT:.2f}, PRT scales as d^{alpha_PRT:.2f}. "
               f"0.5B: |dU/dT|={dUdT_vals[0]:.1f}, 1.5B: |dU/dT|={dUdT_vals[1]:.1f}.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase31_scaling_law", {
        'name': 'Phase 31: dU/dT Scaling Law',
        'summary': {'verdict': verdict, 'alpha_dUdT': float(alpha_dUdT),
                    'alpha_PRT': float(alpha_PRT), 'model_data': {
                        k: {kk: vv for kk, vv in v.items() if kk != 'dUdT_all'}
                        for k, v in model_data.items()}},
    })


if __name__ == '__main__':
    main()
