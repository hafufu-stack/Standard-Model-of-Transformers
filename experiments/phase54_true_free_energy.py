# -*- coding: utf-8 -*-
"""
Phase 54: The True Free Energy (F = U - T*ln(PR))
Fix Phase 52: Use S = ln(PR) instead of S = T.
If F decreases across layers, LLMs perform variational free energy minimization.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics


def main():
    print("=" * 70)
    print("Phase 54: The True Free Energy (F = U - T*ln(PR))")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration through",
        "In quantum mechanics, the wave function collapse occurs when a measurement is",
        "The human genome contains approximately three billion base pairs of DNA that encode",
        "Artificial neural networks are inspired by the biological structure of the brain and",
        "Black holes form when massive stars exhaust their nuclear fuel and undergo gravitational",
        "The Turing test evaluates whether a machine can exhibit intelligent behavior indistinguishable from",
        "The standard model of particle physics classifies all known elementary particles into",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from",
        "Climate models use differential equations to simulate atmospheric dynamics and predict future",
        "Cryptographic hash functions transform arbitrary input data into fixed-size output that is",
        "Evolution by natural selection operates on heritable variation within populations over many",
        "The cosmic microwave background radiation provides a snapshot of the universe approximately",
    ]

    all_profiles = []

    for prompt in prompts:
        thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
        if len(thermo) < 3:
            continue

        U = np.array([r['U'] for r in thermo])
        T = np.array([r['T'] for r in thermo])
        PR = np.array([r['PR'] for r in thermo])
        PRT = np.array([r['PRT'] for r in thermo])

        # TRUE entropy: S = ln(PR)
        S_true = np.log(PR + 1e-10)

        # TRUE free energy: F = U - T * S = U - T * ln(PR)
        F_true = U - T * S_true

        # Old (wrong) free energy for comparison: F_old = U - T^2
        F_old = U - T * T

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' F_true: {F_true[0]:.1f} -> {F_true[-1]:.1f} "
              f"(dF={F_true[-1]-F_true[0]:.1f})")

        all_profiles.append({
            'prompt': prompt[:60],
            'U': U.tolist(), 'T': T.tolist(), 'PR': PR.tolist(),
            'S_true': S_true.tolist(),
            'F_true': F_true.tolist(), 'F_old': F_old.tolist(),
            'PRT': PRT.tolist(),
        })

    n_layers = len(all_profiles[0]['F_true'])
    layers_x = np.arange(n_layers)

    mean_F_true = np.mean([p['F_true'] for p in all_profiles], axis=0)
    mean_F_old = np.mean([p['F_old'] for p in all_profiles], axis=0)
    std_F_true = np.std([p['F_true'] for p in all_profiles], axis=0)
    mean_U = np.mean([p['U'] for p in all_profiles], axis=0)
    mean_T = np.mean([p['T'] for p in all_profiles], axis=0)
    mean_S = np.mean([p['S_true'] for p in all_profiles], axis=0)

    # F_true trend
    slope_new, intc_new, r_new, p_new, _ = stats.linregress(layers_x, mean_F_true)
    slope_old, intc_old, r_old, p_old, _ = stats.linregress(layers_x, mean_F_old)

    dF = np.diff(mean_F_true)
    pct_decreasing = np.sum(dF < 0) / len(dF) * 100
    F_min_layer = np.argmin(mean_F_true)

    print(f"\n=== Free Energy Comparison ===")
    print(f"  F_old (U - T^2):    slope={slope_old:.2f}, r={r_old:.3f} (WRONG)")
    print(f"  F_true (U - T*lnPR): slope={slope_new:.2f}, r={r_new:.3f} (CORRECT)")
    print(f"  {pct_decreasing:.0f}% transitions decrease F_true")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) F_true landscape
    axes[0, 0].plot(layers_x, mean_F_true, color='#e74c3c', linewidth=2, label='F = U - T*ln(PR)')
    axes[0, 0].fill_between(layers_x, mean_F_true - std_F_true, mean_F_true + std_F_true,
                            alpha=0.2, color='#e74c3c')
    axes[0, 0].axvline(x=F_min_layer, color='gray', linestyle='--',
                       label=f'Min at L{F_min_layer}')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Free Energy F')
    axes[0, 0].set_title(f'(a) TRUE Free Energy (slope={slope_new:.1f})')
    axes[0, 0].legend(fontsize=8)

    # (b) Old vs New comparison
    axes[0, 1].plot(layers_x, mean_F_old, color='gray', linewidth=1.5, linestyle='--',
                    label=f'F_old = U - T^2 (slope={slope_old:.1f})')
    axes[0, 1].plot(layers_x, mean_F_true, color='#e74c3c', linewidth=2,
                    label=f'F_true = U - T*ln(PR) (slope={slope_new:.1f})')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Free Energy')
    axes[0, 1].set_title('(b) Old vs Corrected F')
    axes[0, 1].legend(fontsize=8)

    # (c) Decomposition: U and T*S
    TS = mean_T * mean_S
    axes[0, 2].plot(layers_x, mean_U, 'b-', linewidth=1.5, label='U (energy)')
    axes[0, 2].plot(layers_x, TS, 'g-', linewidth=1.5, label='T*ln(PR) (entropy term)')
    axes[0, 2].plot(layers_x, mean_F_true, 'r-', linewidth=2, label='F = U - T*ln(PR)')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].set_title('(c) Thermodynamic Decomposition')
    axes[0, 2].legend(fontsize=8)

    # (d) dF/dl gradient
    colors_df = ['#2ecc71' if d < 0 else '#e74c3c' for d in dF]
    axes[1, 0].bar(np.arange(len(dF)), dF, color=colors_df, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linewidth=1)
    axes[1, 0].set_xlabel('Layer Transition')
    axes[1, 0].set_ylabel('dF/dl')
    axes[1, 0].set_title(f'(d) Free Energy Gradient ({pct_decreasing:.0f}% decreasing)')

    # (e) Individual trajectories
    for p in all_profiles:
        axes[1, 1].plot(p['F_true'], alpha=0.3, color='#e74c3c', linewidth=0.8)
    axes[1, 1].plot(mean_F_true, 'k-', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('F_true')
    axes[1, 1].set_title('(e) Individual Trajectories')
    axes[1, 1].legend()

    # (f) F vs T phase diagram
    sc = axes[1, 2].scatter(mean_T, mean_F_true, c=layers_x, cmap='viridis', s=40,
                           edgecolors='black', linewidth=0.5)
    for i in range(0, n_layers-1, max(1, n_layers//6)):
        axes[1, 2].annotate('', xy=(mean_T[i+1], mean_F_true[i+1]),
                           xytext=(mean_T[i], mean_F_true[i]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    axes[1, 2].set_xlabel('Temperature T')
    axes[1, 2].set_ylabel('Free Energy F')
    axes[1, 2].set_title('(f) Phase Diagram')
    plt.colorbar(sc, ax=axes[1, 2], label='Layer')

    fig.suptitle('Phase 54: The True Free Energy Principle',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase54_true_free_energy')
    plt.close()

    is_decreasing = slope_new < 0 and p_new < 0.05

    print(f"\n{'='*70}")
    print(f"VERDICT: F_true slope={slope_new:.1f} (p={p_new:.2e}), "
          f"{pct_decreasing:.0f}% decreasing, min at L{F_min_layer}. "
          f"Free Energy Principle {'CONFIRMED' if is_decreasing else 'NOT confirmed'} "
          f"with S=ln(PR). [Old F slope was {slope_old:.1f}]")
    print(f"{'='*70}")

    save_results('phase54_true_free_energy', {
        'experiment': 'True Free Energy',
        'F_true_slope': float(slope_new), 'F_true_r': float(r_new), 'F_true_p': float(p_new),
        'F_old_slope': float(slope_old),
        'pct_decreasing': float(pct_decreasing),
        'F_min_layer': int(F_min_layer),
        'summary': {
            'is_decreasing': bool(is_decreasing),
            'slope_new': float(slope_new), 'slope_old': float(slope_old),
        }
    })


if __name__ == '__main__':
    main()
