# -*- coding: utf-8 -*-
"""
Phase 52: Free Energy Landscape
Compute Helmholtz free energy F = U - TS across layers.
If F is minimized at the output, LLMs perform variational free energy minimization
(connecting to the Free Energy Principle in neuroscience).
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
    print("Phase 52: Free Energy Landscape")
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
        S = T.copy()  # S = entropy = T in our framework
        PR = np.array([r['PR'] for r in thermo])
        PRT = np.array([r['PRT'] for r in thermo])

        # Free energy: F = U - T*S (here S=T, so F = U - T^2)
        # But more physically: F = U - beta^{-1} * S where beta^{-1} = kT
        # Using macro temperature: F = U - T * S = U - T^2
        F = U - T * S  # = U - T^2

        # Also compute Gibbs free energy G = F + PV (P = PR, V = 1/PR)
        # G = F + 1 (trivially constant), so more useful:
        # Use G = U - TS + PRT as a "generalized potential"
        G = U - T * S + PRT

        # Landau free energy: F_L = U - T*S + a*PRT^2 (with a chosen for stability)
        # Simpler: just track F and see if it's minimized

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' F_0={F[0]:.1f}, F_end={F[-1]:.1f}, "
              f"dF={F[-1]-F[0]:.1f}")

        all_profiles.append({
            'prompt': prompt[:60],
            'U': U.tolist(), 'T': T.tolist(), 'S': S.tolist(),
            'F': F.tolist(), 'G': G.tolist(),
            'PR': PR.tolist(), 'PRT': PRT.tolist(),
        })

    # === Analysis ===
    n_layers = len(all_profiles[0]['F'])

    mean_F = np.mean([p['F'] for p in all_profiles], axis=0)
    mean_U = np.mean([p['U'] for p in all_profiles], axis=0)
    mean_T = np.mean([p['T'] for p in all_profiles], axis=0)
    mean_S = np.mean([p['S'] for p in all_profiles], axis=0)
    mean_G = np.mean([p['G'] for p in all_profiles], axis=0)
    mean_PRT = np.mean([p['PRT'] for p in all_profiles], axis=0)

    std_F = np.std([p['F'] for p in all_profiles], axis=0)

    # Check monotonicity of F
    dF = np.diff(mean_F)
    n_decreasing = np.sum(dF < 0)
    pct_decreasing = n_decreasing / len(dF) * 100

    # Find minimum layer
    F_min_layer = np.argmin(mean_F)
    F_min_val = mean_F[F_min_layer]

    # Linear trend of F
    layers_x = np.arange(n_layers)
    slope_F, intercept_F, r_F, p_F, _ = stats.linregress(layers_x, mean_F)

    print(f"\n=== Free Energy Analysis ===")
    print(f"  F trend: slope={slope_F:.2f}, r={r_F:.3f}, p={p_F:.2e}")
    print(f"  F minimum at layer {F_min_layer}")
    print(f"  {pct_decreasing:.0f}% transitions decrease F")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Free energy landscape
    axes[0, 0].plot(layers_x, mean_F, color='#e74c3c', linewidth=2, label='Mean F')
    axes[0, 0].fill_between(layers_x, mean_F - std_F, mean_F + std_F,
                            alpha=0.2, color='#e74c3c')
    axes[0, 0].axvline(x=F_min_layer, color='gray', linestyle='--',
                       label=f'Min at L{F_min_layer}')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Free Energy F = U - TS')
    axes[0, 0].set_title(f'(a) Free Energy Landscape (slope={slope_F:.1f})')
    axes[0, 0].legend()

    # (b) U, T*S decomposition
    TS = mean_T * mean_S
    axes[0, 1].plot(layers_x, mean_U, color='#3498db', linewidth=1.5, label='U')
    axes[0, 1].plot(layers_x, TS, color='#2ecc71', linewidth=1.5, label='TS')
    axes[0, 1].plot(layers_x, mean_F, color='#e74c3c', linewidth=2, label='F = U - TS')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('(b) Thermodynamic Decomposition')
    axes[0, 1].legend()

    # (c) dF/dl (free energy gradient)
    axes[0, 2].bar(np.arange(len(dF)), dF,
                   color=['#2ecc71' if d < 0 else '#e74c3c' for d in dF], alpha=0.7)
    axes[0, 2].axhline(y=0, color='black', linewidth=1)
    axes[0, 2].set_xlabel('Layer Transition')
    axes[0, 2].set_ylabel('dF/dl')
    axes[0, 2].set_title(f'(c) Free Energy Gradient ({pct_decreasing:.0f}% negative)')

    # (d) Individual F trajectories
    for p in all_profiles:
        axes[1, 0].plot(p['F'], alpha=0.3, color='#e74c3c', linewidth=0.8)
    axes[1, 0].plot(mean_F, color='black', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Free Energy F')
    axes[1, 0].set_title('(d) Individual Trajectories')
    axes[1, 0].legend()

    # (e) F vs PRT scatter
    axes[1, 1].scatter(mean_PRT, mean_F, c=layers_x, cmap='viridis', s=40,
                      edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('PRT (Conserved Quantity)')
    axes[1, 1].set_ylabel('Free Energy F')
    axes[1, 1].set_title('(e) F vs PRT')
    cb = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cb.set_label('Layer')

    # (f) Phase diagram: F vs T
    axes[1, 2].scatter(mean_T, mean_F, c=layers_x, cmap='plasma', s=40,
                      edgecolors='black', linewidth=0.5)
    # Arrow showing direction of flow
    for i in range(0, n_layers-1, max(1, n_layers//8)):
        dx = mean_T[i+1] - mean_T[i]
        dy = mean_F[i+1] - mean_F[i]
        axes[1, 2].annotate('', xy=(mean_T[i+1], mean_F[i+1]),
                           xytext=(mean_T[i], mean_F[i]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    axes[1, 2].set_xlabel('Temperature T')
    axes[1, 2].set_ylabel('Free Energy F')
    axes[1, 2].set_title('(f) Phase Diagram (F vs T)')
    cb2 = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cb2.set_label('Layer')

    fig.suptitle('Phase 52: Free Energy Landscape of LLM Inference',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase52_free_energy')
    plt.close()

    # === Verdict ===
    is_decreasing = slope_F < 0 and p_F < 0.05

    print(f"\n{'='*70}")
    print(f"VERDICT: F slope={slope_F:.1f} (p={p_F:.2e}), "
          f"{pct_decreasing:.0f}% decreasing, min at L{F_min_layer}. "
          f"LLM {'DOES' if is_decreasing else 'does NOT'} minimize free energy. "
          f"{'Consistent with Free Energy Principle.' if is_decreasing else ''}")
    print(f"{'='*70}")

    save_results('phase52_free_energy', {
        'experiment': 'Free Energy Landscape',
        'F_slope': float(slope_F), 'F_r': float(r_F), 'F_p': float(p_F),
        'F_min_layer': int(F_min_layer),
        'pct_decreasing': float(pct_decreasing),
        'summary': {
            'is_decreasing': bool(is_decreasing),
            'slope': float(slope_F),
            'p_value': float(p_F),
        }
    })


if __name__ == '__main__':
    main()
