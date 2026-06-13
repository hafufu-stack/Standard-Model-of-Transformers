# -*- coding: utf-8 -*-
"""
Phase 90: Virial Equilibrium Dynamics
Precise measurement of virial ratio 2K/|U| at every layer,
where K = kinetic energy (layer-to-layer change) and U = potential energy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects differentiation and",
    "Quantum mechanics describes particles at the atomic scale",
    "The human genome contains three billion base pairs encoding",
    "Neural networks learn through layers of interconnected nodes",
    "Black holes form from gravitational collapse of massive stars",
    "The periodic table organizes chemical elements by number",
    "Evolution by natural selection operates on heritable variation",
    "Climate change affects ecosystems through rising temperatures",
]


def main():
    print("=" * 70)
    print("Phase 90: Virial Equilibrium Dynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    all_virial_profiles = []
    all_K_profiles = []
    all_U_profiles = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Extract hidden state norms and full vectors
        hs_list = [out.hidden_states[li][0, -1, :].cpu().float() for li in range(len(out.hidden_states))]

        # U = potential energy (norm of hidden state)
        Us = [h.norm().item() for h in hs_list]

        # K = kinetic energy = ||h_{l+1} - h_l||^2 / 2  (velocity squared)
        Ks = []
        for i in range(1, len(hs_list)):
            delta = hs_list[i] - hs_list[i-1]
            K = 0.5 * (delta.norm().item() ** 2)
            Ks.append(K)

        # Virial ratio at each layer transition: 2K / |U|
        virial_ratios = []
        for i in range(len(Ks)):
            U_avg = (abs(Us[i]) + abs(Us[i+1])) / 2
            if U_avg > 1e-6:
                vr = 2 * Ks[i] / U_avg
            else:
                vr = 0.0
            virial_ratios.append(vr)

        all_virial_profiles.append(virial_ratios)
        all_K_profiles.append(Ks)
        all_U_profiles.append(Us)

    # Average profiles
    n_trans = min(len(p) for p in all_virial_profiles)
    mean_virial = np.mean([p[:n_trans] for p in all_virial_profiles], axis=0)
    std_virial = np.std([p[:n_trans] for p in all_virial_profiles], axis=0)
    mean_K = np.mean([p[:n_trans] for p in all_K_profiles], axis=0)
    mean_U = np.mean([p[:n_trans+1] for p in all_U_profiles], axis=0)

    layers = np.arange(n_trans)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Virial ratio profile
    axes[0].plot(layers, mean_virial, 'o-', color='#c0392b', linewidth=2, markersize=4)
    axes[0].fill_between(layers, mean_virial - std_virial, mean_virial + std_virial,
                          alpha=0.15, color='#c0392b')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Virial equilibrium (1.0)')
    axes[0].set_xlabel('Layer Transition')
    axes[0].set_ylabel('Virial Ratio $2K/|U|$')
    axes[0].set_title('(a) Virial Ratio Profile')
    axes[0].legend(fontsize=8)
    axes[0].set_yscale('log')

    # (b) K and U profiles
    axes[1].plot(layers, mean_K, 'o-', color='#2980b9', linewidth=2, markersize=3, label='$K$ (kinetic)')
    ax1b = axes[1].twinx()
    ax1b.plot(np.arange(len(mean_U)), mean_U, 's-', color='#c0392b', linewidth=2, markersize=3, label='$U$ (potential)')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Kinetic Energy $K$', color='#2980b9')
    ax1b.set_ylabel('Potential Energy $U$', color='#c0392b')
    axes[1].set_title('(b) K and U Profiles')
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (c) Convergence analysis - rolling mean of virial ratio
    if n_trans > 5:
        window = 5
        rolling = np.convolve(mean_virial, np.ones(window)/window, mode='valid')
        axes[2].plot(np.arange(len(rolling)), rolling, '-', color='#8e44ad',
                     linewidth=2, label=f'Rolling mean (w={window})')
        axes[2].axhline(y=1.0, color='gray', linestyle='--')
        final_value = rolling[-1] if len(rolling) > 0 else mean_virial[-1]
        axes[2].axhline(y=final_value, color='#27ae60', linestyle=':',
                        label=f'Final = {final_value:.2f}')
    else:
        final_value = mean_virial[-1]
        axes[2].plot(layers, mean_virial, 'o-', color='#8e44ad')
    axes[2].set_xlabel('Layer Transition')
    axes[2].set_ylabel('Virial Ratio')
    axes[2].set_title(f'(c) Convergence (final = {final_value:.2f})')
    axes[2].legend(fontsize=8)

    fig.suptitle(f'Phase 90: Virial Equilibrium (final ratio = {final_value:.2f}, not 1.0)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase90_virial_equilibrium')
    plt.close()

    # Phase analysis
    early_virial = float(np.mean(mean_virial[:5]))
    mid_virial = float(np.mean(mean_virial[n_trans//3:2*n_trans//3]))
    late_virial = float(np.mean(mean_virial[-5:]))

    print(f"\n{'='*70}")
    print(f"Early layers (0-4): virial = {early_virial:.2f}")
    print(f"Mid layers: virial = {mid_virial:.2f}")
    print(f"Late layers: virial = {late_virial:.2f}")
    print(f"Final convergence: {final_value:.2f} (equilibrium = 1.0)")
    print(f"{'='*70}")

    save_results('phase90_virial_equilibrium', {
        'experiment': 'Virial Equilibrium Dynamics',
        'mean_virial_profile': [float(v) for v in mean_virial],
        'summary': {
            'early_virial': early_virial,
            'mid_virial': mid_virial,
            'late_virial': late_virial,
            'final_value': float(final_value),
            'at_equilibrium': bool(abs(final_value - 1.0) < 0.5),
        }
    })


if __name__ == '__main__':
    main()
