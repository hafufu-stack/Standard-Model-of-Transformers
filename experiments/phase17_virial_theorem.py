# -*- coding: utf-8 -*-
"""
Phase 17: The Virial Theorem of Attention (v2)
===============================================
Hook-free approach: use hidden_states differences to compute K and U.
  K = kinetic energy = ||h_l - h_{l-1}||^2 / 2
  U = potential energy ~ -(sum of velocity reduction) 
For self-gravitating systems: 2K + U = 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 17: The Virial Theorem of Attention (v2)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The fundamental laws of physics govern all matter",
        "Neural networks learn representations from data",
        "The second law of thermodynamics states that entropy",
        "Quantum entanglement connects distant particles",
        "Stars form from collapsing clouds of gas and dust",
        "The gradient descent algorithm minimizes loss functions",
        "Information theory quantifies uncertainty in signals",
        "Black holes have an event horizon beyond which nothing",
    ]

    all_K, all_U_pot = [], []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hs = [out.hidden_states[l][0, -1, :].float().cpu() for l in range(len(out.hidden_states))]

        # K = kinetic energy: ||delta_l||^2 / 2 where delta = h_l - h_{l-1}
        K_layers = []
        for l in range(1, len(hs)):
            delta = (hs[l] - hs[l-1])
            K = 0.5 * (delta.norm().item() ** 2)
            K_layers.append(K)

        # U = gravitational potential energy
        # In self-gravitating systems, U ~ -GM^2/R
        # Here: U ~ -(mass * binding) where mass = ||h_l|| and binding = cos(h_l, h_{l-1})
        # This measures how much each layer "binds" to the previous (attraction)
        U_layers = []
        for l in range(1, len(hs)):
            norm_curr = hs[l].norm().item()
            norm_prev = hs[l-1].norm().item()
            cos_sim = torch.dot(hs[l], hs[l-1]) / (norm_curr * norm_prev + 1e-10)
            # Binding energy: stronger when aligned (cos~1), weaker when orthogonal
            U = -(norm_curr * norm_prev * cos_sim.item())
            U_layers.append(U)

        all_K.append(K_layers)
        all_U_pot.append(U_layers)

    # Average
    min_len = min(len(k) for k in all_K)
    avg_K = np.mean([k[:min_len] for k in all_K], axis=0)
    avg_U = np.mean([u[:min_len] for u in all_U_pot], axis=0)

    # Virial ratio
    virial_ratio = np.array([2*k / (abs(u) + 1e-10) for k, u in zip(avg_K, avg_U)])
    virial_sum = 2 * avg_K + avg_U

    print("\n--- Virial Theorem Check ---")
    print(f"  Mean 2K = {np.mean(2*avg_K):.2f}")
    print(f"  Mean |U| = {np.mean(np.abs(avg_U)):.2f}")
    print(f"  Mean 2K+U = {np.mean(virial_sum):.2f}")
    mean_ratio = float(np.nanmean(virial_ratio))
    print(f"  Mean virial ratio 2K/|U| = {mean_ratio:.4f}")
    print(f"  (Perfect virial = 1.0)")

    for l in range(0, min_len, 4):
        print(f"  L{l+1}: 2K={2*avg_K[l]:.1f}, |U|={abs(avg_U[l]):.1f}, "
              f"ratio={virial_ratio[l]:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    layers = np.arange(min_len)
    ax = axes[0]
    ax.plot(layers, 2*avg_K, 'o-', color='#e74c3c', ms=4, label='2K (kinetic)')
    ax.plot(layers, np.abs(avg_U), 's-', color='#3498db', ms=4, label='|U| (potential)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Energy')
    ax.set_title('(a) Kinetic vs Potential Energy')
    ax.legend()

    ax = axes[1]
    ax.plot(layers, virial_sum, 'o-', color='#9b59b6', ms=4)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5, label='Virial equilibrium')
    ax.set_xlabel('Layer'); ax.set_ylabel('2K + U')
    ax.set_title('(b) Virial Deviation')
    ax.legend()

    ax = axes[2]
    ax.plot(layers, virial_ratio, 'o-', color='#2ecc71', ms=4)
    ax.axhline(y=1.0, color='red', ls='--', alpha=0.5, label='Perfect virial (1.0)')
    ax.set_xlabel('Layer'); ax.set_ylabel('2K / |U|')
    ax.set_title('(c) Virial Ratio')
    ax.legend()

    fig.suptitle(
        f"Phase 17: Virial Theorem\n"
        f"Mean virial ratio = {mean_ratio:.4f} (1.0 = equilibrium)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase17_virial_theorem")
    plt.close()

    if abs(mean_ratio - 1.0) < 0.3:
        verdict = (f"VIRIAL EQUILIBRIUM: 2K/|U| = {mean_ratio:.3f}. "
                   f"Transformer IS a self-gravitating system in equilibrium!")
    else:
        verdict = (f"NON-VIRIAL: 2K/|U| = {mean_ratio:.3f} (deviate {abs(mean_ratio-1.0)*100:.0f}%). "
                   f"Mean 2K={np.mean(2*avg_K):.0f}, Mean |U|={np.mean(np.abs(avg_U)):.0f}.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 17: Virial Theorem',
        'summary': {'verdict': verdict, 'mean_virial_ratio': mean_ratio,
                    'mean_2K': float(np.mean(2*avg_K)),
                    'mean_abs_U': float(np.mean(np.abs(avg_U)))},
    }
    save_results("phase17_virial_theorem", result)
    return result


if __name__ == '__main__':
    main()
