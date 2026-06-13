# -*- coding: utf-8 -*-
"""
Phase 67: Virial Theorem (2K = -U for self-gravitating systems)
In self-gravitating systems: 2*KineticEnergy = -PotentialEnergy.
For LLM: K = attention energy (variance), U_pot = -||h||^2.
If 2K ~ |U_pot|, Attention truly behaves as gravity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 67: Virial Theorem (2K = -U)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration",
        "Quantum mechanics describes the behavior of particles at the atomic scale",
        "The human brain processes information through billions of interconnected neurons",
        "Evolution by natural selection operates on heritable variation within populations",
        "The periodic table organizes chemical elements by atomic number and electron",
        "Climate change affects global ecosystems through rising temperatures and shifting",
        "Cryptographic hash functions transform arbitrary data into fixed size output",
        "The cosmic microwave background radiation provides evidence of the early universe",
        "Photosynthesis converts carbon dioxide and water into glucose using light energy",
        "Machine learning algorithms discover patterns in data without explicit programming",
        "Black holes form when massive stars exhaust their nuclear fuel and collapse",
        "The speed of light in vacuum is approximately three hundred million meters",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        K_values = []  # kinetic energy (attention-driven variance)
        U_pot_values = []  # potential energy (norm squared)
        virial_ratios = []

        for li in range(1, len(out.hidden_states)):
            h_curr = out.hidden_states[li][0].float()  # (seq, hidden)
            h_prev = out.hidden_states[li - 1][0].float()

            # K = mean kinetic energy = variance of state changes (velocity^2)
            dh = h_curr - h_prev  # (seq, hidden)
            K = 0.5 * (dh ** 2).mean().item()  # mean kinetic energy

            # U_pot = potential energy from binding (negative of norm)
            U_pot = -(h_curr ** 2).mean().item()  # gravitational potential

            # Virial ratio: 2K / |U_pot|
            virial = 2 * K / (abs(U_pot) + 1e-10)

            K_values.append(K)
            U_pot_values.append(U_pot)
            virial_ratios.append(virial)

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        mean_virial = np.mean(virial_ratios)
        print(f"  '{safe_p}...' mean virial ratio = {mean_virial:.4f}")

        all_results.append({
            'prompt': prompt[:60],
            'K': K_values, 'U_pot': U_pot_values,
            'virial_ratios': virial_ratios,
            'mean_virial': float(mean_virial),
        })

    overall_virial = np.mean([r['mean_virial'] for r in all_results])
    n_layers_minus1 = len(all_results[0]['virial_ratios'])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    layers_x = np.arange(n_layers_minus1)

    # (a) Virial ratio per layer
    mean_vr = np.mean([r['virial_ratios'] for r in all_results], axis=0)
    std_vr = np.std([r['virial_ratios'] for r in all_results], axis=0)
    axes[0, 0].plot(layers_x, mean_vr, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    axes[0, 0].fill_between(layers_x, mean_vr - std_vr, mean_vr + std_vr,
                            alpha=0.2, color='#e74c3c')
    axes[0, 0].axhline(y=1.0, color='blue', linestyle='--', linewidth=2,
                       label='Virial equilibrium (2K=|U|)')
    axes[0, 0].set_xlabel('Layer Transition')
    axes[0, 0].set_ylabel('Virial Ratio (2K/|U_pot|)')
    axes[0, 0].set_title(f'(a) Virial Ratio (mean={overall_virial:.3f})')
    axes[0, 0].legend()

    # (b) K vs |U_pot| scatter
    all_K = [k for r in all_results for k in r['K']]
    all_U = [abs(u) for r in all_results for u in r['U_pot']]
    axes[0, 1].scatter(all_U, [2*k for k in all_K], s=10, alpha=0.3, color='#e74c3c')
    max_val = max(max(all_U), max([2*k for k in all_K]))
    axes[0, 1].plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='2K = |U_pot|')
    axes[0, 1].set_xlabel('|U_pot| (Potential Energy)')
    axes[0, 1].set_ylabel('2K (Kinetic Energy)')
    axes[0, 1].set_title('(b) Virial Relation')
    axes[0, 1].legend()

    # (c) K and U_pot profiles
    mean_K = np.mean([r['K'] for r in all_results], axis=0)
    mean_U = np.mean([abs(u) for r in all_results for u in [r['U_pot']]], axis=0) \
             if False else np.mean([[-u for u in r['U_pot']] for r in all_results], axis=0)
    axes[0, 2].plot(layers_x, mean_K, 'r-', linewidth=2, label='K (kinetic)')
    axes[0, 2].plot(layers_x, mean_U, 'b-', linewidth=2, label='|U_pot| (potential)')
    axes[0, 2].set_xlabel('Layer Transition')
    axes[0, 2].set_ylabel('Energy')
    axes[0, 2].set_title('(c) Energy Components')
    axes[0, 2].legend()

    # (d) Virial ratio distribution
    all_virial = [v for r in all_results for v in r['virial_ratios']]
    axes[1, 0].hist(all_virial, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=1.0, color='red', linewidth=2, linestyle='--', label='Virial (2K=|U|)')
    axes[1, 0].axvline(x=overall_virial, color='blue', linewidth=2,
                       label=f'Mean={overall_virial:.3f}')
    axes[1, 0].set_xlabel('Virial Ratio')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('(d) Distribution')
    axes[1, 0].legend(fontsize=8)

    # (e) Per-prompt mean virial
    means = sorted([r['mean_virial'] for r in all_results])
    axes[1, 1].bar(range(len(means)), means, color='#3498db', alpha=0.8)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Prompt')
    axes[1, 1].set_ylabel('Mean Virial Ratio')
    axes[1, 1].set_title('(e) Per-Prompt Virial')

    # (f) Correlation: K vs U
    corr, pval = stats.pearsonr(all_K, all_U)
    axes[1, 2].scatter(all_K, all_U, s=10, alpha=0.3, color='#2ecc71')
    axes[1, 2].set_xlabel('K (kinetic)')
    axes[1, 2].set_ylabel('|U_pot| (potential)')
    axes[1, 2].set_title(f'(f) K-U Correlation (r={corr:.3f}, p={pval:.2e})')

    virial_holds = abs(overall_virial - 1.0) < 0.3
    fig.suptitle(f'Phase 67: Virial Theorem (2K/|U| = {overall_virial:.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase67_virial_theorem')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Mean virial ratio = {overall_virial:.3f} (ideal=1.0). "
          f"K-U correlation r={corr:.3f}. "
          f"Virial theorem {'HOLDS' if virial_holds else 'DOES NOT hold'} "
          f"(deviation: {abs(overall_virial-1)*100:.0f}%).")
    print(f"{'='*70}")

    save_results('phase67_virial_theorem', {
        'experiment': 'Virial Theorem',
        'summary': {
            'mean_virial': float(overall_virial),
            'virial_holds': bool(virial_holds),
            'correlation': float(corr),
        }
    })


if __name__ == '__main__':
    main()
