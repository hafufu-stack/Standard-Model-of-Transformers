# -*- coding: utf-8 -*-
"""
Phase 51: Carnot Efficiency
Measure how much of the model's 'energy budget' goes to useful computation.
Carnot efficiency eta = 1 - T_cold/T_hot where T_hot = first layer, T_cold = last layer.
Also compute information-theoretic efficiency: bits of useful output per unit energy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure, measure_full_thermodynamics


def main():
    print("=" * 70)
    print("Phase 51: Carnot Efficiency")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration through",
        "In the standard model of particle physics, quarks interact via the strong force",
        "The evolutionary theory of natural selection proposed by Darwin explains how species",
        "The blockchain technology underlying Bitcoin uses cryptographic hash functions to",
        "Quantum entanglement allows two particles to be correlated regardless of the distance",
        "The Renaissance period in Europe was characterized by a revival of interest in",
        "Machine learning algorithms can be broadly classified into supervised and unsupervised",
        "The human immune system consists of innate and adaptive components that work",
        "Climate models predict that global temperatures will rise by several degrees",
        "The double-slit experiment demonstrates the wave-particle duality of matter and",
        "Photosynthesis in plants converts carbon dioxide and water into glucose using",
        "The Navier-Stokes equations describe the motion of viscous fluid substances",
    ]

    all_efficiencies = []

    for prompt in prompts:
        thermo, out = measure_full_thermodynamics(model, tok, prompt, device)

        if len(thermo) < 3:
            continue

        # Extract T profile
        T_profile = [r['T'] for r in thermo]
        U_profile = [r['U'] for r in thermo]
        PR_profile = [r['PR'] for r in thermo]
        PRT_profile = [r['PRT'] for r in thermo]

        T_hot = max(T_profile)  # Highest entropy (early layers)
        T_cold = min(t for t in T_profile if t > 0.01)  # Lowest entropy (final layers)

        # Carnot efficiency
        if T_hot > 0:
            eta_carnot = 1 - T_cold / T_hot
        else:
            eta_carnot = 0

        # Actual efficiency: how much entropy was removed
        T_initial = T_profile[0] if T_profile[0] > 0 else T_profile[1]
        T_final = T_profile[-1]
        if T_initial > 0:
            eta_actual = (T_initial - T_final) / T_initial
        else:
            eta_actual = 0

        # Energy efficiency: U_final / U_total_invested
        U_total = sum(abs(U_profile[i+1] - U_profile[i]) for i in range(len(U_profile)-1))
        U_net = abs(U_profile[-1] - U_profile[0])
        if U_total > 0:
            eta_energy = U_net / U_total
        else:
            eta_energy = 0

        # Information compression: PR reduction (how much the distribution sharpens)
        PR_initial = PR_profile[0]
        PR_final = PR_profile[-1]
        if PR_initial > 0:
            compression = 1 - PR_final / PR_initial
        else:
            compression = 0

        # PRT conservation quality
        prt_cv = np.std(PRT_profile) / (np.mean(PRT_profile) + 1e-10)

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' Carnot={eta_carnot:.3f}, "
              f"Actual={eta_actual:.3f}, Compress={compression:.3f}")

        all_efficiencies.append({
            'prompt': prompt[:60],
            'T_hot': float(T_hot),
            'T_cold': float(T_cold),
            'eta_carnot': float(eta_carnot),
            'eta_actual': float(eta_actual),
            'eta_energy': float(eta_energy),
            'compression': float(compression),
            'prt_cv': float(prt_cv),
            'T_profile': [float(t) for t in T_profile],
            'U_profile': [float(u) for u in U_profile],
        })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Carnot vs Actual efficiency
    carnots = [e['eta_carnot'] for e in all_efficiencies]
    actuals = [e['eta_actual'] for e in all_efficiencies]
    axes[0, 0].scatter(carnots, actuals, s=60, c='#e74c3c', edgecolors='black')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Carnot limit')
    axes[0, 0].set_xlabel('Carnot Efficiency (theoretical max)')
    axes[0, 0].set_ylabel('Actual Efficiency')
    axes[0, 0].set_title('(a) Carnot vs Actual')
    axes[0, 0].legend()

    # (b) Efficiency distributions
    axes[0, 1].hist(carnots, bins=10, alpha=0.6, color='#3498db', label='Carnot')
    axes[0, 1].hist(actuals, bins=10, alpha=0.6, color='#e74c3c', label='Actual')
    axes[0, 1].set_xlabel('Efficiency')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('(b) Efficiency Distribution')
    axes[0, 1].legend()

    # (c) T profiles (all prompts overlaid)
    for e in all_efficiencies:
        axes[0, 2].plot(e['T_profile'], alpha=0.3, color='#3498db', linewidth=0.8)
    mean_T = np.mean([e['T_profile'] for e in all_efficiencies], axis=0)
    axes[0, 2].plot(mean_T, color='#e74c3c', linewidth=2, label='Mean T')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Temperature T')
    axes[0, 2].set_title('(c) Temperature Profiles (T_hot -> T_cold)')
    axes[0, 2].legend()

    # (d) U profiles
    for e in all_efficiencies:
        axes[1, 0].plot(e['U_profile'], alpha=0.3, color='#2ecc71', linewidth=0.8)
    mean_U = np.mean([e['U_profile'] for e in all_efficiencies], axis=0)
    axes[1, 0].plot(mean_U, color='#e74c3c', linewidth=2, label='Mean U')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Internal Energy U')
    axes[1, 0].set_title('(d) Energy Profiles (U increases)')
    axes[1, 0].legend()

    # (e) Compression vs Carnot
    compressions = [e['compression'] for e in all_efficiencies]
    axes[1, 1].scatter(carnots, compressions, s=60, c='#9b59b6', edgecolors='black')
    axes[1, 1].set_xlabel('Carnot Efficiency')
    axes[1, 1].set_ylabel('Information Compression')
    axes[1, 1].set_title('(e) Compression vs Efficiency')

    # (f) Summary bar chart
    metrics = ['Carnot', 'Actual', 'Energy', 'Compression']
    means = [np.mean(carnots), np.mean(actuals),
             np.mean([e['eta_energy'] for e in all_efficiencies]),
             np.mean(compressions)]
    stds = [np.std(carnots), np.std(actuals),
            np.std([e['eta_energy'] for e in all_efficiencies]),
            np.std(compressions)]
    bars = axes[1, 2].bar(metrics, means, yerr=stds, capsize=5,
                          color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], alpha=0.8)
    axes[1, 2].set_ylabel('Efficiency')
    axes[1, 2].set_title('(f) Mean Efficiencies')
    for bar, m in zip(bars, means):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{m:.3f}', ha='center', fontsize=10)

    fig.suptitle('Phase 51: Carnot Efficiency of LLM Inference',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase51_carnot')
    plt.close()

    # Verdict
    mean_carnot = np.mean(carnots)
    mean_actual = np.mean(actuals)
    ratio = mean_actual / (mean_carnot + 1e-10)

    print(f"\n{'='*70}")
    print(f"VERDICT: Mean Carnot={mean_carnot:.3f}, Actual={mean_actual:.3f}, "
          f"Ratio={ratio:.2f} (LLM operates at {ratio*100:.0f}% of Carnot limit). "
          f"Compression={np.mean(compressions):.3f}.")
    print(f"{'='*70}")

    save_results('phase51_carnot', {
        'experiment': 'Carnot Efficiency',
        'results': all_efficiencies,
        'summary': {
            'mean_carnot': float(mean_carnot),
            'mean_actual': float(mean_actual),
            'carnot_ratio': float(ratio),
            'mean_compression': float(np.mean(compressions)),
        }
    })


if __name__ == '__main__':
    main()
