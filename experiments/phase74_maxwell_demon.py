# -*- coding: utf-8 -*-
"""
Phase 74: Maxwell's Demon (Information-Thermodynamic Coupling)
Maxwell's demon can decrease entropy by using information.
Does the model's "information" about the correct answer (measured by
top-1 probability) correlate with the thermodynamic work done (dF/dl)?
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
    print("Phase 74: Maxwell's Demon (Info-Thermo Coupling)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Mix of easy and hard prompts to get variability in "knowledge"
    prompts = [
        # Easy (model "knows" the answer)
        "The capital of France is",
        "Water freezes at zero degrees",
        "The chemical symbol for gold is",
        "Two plus two equals",
        "The Earth orbits the",
        "The speed of light is approximately",
        # Hard (model less certain)
        "The 37th prime number is",
        "The population of Liechtenstein in 2019 was approximately",
        "The exact melting point of tungsten in Kelvin is",
        "The name of the third moon of Neptune is",
        "The author of the 1847 novel Jane Eyre is",
        "The half-life of Carbon-14 is approximately",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        confidence_profile = []
        entropy_profile = []
        F_profile = []

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            top1 = probs.max().item()

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

            F = U - T * S

            confidence_profile.append(top1)
            entropy_profile.append(T)
            F_profile.append(F)

        # Measure "demon efficiency": how much info gain per unit of work
        dF = [F_profile[i+1] - F_profile[i] for i in range(len(F_profile)-1)]
        dC = [confidence_profile[i+1] - confidence_profile[i] for i in range(len(confidence_profile)-1)]

        total_work = sum(dF)
        total_info_gain = confidence_profile[-1] - confidence_profile[0]
        demon_efficiency = total_info_gain / (abs(total_work) + 1e-10)

        final_confidence = confidence_profile[-1]

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' conf={final_confidence:.3f}, "
              f"work={total_work:.0f}, demon_eff={demon_efficiency:.4f}")

        all_results.append({
            'prompt': prompt[:60],
            'final_confidence': float(final_confidence),
            'total_work': float(total_work),
            'demon_efficiency': float(demon_efficiency),
            'confidence_profile': [float(c) for c in confidence_profile],
            'entropy_profile': [float(e) for e in entropy_profile],
            'F_profile': [float(f) for f in F_profile],
            'dF': [float(d) for d in dF],
            'dC': [float(d) for d in dC],
        })

    # === Analysis ===
    confs = [r['final_confidence'] for r in all_results]
    works = [r['total_work'] for r in all_results]
    effs = [r['demon_efficiency'] for r in all_results]

    corr_cw, p_cw = stats.pearsonr(confs, works)
    corr_ce, p_ce = stats.pearsonr(confs, effs)

    # Landauer bound: minimum work to erase 1 bit = kT*ln(2)
    # In our units: minimum work per bit of info = mean_T * ln(2)
    mean_T = np.mean([np.mean(r['entropy_profile']) for r in all_results])
    landauer_bound = mean_T * np.log(2)

    print(f"\n=== Maxwell's Demon Analysis ===")
    print(f"  Confidence-Work corr: r={corr_cw:.3f} (p={p_cw:.3e})")
    print(f"  Confidence-Efficiency corr: r={corr_ce:.3f}")
    print(f"  Landauer bound: {landauer_bound:.1f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Confidence vs Work scatter
    axes[0, 0].scatter(works, confs, s=80, c=effs, cmap='RdYlGn', edgecolors='black')
    axes[0, 0].set_xlabel('Total Work (dF)')
    axes[0, 0].set_ylabel('Final Confidence (top-1 prob)')
    axes[0, 0].set_title(f'(a) Confidence vs Work (r={corr_cw:.2f})')

    # (b) Confidence profiles
    for r in all_results:
        c = '#2ecc71' if r['final_confidence'] > 0.5 else '#e74c3c'
        axes[0, 1].plot(r['confidence_profile'], color=c, alpha=0.5, linewidth=1)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Top-1 Probability')
    axes[0, 1].set_title('(b) Confidence Evolution')

    # (c) Entropy vs Confidence trajectories
    for r in all_results:
        axes[0, 2].plot(r['entropy_profile'], r['confidence_profile'],
                       '-', alpha=0.5, linewidth=1)
        axes[0, 2].scatter(r['entropy_profile'][-1], r['confidence_profile'][-1],
                          s=50, edgecolors='black', zorder=5)
    axes[0, 2].set_xlabel('Entropy T')
    axes[0, 2].set_ylabel('Confidence')
    axes[0, 2].set_title('(c) Phase Space Trajectories')

    # (d) Demon efficiency
    sorted_res = sorted(all_results, key=lambda r: r['demon_efficiency'])
    labels = [r['prompt'][:20] for r in sorted_res]
    effs_sorted = [r['demon_efficiency'] for r in sorted_res]
    colors_d = ['#2ecc71' if e > 0 else '#e74c3c' for e in effs_sorted]
    axes[1, 0].barh(range(len(labels)), effs_sorted, color=colors_d, alpha=0.7)
    axes[1, 0].set_yticks(range(len(labels)))
    axes[1, 0].set_yticklabels(labels, fontsize=7)
    axes[1, 0].set_xlabel('Demon Efficiency (info/work)')
    axes[1, 0].set_title('(d) Maxwell Demon Efficiency')

    # (e) dF vs dConfidence correlation per layer
    all_dF = np.mean([r['dF'] for r in all_results], axis=0)
    all_dC = np.mean([r['dC'] for r in all_results], axis=0)
    axes[1, 1].scatter(all_dF, all_dC, s=40, color='#e74c3c')
    r_layer, p_layer = stats.pearsonr(all_dF, all_dC)
    axes[1, 1].set_xlabel('Mean dF (work per layer)')
    axes[1, 1].set_ylabel('Mean dConfidence')
    axes[1, 1].set_title(f'(e) Layer-wise dF vs dC (r={r_layer:.2f})')

    # (f) Landauer analysis
    bits_gained = [np.log2(r['final_confidence'] / 
                   (r['confidence_profile'][0] + 1e-10) + 1e-10)
                   for r in all_results]
    work_per_bit = [r['total_work'] / (abs(b) + 1e-10) for r, b in zip(all_results, bits_gained)]
    axes[1, 2].scatter(bits_gained, work_per_bit, s=80, color='#9b59b6', edgecolors='black')
    axes[1, 2].axhline(y=landauer_bound, color='red', linestyle='--',
                       label=f'Landauer bound={landauer_bound:.0f}')
    axes[1, 2].set_xlabel('Information Gained (bits)')
    axes[1, 2].set_ylabel('Work per bit')
    axes[1, 2].set_title('(f) Landauer Bound Test')
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Phase 74: Maxwell's Demon (Info-Thermo Coupling)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase74_maxwell_demon')
    plt.close()

    is_coupled = abs(corr_cw) > 0.5 or abs(corr_ce) > 0.5

    print(f"\n{'='*70}")
    print(f"VERDICT: Conf-Work r={corr_cw:.3f}, Conf-Eff r={corr_ce:.3f}. "
          f"Info-thermo coupling {'CONFIRMED' if is_coupled else 'WEAK'}. "
          f"Mean demon efficiency={np.mean(effs):.4f}.")
    print(f"{'='*70}")

    save_results('phase74_maxwell_demon', {
        'experiment': "Maxwell's Demon",
        'summary': {
            'corr_conf_work': float(corr_cw),
            'corr_conf_eff': float(corr_ce),
            'mean_demon_eff': float(np.mean(effs)),
            'landauer_bound': float(landauer_bound),
            'is_coupled': bool(is_coupled),
        }
    })


if __name__ == '__main__':
    main()
