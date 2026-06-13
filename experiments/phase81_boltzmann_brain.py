# -*- coding: utf-8 -*-
"""
Phase 81: Boltzmann Brain Paradox
In equilibrium thermodynamics, thermal fluctuations can spontaneously
create ordered structures (Boltzmann brains). Do LLM hidden states
occasionally "fluctuate" into unexpected high-confidence states?
Measure the frequency and magnitude of spontaneous confidence spikes.
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
    print("Phase 81: Boltzmann Brain (Spontaneous Confidence Fluctuations)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation",
        "Quantum mechanics describes particles at the atomic scale",
        "The human genome contains three billion base pairs",
        "Neural networks learn through gradient descent optimization",
        "Black holes form from gravitational collapse of massive stars",
        "The periodic table organizes chemical elements by number",
        "Evolution by natural selection operates on heritable variation",
        "Climate change affects global ecosystems through temperature",
        "Photosynthesis converts sunlight into chemical energy stored",
        "Machine learning discovers hidden patterns in large datasets",
        "General relativity describes gravity as curvature of spacetime",
        "Cryptographic protocols ensure secure communication channels",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        conf_profile = []
        for li, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            conf_profile.append(probs.max().item())

        # Detect Boltzmann brains: confidence spikes in middle layers
        # that are higher than both neighbors
        brains = []
        for i in range(2, len(conf_profile) - 2):
            local_mean = np.mean(conf_profile[max(0,i-3):i] + conf_profile[i+1:min(len(conf_profile),i+4)])
            if conf_profile[i] > local_mean * 1.5 and conf_profile[i] > 0.1:
                brains.append({
                    'layer': i,
                    'confidence': conf_profile[i],
                    'local_mean': local_mean,
                    'spike_ratio': conf_profile[i] / (local_mean + 1e-10),
                })

        # Monotonicity: is confidence monotonically increasing?
        diffs = [conf_profile[i+1] - conf_profile[i] for i in range(len(conf_profile)-1)]
        pct_increasing = sum(1 for d in diffs if d > 0) / len(diffs) * 100

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' brains={len(brains)}, monotone={pct_increasing:.0f}%")

        all_results.append({
            'prompt': prompt[:60],
            'conf_profile': conf_profile,
            'brains': brains,
            'n_brains': len(brains),
            'pct_increasing': float(pct_increasing),
        })

    n_layers = len(all_results[0]['conf_profile'])
    total_brains = sum(r['n_brains'] for r in all_results)
    mean_monotone = np.mean([r['pct_increasing'] for r in all_results])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) All confidence profiles
    for r in all_results:
        axes[0, 0].plot(r['conf_profile'], alpha=0.4, linewidth=1)
        for b in r['brains']:
            axes[0, 0].scatter(b['layer'], b['confidence'], s=80, c='red',
                             edgecolors='black', zorder=5)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Top-1 Confidence')
    axes[0, 0].set_title(f'(a) Confidence Profiles ({total_brains} "brains")')

    # (b) Mean profile with std
    mean_conf = np.mean([r['conf_profile'] for r in all_results], axis=0)
    std_conf = np.std([r['conf_profile'] for r in all_results], axis=0)
    axes[0, 1].plot(range(n_layers), mean_conf, 'o-', color='#e74c3c', linewidth=2, markersize=3)
    axes[0, 1].fill_between(range(n_layers), mean_conf - std_conf, mean_conf + std_conf,
                            alpha=0.2, color='#e74c3c')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Mean Confidence')
    axes[0, 1].set_title('(b) Mean Confidence Evolution')

    # (c) Brain locations histogram
    brain_layers = [b['layer'] for r in all_results for b in r['brains']]
    if brain_layers:
        axes[0, 2].hist(brain_layers, bins=range(n_layers+1), color='#9b59b6',
                       alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Brain Count')
    axes[0, 2].set_title('(c) Boltzmann Brain Locations')

    # (d) Monotonicity per prompt
    monos = sorted([r['pct_increasing'] for r in all_results])
    axes[1, 0].bar(range(len(monos)), monos, color='#2ecc71', alpha=0.7)
    axes[1, 0].axhline(y=50, color='red', linestyle='--', label='Random')
    axes[1, 0].set_xlabel('Prompt (sorted)')
    axes[1, 0].set_ylabel('% Increasing')
    axes[1, 0].set_title(f'(d) Monotonicity ({mean_monotone:.0f}% mean)')
    axes[1, 0].legend()

    # (e) Confidence variance per layer
    var_per_layer = [np.var([r['conf_profile'][l] for r in all_results]) for l in range(n_layers)]
    axes[1, 1].plot(range(n_layers), var_per_layer, 'o-', color='#f39c12', linewidth=2, markersize=3)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].set_title('(e) Confidence Variance (fluctuation)')

    # (f) Summary
    summary = (
        f"Boltzmann Brain Analysis:\n\n"
        f"Total brains: {total_brains}\n"
        f"Mean monotonicity: {mean_monotone:.0f}%\n"
        f"Final confidence: {mean_conf[-1]:.3f}\n"
        f"Initial confidence: {mean_conf[0]:.3f}\n"
        f"Max variance layer: {np.argmax(var_per_layer)}\n\n"
        f"{'Confidence is mostly monotonic' if mean_monotone > 60 else 'Non-monotonic evolution'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=11, va='center', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 81: Boltzmann Brain Paradox ({total_brains} fluctuations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase81_boltzmann_brain')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: {total_brains} Boltzmann brains detected. "
          f"Monotonicity={mean_monotone:.0f}%. "
          f"{'MONOTONIC (no brains)' if total_brains == 0 else f'{total_brains} FLUCTUATIONS found'}.")
    print(f"{'='*70}")

    save_results('phase81_boltzmann_brain', {
        'experiment': 'Boltzmann Brain',
        'summary': {
            'total_brains': total_brains,
            'mean_monotonicity': float(mean_monotone),
            'final_confidence': float(mean_conf[-1]),
        }
    })


if __name__ == '__main__':
    main()
