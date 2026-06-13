# -*- coding: utf-8 -*-
"""
Phase 83: Ergodic Hypothesis
Is the time average (across tokens) equal to the ensemble average (across prompts)?
If ergodic: one long sequence gives the same statistics as many short ones.
This is fundamental to whether the thermodynamic framework applies.
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
    print("Phase 83: Ergodic Hypothesis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Many short prompts (ensemble)
    short_prompts = [
        "The speed of light is",
        "Water freezes at zero",
        "The Earth orbits the",
        "DNA consists of four",
        "Gravity pulls objects down",
        "The sun is a",
        "Electrons carry negative charge",
        "The human heart pumps",
        "Plants produce oxygen through",
        "The moon orbits Earth",
        "Sound travels through air",
        "Iron is a magnetic",
        "The brain contains billions",
        "Volcanoes erupt when magma",
        "Lightning is an electrical",
        "The ocean covers most",
        "Diamonds are made of",
        "Fire requires oxygen to",
        "The sky appears blue",
        "Ice is less dense",
    ]

    # One long prompt (time series)
    long_prompt = (
        "The speed of light is a fundamental constant in physics that determines "
        "the maximum speed at which all massless particles and associated fields "
        "can travel in a vacuum. This value is exactly two hundred ninety nine "
        "million seven hundred ninety two thousand four hundred fifty eight meters "
        "per second. The constancy of the speed of light is a postulate of special "
        "relativity and plays a central role in modern physics including quantum "
        "mechanics and general relativity. The speed of light also appears in the "
        "famous equation relating energy and mass."
    )

    LAYER_IDX = 14  # mid-layer for comparison

    # === Ensemble average: each prompt at LAYER_IDX ===
    ensemble_T = []
    ensemble_U = []
    ensemble_PR = []

    for prompt in short_prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        li = min(LAYER_IDX, len(out.hidden_states) - 1)
        h = out.hidden_states[li][0, -1, :].float()
        U = h.norm().item()

        with torch.no_grad():
            normed = model.model.norm(out.hidden_states[li][:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        PR = 1.0 / (h_prob ** 2).sum().item()

        ensemble_T.append(T if not np.isnan(T) else 0)
        ensemble_U.append(U)
        ensemble_PR.append(PR)

    # === Time average: long prompt, each token at LAYER_IDX ===
    long_inp = tok(long_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        long_out = model(**long_inp, output_hidden_states=True)

    li = min(LAYER_IDX, len(long_out.hidden_states) - 1)
    hs = long_out.hidden_states[li][0]  # (seq, hidden)

    time_T = []
    time_U = []
    time_PR = []

    for tok_idx in range(hs.shape[0]):
        h = hs[tok_idx, :].float()
        U = h.norm().item()

        # Need to compute logits for each token position
        with torch.no_grad():
            normed = model.model.norm(long_out.hidden_states[li][:, tok_idx:tok_idx+1, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        PR = 1.0 / (h_prob ** 2).sum().item()

        time_T.append(T if not np.isnan(T) else 0)
        time_U.append(U)
        time_PR.append(PR)

    # Compare statistics
    comparisons = {
        'T': (ensemble_T, time_T),
        'U': (ensemble_U, time_U),
        'PR': (ensemble_PR, time_PR),
    }

    ergodic_scores = {}
    for name, (ens, tim) in comparisons.items():
        ens_mean = np.mean(ens)
        tim_mean = np.mean(tim)
        ens_std = np.std(ens)
        tim_std = np.std(tim)
        # Relative difference
        rel_diff = abs(ens_mean - tim_mean) / (abs(ens_mean) + abs(tim_mean) + 1e-10) * 2
        # KS test
        ks_stat, ks_p = stats.ks_2samp(ens, tim)
        ergodic_scores[name] = {
            'ens_mean': float(ens_mean), 'tim_mean': float(tim_mean),
            'ens_std': float(ens_std), 'tim_std': float(tim_std),
            'rel_diff': float(rel_diff), 'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
        }
        print(f"  {name}: ensemble={ens_mean:.3f}+/-{ens_std:.3f}, "
              f"time={tim_mean:.3f}+/-{tim_std:.3f}, rel_diff={rel_diff:.3f}, "
              f"KS p={ks_p:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) T comparison
    axes[0, 0].hist(ensemble_T, bins=10, alpha=0.5, color='#e74c3c', label='Ensemble', density=True)
    axes[0, 0].hist(time_T, bins=10, alpha=0.5, color='#3498db', label='Time', density=True)
    axes[0, 0].set_xlabel('T')
    axes[0, 0].set_title(f'(a) T (rel_diff={ergodic_scores["T"]["rel_diff"]:.3f})')
    axes[0, 0].legend()

    # (b) U comparison
    axes[0, 1].hist(ensemble_U, bins=10, alpha=0.5, color='#e74c3c', label='Ensemble', density=True)
    axes[0, 1].hist(time_U, bins=10, alpha=0.5, color='#3498db', label='Time', density=True)
    axes[0, 1].set_xlabel('U')
    axes[0, 1].set_title(f'(b) U (rel_diff={ergodic_scores["U"]["rel_diff"]:.3f})')
    axes[0, 1].legend()

    # (c) PR comparison
    axes[0, 2].hist(ensemble_PR, bins=10, alpha=0.5, color='#e74c3c', label='Ensemble', density=True)
    axes[0, 2].hist(time_PR, bins=10, alpha=0.5, color='#3498db', label='Time', density=True)
    axes[0, 2].set_xlabel('PR')
    axes[0, 2].set_title(f'(c) PR (rel_diff={ergodic_scores["PR"]["rel_diff"]:.3f})')
    axes[0, 2].legend()

    # (d) Mean comparison bar chart
    names = ['T', 'U', 'PR']
    ens_means = [ergodic_scores[n]['ens_mean'] for n in names]
    tim_means = [ergodic_scores[n]['tim_mean'] for n in names]
    x = np.arange(3)
    axes[1, 0].bar(x - 0.2, ens_means, 0.35, color='#e74c3c', label='Ensemble', alpha=0.8)
    axes[1, 0].bar(x + 0.2, tim_means, 0.35, color='#3498db', label='Time', alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_title('(d) Mean Comparison')
    axes[1, 0].legend()

    # (e) KS test p-values
    ks_ps = [ergodic_scores[n]['ks_p'] for n in names]
    axes[1, 1].bar(names, ks_ps, color=['#2ecc71' if p > 0.05 else '#e74c3c' for p in ks_ps],
                   alpha=0.8)
    axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
    axes[1, 1].set_ylabel('KS p-value')
    axes[1, 1].set_title('(e) KS Test (p>0.05 = ergodic)')
    axes[1, 1].legend()

    # (f) Summary
    n_ergodic = sum(1 for n in names if ergodic_scores[n]['ks_p'] > 0.05)
    summary = (
        f"ERGODIC HYPOTHESIS TEST\n"
        f"Layer {LAYER_IDX}\n\n"
        f"Ensemble: {len(short_prompts)} prompts\n"
        f"Time: {len(time_T)} tokens\n\n"
        + '\n'.join([f"{n}: diff={ergodic_scores[n]['rel_diff']:.3f}, "
                     f"KS p={ergodic_scores[n]['ks_p']:.3f}"
                     for n in names])
        + f"\n\n{'ERGODIC' if n_ergodic >= 2 else 'NON-ERGODIC'}"
        f" ({n_ergodic}/3 pass KS)"
    )
    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=10, va='center', ha='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    is_ergodic = n_ergodic >= 2
    fig.suptitle(f'Phase 83: Ergodic Hypothesis ({n_ergodic}/3 ergodic)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase83_ergodic')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: {n_ergodic}/3 observables pass KS test. "
          f"System is {'ERGODIC' if is_ergodic else 'NON-ERGODIC'}.")
    print(f"{'='*70}")

    save_results('phase83_ergodic', {
        'experiment': 'Ergodic Hypothesis',
        'ergodic_scores': ergodic_scores,
        'summary': {
            'n_ergodic': n_ergodic,
            'is_ergodic': bool(is_ergodic),
        }
    })


if __name__ == '__main__':
    main()
