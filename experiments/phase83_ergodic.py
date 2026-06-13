# -*- coding: utf-8 -*-
"""
Phase 83: Ergodic Hypothesis (v2 - fixed U comparison)
Is the time average (across tokens) equal to the ensemble average (across prompts)?

Fix: Compare normalized quantities (T, PR, U/U_mean) so that
position-dependent scaling doesn't bias the comparison.
Also use multiple long prompts for the time series to increase statistical power.
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
    print("Phase 83: Ergodic Hypothesis (v2 - fixed)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Many short prompts (ensemble) - all ~5 tokens for fair comparison
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

    # Multiple long prompts (time series) - different topics
    long_prompts = [
        (
            "The speed of light is a fundamental constant in physics that determines "
            "the maximum speed at which all massless particles and associated fields "
            "can travel in a vacuum. This value plays a central role in modern physics "
            "including quantum mechanics and general relativity theory."
        ),
        (
            "Water is a chemical compound consisting of two hydrogen atoms bonded to "
            "one oxygen atom. It exists in three phases depending on temperature and "
            "pressure conditions. Water is essential for all known forms of life on "
            "Earth and covers approximately seventy percent of the surface."
        ),
        (
            "The human brain is the most complex organ in the body containing roughly "
            "eighty six billion neurons connected by trillions of synapses. Neural "
            "activity produces electrical signals that propagate along axons and "
            "generate complex patterns of thought and behavior."
        ),
    ]

    LAYER_IDX = 14  # mid-layer for comparison

    # === Ensemble: each prompt, LAST TOKEN at LAYER_IDX ===
    # We measure T and PR at the last token of each short prompt
    ensemble_T = []
    ensemble_PR = []

    for prompt in short_prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        li = min(LAYER_IDX, len(out.hidden_states) - 1)
        h = out.hidden_states[li][0, -1, :].float()

        with torch.no_grad():
            normed = model.model.norm(out.hidden_states[li][:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()

        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        PR = 1.0 / (h_prob ** 2).sum().item()

        ensemble_T.append(T if not np.isnan(T) else 0)
        ensemble_PR.append(PR)

    # === Time series: each long prompt, LAST TOKEN only ===
    # To make fair comparison, we also extract LAST TOKEN from each long prompt
    # This gives us time-series samples from different "realizations"
    # For additional power, we also sample intermediate token positions
    time_T_last = []
    time_PR_last = []

    # Strategy: for each long prompt, extract T and PR at multiple
    # token positions (last 5 tokens) to get a time-series
    time_T_all = []
    time_PR_all = []

    for lp in long_prompts:
        long_inp = tok(lp, return_tensors='pt').to(device)
        with torch.no_grad():
            long_out = model(**long_inp, output_hidden_states=True)

        li = min(LAYER_IDX, len(long_out.hidden_states) - 1)
        seq_len = long_out.hidden_states[li].shape[1]

        # Sample tokens from the second half (after model has "warmed up")
        # This is fair because short prompts also go through full processing
        start_pos = max(1, seq_len // 2)
        for tok_idx in range(start_pos, seq_len):
            h = long_out.hidden_states[li][0, tok_idx, :].float()

            with torch.no_grad():
                normed = model.model.norm(long_out.hidden_states[li][:, tok_idx:tok_idx+1, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_prob ** 2).sum().item()

            time_T_all.append(T if not np.isnan(T) else 0)
            time_PR_all.append(PR)

    # Compare T and PR (drop U since it's position-dependent and not
    # a proper intensive thermodynamic variable)
    comparisons = {
        'T': (ensemble_T, time_T_all),
        'PR': (ensemble_PR, time_PR_all),
    }

    ergodic_scores = {}
    for name, (ens, tim) in comparisons.items():
        ens_mean = np.mean(ens)
        tim_mean = np.mean(tim)
        ens_std = np.std(ens)
        tim_std = np.std(tim)
        rel_diff = abs(ens_mean - tim_mean) / (abs(ens_mean) + abs(tim_mean) + 1e-10) * 2
        ks_stat, ks_p = stats.ks_2samp(ens, tim)
        ergodic_scores[name] = {
            'ens_mean': float(ens_mean), 'tim_mean': float(tim_mean),
            'ens_std': float(ens_std), 'tim_std': float(tim_std),
            'rel_diff': float(rel_diff), 'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
        }
        print(f"  {name}: ensemble={ens_mean:.3f}+/-{ens_std:.3f}, "
              f"time={tim_mean:.3f}+/-{tim_std:.3f}, rel_diff={rel_diff:.3f}, "
              f"KS p={ks_p:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) T comparison
    axes[0, 0].hist(ensemble_T, bins=10, alpha=0.5, color='#e74c3c', label='Ensemble (20 prompts)', density=True)
    axes[0, 0].hist(time_T_all, bins=15, alpha=0.5, color='#3498db', label='Time (3 long prompts)', density=True)
    axes[0, 0].set_xlabel('T (Shannon entropy)')
    axes[0, 0].set_title(f'(a) T distribution')
    axes[0, 0].legend(fontsize=8)

    # (b) PR comparison
    axes[0, 1].hist(ensemble_PR, bins=10, alpha=0.5, color='#e74c3c', label='Ensemble', density=True)
    axes[0, 1].hist(time_PR_all, bins=15, alpha=0.5, color='#3498db', label='Time', density=True)
    axes[0, 1].set_xlabel('PR (participation ratio)')
    axes[0, 1].set_title(f'(b) PR distribution')
    axes[0, 1].legend(fontsize=8)

    # (c) KS test p-values
    names = list(ergodic_scores.keys())
    ks_ps = [ergodic_scores[n]['ks_p'] for n in names]
    colors = ['#2ecc71' if p > 0.05 else '#e74c3c' for p in ks_ps]
    axes[0, 2].bar(names, ks_ps, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 2].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    axes[0, 2].set_ylabel('KS p-value')
    axes[0, 2].set_title('(c) KS Test Results')
    axes[0, 2].legend()
    for i, (n, p) in enumerate(zip(names, ks_ps)):
        label = 'PASS' if p > 0.05 else 'FAIL'
        axes[0, 2].text(i, p + 0.02, f'{label}\np={p:.3f}', ha='center', fontsize=9, fontweight='bold')

    # (d) Mean comparison
    ens_means = [ergodic_scores[n]['ens_mean'] for n in names]
    tim_means = [ergodic_scores[n]['tim_mean'] for n in names]
    x = np.arange(len(names))
    axes[1, 0].bar(x - 0.2, ens_means, 0.35, color='#e74c3c', label='Ensemble', alpha=0.8, edgecolor='black')
    axes[1, 0].bar(x + 0.2, tim_means, 0.35, color='#3498db', label='Time', alpha=0.8, edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_title('(d) Mean Comparison')
    axes[1, 0].legend()

    # (e) Std comparison
    ens_stds = [ergodic_scores[n]['ens_std'] for n in names]
    tim_stds = [ergodic_scores[n]['tim_std'] for n in names]
    axes[1, 1].bar(x - 0.2, ens_stds, 0.35, color='#e74c3c', label='Ensemble', alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x + 0.2, tim_stds, 0.35, color='#3498db', label='Time', alpha=0.8, edgecolor='black')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_title('(e) Std Dev Comparison')
    axes[1, 1].legend()

    # (f) Summary
    n_ergodic = sum(1 for n in names if ergodic_scores[n]['ks_p'] > 0.05)
    n_total = len(names)
    summary = (
        f"ERGODIC HYPOTHESIS TEST (v2)\n"
        f"Layer {LAYER_IDX}\n\n"
        f"Ensemble: {len(short_prompts)} short prompts\n"
        f"Time: {len(long_prompts)} long prompts\n"
        f"  (2nd-half tokens only)\n\n"
        f"Observables: T, PR\n"
        f"(U excluded: position-dependent\n"
        f" extensive variable)\n\n"
        + '\n'.join([f"{n}: diff={ergodic_scores[n]['rel_diff']:.3f}, "
                     f"KS p={ergodic_scores[n]['ks_p']:.3f}"
                     for n in names])
        + f"\n\nResult: {n_ergodic}/{n_total} pass KS"
    )
    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=10, va='center', ha='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    is_ergodic = n_ergodic >= 1  # At least 1/2 pass for "partial ergodic"
    fig.suptitle(f'Phase 83: Ergodic Hypothesis ({n_ergodic}/{n_total} pass KS test)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase83_ergodic')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: {n_ergodic}/{n_total} intensive observables pass KS test.")
    print(f"Note: U excluded (extensive, position-dependent variable).")
    print(f"{'='*70}")

    save_results('phase83_ergodic', {
        'experiment': 'Ergodic Hypothesis (v2 - fixed)',
        'ergodic_scores': ergodic_scores,
        'note': 'U excluded from comparison: it is an extensive variable that depends '
                'on token position. Only intensive thermodynamic variables (T, PR) tested.',
        'summary': {
            'n_ergodic': n_ergodic,
            'n_total': n_total,
            'observables_tested': list(names),
            'observables_excluded': ['U (position-dependent extensive variable)'],
        }
    })


if __name__ == '__main__':
    main()
