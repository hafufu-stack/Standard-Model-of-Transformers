# -*- coding: utf-8 -*-
"""
Phase 119: Symmetry Breaking at Transition
What symmetry is broken at L0? Analyze the distribution of hidden state
activations before and after the transition. If a symmetry breaks,
we expect the distribution to change from symmetric to asymmetric.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def main():
    print("=" * 70)
    print("Phase 119: Symmetry Breaking at Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Collect per-layer statistics
    skewness = []
    kurtosis = []
    mean_abs = []
    std_vals = []
    neg_frac = []  # fraction of negative activations
    top10_frac = []  # fraction of energy in top 10% of neurons
    entropy_act = []  # entropy of |activation| distribution

    for li in range(n_layers):
        sk_all, ku_all, ma_all, sd_all, nf_all, t10_all, ea_all = [], [], [], [], [], [], []

        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()

            sk_all.append(float(sp_stats.skew(h)))
            ku_all.append(float(sp_stats.kurtosis(h)))
            ma_all.append(float(np.mean(np.abs(h))))
            sd_all.append(float(np.std(h)))
            nf_all.append(float(np.mean(h < 0)))

            # Top 10% energy concentration
            h_sq = h**2
            h_sq_sorted = np.sort(h_sq)[::-1]
            top10 = int(len(h) * 0.1)
            t10_all.append(float(h_sq_sorted[:top10].sum() / (h_sq.sum() + 1e-10)))

            # Activation entropy
            p = np.abs(h) / (np.sum(np.abs(h)) + 1e-10)
            ea = -np.sum(p * np.log(p + 1e-10))
            ea_all.append(float(ea))

        skewness.append(np.mean(sk_all))
        kurtosis.append(np.mean(ku_all))
        mean_abs.append(np.mean(ma_all))
        std_vals.append(np.mean(sd_all))
        neg_frac.append(np.mean(nf_all))
        top10_frac.append(np.mean(t10_all))
        entropy_act.append(np.mean(ea_all))

    layers = np.arange(n_layers)

    # Derivatives of symmetry measures
    d_skew = np.gradient(skewness)
    d_kurtosis = np.gradient(kurtosis)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Skewness
    axes[0, 0].plot(layers, skewness, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Skewness')
    axes[0, 0].set_title('(a) Activation Skewness')
    axes[0, 0].legend()

    # (b) Kurtosis (excess)
    axes[0, 1].plot(layers, kurtosis, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Excess Kurtosis')
    axes[0, 1].set_title('(b) Kurtosis (tail heaviness)')

    # (c) Negative fraction (symmetry indicator)
    axes[0, 2].plot(layers, neg_frac, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Symmetric')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Fraction Negative')
    axes[0, 2].set_title('(c) Sign Symmetry')
    axes[0, 2].legend()

    # (d) Top-10% energy concentration
    axes[1, 0].plot(layers, top10_frac, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Energy in Top 10% Neurons')
    axes[1, 0].set_title('(d) Energy Concentration')

    # (e) Activation entropy
    axes[1, 1].plot(layers, entropy_act, 'o-', color='#e67e22', markersize=4, linewidth=2)
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('$H(|h|)$')
    axes[1, 1].set_title('(e) Activation Entropy')

    # (f) Summary
    pre_skew = np.mean(skewness[:int(L0)])
    post_skew = np.mean(skewness[int(L0):])
    pre_kurt = np.mean(kurtosis[:int(L0)])
    post_kurt = np.mean(kurtosis[int(L0):])
    pre_neg = np.mean(neg_frac[:int(L0)])
    post_neg = np.mean(neg_frac[int(L0):])
    pre_conc = np.mean(top10_frac[:int(L0)])
    post_conc = np.mean(top10_frac[int(L0):])

    summary = (
        f"Symmetry Breaking Analysis\n\n"
        f"Metric        Pre    Post   Change\n"
        f"Skewness:   {pre_skew:+.3f}  {post_skew:+.3f}  {post_skew-pre_skew:+.3f}\n"
        f"Kurtosis:   {pre_kurt:+.2f}  {post_kurt:+.2f}  {post_kurt-pre_kurt:+.2f}\n"
        f"Neg frac:   {pre_neg:.3f}  {post_neg:.3f}  {post_neg-pre_neg:+.3f}\n"
        f"Top10 conc: {pre_conc:.3f}  {post_conc:.3f}  {post_conc-pre_conc:+.3f}\n\n"
        f"Broken symmetry:\n"
        f"{'Sign symmetry' if abs(post_neg - 0.5) > abs(pre_neg - 0.5) else 'No sign break'}\n"
        f"{'Energy concentrates' if post_conc > pre_conc else 'Energy distributes'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 119: Symmetry Breaking at Transition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase119_symmetry')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Skewness: pre={pre_skew:+.3f}, post={post_skew:+.3f}")
    print(f"Kurtosis: pre={pre_kurt:+.2f}, post={post_kurt:+.2f}")
    print(f"Neg frac: pre={pre_neg:.3f}, post={post_neg:.3f}")
    print(f"Top10: pre={pre_conc:.3f}, post={post_conc:.3f}")
    print(f"{'='*70}")

    save_results('phase119_symmetry', {
        'experiment': 'Symmetry Breaking at Transition',
        'skewness': [float(v) for v in skewness],
        'kurtosis': [float(v) for v in kurtosis],
        'neg_frac': [float(v) for v in neg_frac],
        'top10_frac': [float(v) for v in top10_frac],
        'entropy_act': [float(v) for v in entropy_act],
        'summary': {
            'pre_skew': float(pre_skew),
            'post_skew': float(post_skew),
            'pre_kurtosis': float(pre_kurt),
            'post_kurtosis': float(post_kurt),
            'pre_neg': float(pre_neg),
            'post_neg': float(post_neg),
            'pre_conc': float(pre_conc),
            'post_conc': float(post_conc),
        }
    })


if __name__ == '__main__':
    main()
