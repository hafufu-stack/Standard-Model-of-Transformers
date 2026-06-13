# -*- coding: utf-8 -*-
"""
Phase 111: L0 vs Perplexity Correlation
Phase 110 found L0 shifts with input type (natural=15.7, math=21.4).
Hypothesis: L0 correlates with input difficulty (PPL).
Test with diverse prompts spanning easy to hard.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

# Diverse prompts from easy to hard
DIVERSE_PROMPTS = [
    # Easy (common, predictable)
    "The sun rises in the east and sets in the",
    "Once upon a time there was a beautiful princess who",
    "I went to the store to buy some milk and",
    "The cat sat on the mat and looked at the",
    # Medium
    "The theory of evolution explains how species change over",
    "Machine learning models require large amounts of training data to",
    "The Pythagorean theorem states that in a right triangle the",
    "Climate change is caused primarily by the emission of greenhouse",
    # Hard (technical, unpredictable)
    "The Riemann hypothesis concerns the distribution of zeros of the",
    "In quantum field theory the path integral formulation requires summing over",
    "The P versus NP problem asks whether every problem whose solution can",
    "Godel's incompleteness theorems demonstrate fundamental limitations of formal axiomatic",
    # Very hard (creative, ambiguous)
    "The relationship between consciousness and quantum mechanics remains one of",
    "If we could reverse entropy at the molecular level the implications for",
    "The emergence of complex behavior from simple rules suggests that the universe",
    "In the context of artificial general intelligence the alignment problem requires",
]


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


def main():
    print("=" * 70)
    print("Phase 111: L0 vs Perplexity Correlation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    results = []

    for pi, prompt in enumerate(DIVERSE_PROMPTS):
        # Measure PPL for this prompt
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, labels=inp['input_ids'])
        ppl = torch.exp(out.loss).item()

        # Measure eta profile
        eta_profile = []
        out2 = model(**inp, output_hidden_states=True)

        for L in range(4, n_layers):
            T_vals = []
            for li in range(min(L + 1, len(out2.hidden_states))):
                hs = out2.hidden_states[li]
                with torch.no_grad():
                    normed = model.model.norm(hs[:, -1:, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                if not np.isnan(T):
                    T_vals.append(T)
            if len(T_vals) >= 4:
                T_hot = max(T_vals)
                T_cold = min(T_vals[len(T_vals)//2:])
                if T_hot > 0.01:
                    eta_profile.append({'L': L, 'eta': 1.0 - T_cold / T_hot})

        if len(eta_profile) < 5:
            continue

        Ls = np.array([r['L'] for r in eta_profile])
        etas = np.array([r['eta'] for r in eta_profile])

        try:
            popt, _ = curve_fit(sigmoid, Ls, etas,
                                p0=[20, 0.5, np.min(etas), np.max(etas)],
                                maxfev=10000)
            L0_fit = popt[0]
            sig_pred = sigmoid(Ls, *popt)
            ss_res = np.sum((etas - sig_pred)**2)
            ss_tot = np.sum((etas - np.mean(etas))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
        except Exception:
            L0_fit = 20.0
            r2 = 0.0

        results.append({
            'prompt': prompt[:50],
            'ppl': float(ppl),
            'L0': float(L0_fit),
            'r2': float(r2),
        })
        print(f"  [{pi:2d}] PPL={ppl:8.2f}, L0={L0_fit:5.1f}: {prompt[:40]}...")

    # === Correlation analysis ===
    ppls = np.array([r['ppl'] for r in results])
    L0s = np.array([r['L0'] for r in results])

    # Filter out bad fits
    good = [i for i, r in enumerate(results) if r['r2'] > 0.7 and 4 < r['L0'] < 28]
    ppls_g = ppls[good]
    L0s_g = L0s[good]

    if len(ppls_g) >= 3:
        # Pearson correlation
        r_val, p_val = sp_stats.pearsonr(np.log(ppls_g + 1), L0s_g)
        # Spearman
        rho, p_rho = sp_stats.spearmanr(ppls_g, L0s_g)
    else:
        r_val, p_val = 0, 1
        rho, p_rho = 0, 1

    # Linear fit
    if len(ppls_g) >= 3:
        slope, intercept, _, _, _ = sp_stats.linregress(np.log(ppls_g + 1), L0s_g)
    else:
        slope, intercept = 0, 20

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) L0 vs PPL scatter
    scatter = axes[0].scatter(ppls_g, L0s_g, s=80, c=L0s_g, cmap='coolwarm',
                             edgecolors='black', zorder=5)
    if len(ppls_g) >= 3:
        ppl_fit = np.linspace(min(ppls_g), max(ppls_g), 100)
        axes[0].plot(ppl_fit, slope * np.log(ppl_fit + 1) + intercept,
                    '--', color='gray', linewidth=2,
                    label=f'$r={r_val:.3f}$ ($p={p_val:.3f}$)')
    axes[0].set_xlabel('Perplexity')
    axes[0].set_ylabel('$L_0$ (transition point)')
    axes[0].set_title(f'(a) $L_0$ vs PPL ($\\rho={rho:.3f}$)')
    axes[0].legend(fontsize=8)

    # (b) L0 distribution
    axes[1].hist(L0s_g, bins=10, color='#8e44ad', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=np.mean(L0s_g), color='#f39c12', linewidth=2, linestyle='--',
                    label=f'Mean={np.mean(L0s_g):.1f}')
    axes[1].set_xlabel('$L_0$')
    axes[1].set_ylabel('Count')
    axes[1].set_title('(b) L0 Distribution')
    axes[1].legend()

    # (c) Summary
    summary = (
        f"L0-PPL Correlation\n\n"
        f"N prompts (good fit): {len(ppls_g)}\n\n"
        f"Pearson r: {r_val:.3f} (p={p_val:.3f})\n"
        f"Spearman rho: {rho:.3f} (p={p_rho:.3f})\n\n"
        f"L0 range: [{np.min(L0s_g):.1f}, {np.max(L0s_g):.1f}]\n"
        f"PPL range: [{np.min(ppls_g):.1f}, {np.max(ppls_g):.1f}]\n\n"
        f"Slope: {slope:.3f}\n"
        f"{'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'}"
    )
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('(c) Summary')

    fig.suptitle(f'Phase 111: L0-Perplexity Correlation ($\\rho={rho:.3f}$, '
                 f'{"SIG" if p_val < 0.05 else "NS"})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase111_L0_ppl')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pearson r={r_val:.3f} (p={p_val:.3f})")
    print(f"Spearman rho={rho:.3f} (p={p_rho:.3f})")
    print(f"L0 range: [{np.min(L0s_g):.1f}, {np.max(L0s_g):.1f}]")
    print(f"{'='*70}")

    save_results('phase111_L0_ppl', {
        'experiment': 'L0 vs Perplexity Correlation',
        'results': results,
        'summary': {
            'pearson_r': float(r_val),
            'pearson_p': float(p_val),
            'spearman_rho': float(rho),
            'spearman_p': float(p_rho),
            'slope': float(slope),
            'n_good': len(ppls_g),
            'L0_range': [float(np.min(L0s_g)), float(np.max(L0s_g))],
        }
    })


if __name__ == '__main__':
    main()
