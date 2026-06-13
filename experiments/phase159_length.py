# -*- coding: utf-8 -*-
"""
Phase 159: Prompt Length Scaling
How does prompt length affect the phase transition?
Does L0 shift, does eta change, does the cooling valley deepen?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


# Base prompts that we can extend
BASE_PROMPTS = [
    "The capital of France is",
    "Neural networks learn through",
    "The fundamental theorem of calculus",
    "Quantum mechanics describes particles",
]

# Extended versions (progressively longer)
def make_prompt(base, n_tokens_approx):
    """Create prompts of different lengths by repeating context."""
    extensions = {
        'short': "",
        'medium': " which is a well-known fact in the field of science and mathematics that has been studied extensively",
        'long': " which is a well-known fact in the field of science and mathematics that has been studied extensively by researchers around the world for many decades and continues to be an active area of investigation with many open questions remaining",
        'very_long': " which is a well-known fact in the field of science and mathematics that has been studied extensively by researchers around the world for many decades and continues to be an active area of investigation with many open questions remaining to be answered through careful experimentation and theoretical analysis using the most advanced computational tools available today including machine learning and artificial intelligence systems that can process vast amounts of data",
    }
    return base + extensions.get(n_tokens_approx, "")


def main():
    print("=" * 70)
    print("Phase 159: Prompt Length Scaling")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    lengths = ['short', 'medium', 'long', 'very_long']
    length_results = {}

    for length in lengths:
        all_eta = [[] for _ in range(n_layers)]
        all_S = [[] for _ in range(n_layers)]
        token_counts = []

        for base in BASE_PROMPTS:
            prompt = make_prompt(base, length)
            inp = tok(prompt, return_tensors='pt').to(device)
            token_counts.append(inp['input_ids'].shape[1])

            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            T_vals = []
            for li in range(n_layers):
                hs = out.hidden_states[li]
                with torch.no_grad():
                    normed = model.model.norm(hs[:, -1:, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                S = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_vals.append(S if not np.isnan(S) else 0)
                all_S[li].append(T_vals[-1])

            for li in range(n_layers):
                T_sub = T_vals[:li+1]
                if len(T_sub) >= 4:
                    T_hot = max(T_sub)
                    T_cold = min(T_sub[len(T_sub)//2:])
                    eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
                else:
                    eta = 0
                all_eta[li].append(eta)

        avg_eta = [np.mean(v) if v else 0 for v in all_eta]
        avg_S = [np.mean(v) if v else 0 for v in all_S]

        try:
            Ls = np.arange(4, n_layers)
            popt, _ = curve_fit(sigmoid, Ls, avg_eta[4:],
                                p0=[22, 0.5, 0, 0.9], maxfev=10000)
            L0 = popt[0]
            r2 = 1 - np.sum((np.array(avg_eta[4:]) - sigmoid(Ls, *popt))**2) / (
                np.sum((np.array(avg_eta[4:]) - np.mean(avg_eta[4:]))**2) + 1e-10)
        except:
            L0 = 22
            r2 = 0

        mean_tokens = np.mean(token_counts)
        length_results[length] = {
            'L0': float(L0),
            'R2': float(r2),
            'eta': avg_eta,
            'S': avg_S,
            'mean_tokens': float(mean_tokens),
            'final_eta': float(avg_eta[-1]),
            'final_S': float(avg_S[-1]),
        }
        print(f"  {length} (~{mean_tokens:.0f} tokens): L0={L0:.1f}, R2={r2:.3f}, S_final={avg_S[-1]:.2f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'short': '#e74c3c', 'medium': '#f39c12', 'long': '#2980b9', 'very_long': '#8e44ad'}
    layers = np.arange(n_layers)

    # (a) Eta profiles
    for length, r in length_results.items():
        axes[0,0].plot(layers, r['eta'], 'o-', color=colors[length], markersize=3,
                      linewidth=2, label=f"{length} ({r['mean_tokens']:.0f}t, L0={r['L0']:.1f})")
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Eta vs Prompt Length')
    axes[0,0].legend(fontsize=7)

    # (b) S profiles
    for length, r in length_results.items():
        axes[0,1].plot(layers, r['S'], 'o-', color=colors[length], markersize=3,
                      linewidth=2, label=length)
    axes[0,1].axvline(x=21.7, color='gray', linewidth=1, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$S$')
    axes[0,1].set_title('(b) Entropy vs Length')
    axes[0,1].legend(fontsize=8)

    # (c) L0 vs token count
    toks = [length_results[l]['mean_tokens'] for l in lengths]
    L0s = [length_results[l]['L0'] for l in lengths]
    bar_c = [colors[l] for l in lengths]
    axes[0,2].scatter(toks, L0s, c=bar_c, s=150, edgecolors='black', zorder=5)
    for i, l in enumerate(lengths):
        axes[0,2].annotate(l, (toks[i], L0s[i]), xytext=(5, 5),
                          textcoords='offset points', fontsize=8)
    axes[0,2].set_xlabel('Token Count')
    axes[0,2].set_ylabel('$L_0$')
    axes[0,2].set_title('(c) L0 vs Prompt Length')

    # (d) Final eta
    fetas = [length_results[l]['final_eta'] for l in lengths]
    axes[1,0].bar(range(len(lengths)), fetas, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(lengths)))
    axes[1,0].set_xticklabels(lengths, fontsize=9)
    axes[1,0].set_ylabel('$\\eta_{final}$')
    axes[1,0].set_title('(d) Final Eta by Length')

    # (e) Final S
    fSs = [length_results[l]['final_S'] for l in lengths]
    axes[1,1].bar(range(len(lengths)), fSs, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(lengths)))
    axes[1,1].set_xticklabels(lengths, fontsize=9)
    axes[1,1].set_ylabel('$S_{final}$')
    axes[1,1].set_title('(e) Final Entropy by Length')

    # (f) Summary
    L0_range = max(L0s) - min(L0s)
    summary = (
        f"Prompt Length Scaling\n\n"
        + "\n".join(f"{l}: {length_results[l]['mean_tokens']:.0f}t, "
                    f"L0={length_results[l]['L0']:.1f}"
                    for l in lengths)
        + f"\n\nL0 range: {L0_range:.1f}\n"
        f"L0 {'SHIFTS' if L0_range > 2 else 'STABLE'} with length\n\n"
        f"S_final {'DECREASES' if fSs[-1] < fSs[0] else 'INCREASES'}\n"
        f"with prompt length"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 159: Prompt Length Scaling',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase159_length')
    plt.close()

    print(f"\n{'='*70}")
    print(f"L0 range: {L0_range:.1f}")
    print(f"{'='*70}")

    save_results('phase159_length', {
        'experiment': 'Prompt Length Scaling',
        'results': {l: {k: v for k, v in r.items() if k not in ['eta', 'S']}
                    for l, r in length_results.items()},
    })


if __name__ == '__main__':
    main()
