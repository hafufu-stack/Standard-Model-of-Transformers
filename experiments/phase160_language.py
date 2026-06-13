# -*- coding: utf-8 -*-
"""
Phase 160: Language Dependence of Phase Transition
Do different natural languages have different L0?
Test English, Chinese, Japanese, and code.
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


LANGUAGES = {
    'english': [
        "The fundamental theorem of calculus connects",
        "Black holes form from gravitational collapse of",
        "Machine learning discovers hidden patterns in",
        "The speed of light is approximately three hundred",
    ],
    'chinese': [
        "The Chinese characters for water are written as",
        "In Chinese philosophy the concept of yin and yang",
        "The Great Wall of China was built during the",
        "Traditional Chinese medicine uses herbs and acupuncture",
    ],
    'french': [
        "La tour Eiffel est un monument situe dans la ville de",
        "La revolution francaise a commence en mille sept cent",
        "Le fromage francais est connu dans le monde entier pour",
        "Les mathematiques sont une science fondamentale qui etudie",
    ],
    'python_code': [
        "def quicksort(arr): if len(arr) <= 1: return arr; pivot =",
        "import numpy as np; x = np.linspace(0, 1, 100); y =",
        "class NeuralNetwork(nn.Module): def __init__(self, hidden_size",
        "for i in range(len(data)): if data[i] > threshold: result",
    ],
    'math_notation': [
        "The integral from zero to infinity of e to the negative x squared",
        "The eigenvalue equation A times v equals lambda times v where",
        "The gradient of the loss function with respect to the weights",
        "The probability distribution P of x given theta equals",
    ],
}


def main():
    print("=" * 70)
    print("Phase 160: Language Dependence")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    lang_results = {}
    for lang, prompts in LANGUAGES.items():
        all_eta = [[] for _ in range(n_layers)]
        all_S = [[] for _ in range(n_layers)]

        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
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

        lang_results[lang] = {
            'L0': float(L0),
            'R2': float(r2),
            'eta': avg_eta,
            'S': avg_S,
            'final_eta': float(avg_eta[-1]),
            'final_S': float(avg_S[-1]),
        }
        print(f"  {lang}: L0={L0:.1f}, R2={r2:.3f}, S_final={avg_S[-1]:.2f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'english': '#2980b9', 'chinese': '#c0392b', 'french': '#27ae60',
              'python_code': '#f39c12', 'math_notation': '#8e44ad'}
    layers = np.arange(n_layers)

    # (a) Eta profiles
    for lang, r in lang_results.items():
        axes[0,0].plot(layers, r['eta'], 'o-', color=colors.get(lang, 'gray'),
                      markersize=3, linewidth=2,
                      label=f"{lang} (L0={r['L0']:.1f})")
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Language-Specific Eta')
    axes[0,0].legend(fontsize=7)

    # (b) L0 comparison
    lang_names = list(lang_results.keys())
    L0s = [lang_results[l]['L0'] for l in lang_names]
    bar_c = [colors.get(l, 'gray') for l in lang_names]
    # Filter out diverged fits
    valid = [(l, L0) for l, L0 in zip(lang_names, L0s) if 0 < L0 < n_layers * 2]
    if valid:
        v_names, v_L0s = zip(*valid)
        axes[0,1].bar(range(len(v_names)), v_L0s,
                      color=[colors.get(l, 'gray') for l in v_names],
                      alpha=0.8, edgecolor='black')
        axes[0,1].set_xticks(range(len(v_names)))
        axes[0,1].set_xticklabels(v_names, fontsize=8, rotation=15)
        axes[0,1].set_ylabel('$L_0$')
        axes[0,1].set_title('(b) Critical Point by Language')

    # (c) S profiles
    for lang, r in lang_results.items():
        axes[0,2].plot(layers, r['S'], 'o-', color=colors.get(lang, 'gray'),
                      markersize=3, linewidth=2, label=lang)
    axes[0,2].axvline(x=21.7, color='gray', linewidth=1, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$S$')
    axes[0,2].set_title('(c) Entropy by Language')
    axes[0,2].legend(fontsize=7)

    # (d) Final S
    fSs = [lang_results[l]['final_S'] for l in lang_names]
    axes[1,0].bar(range(len(lang_names)), fSs,
                  color=[colors.get(l, 'gray') for l in lang_names],
                  alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(lang_names)))
    axes[1,0].set_xticklabels(lang_names, fontsize=8, rotation=15)
    axes[1,0].set_ylabel('$S_{final}$')
    axes[1,0].set_title('(d) Final Entropy')

    # (e) L0 vs final_S
    for l in lang_names:
        r = lang_results[l]
        if 0 < r['L0'] < n_layers * 2:
            axes[1,1].scatter(r['L0'], r['final_S'], c=colors.get(l, 'gray'),
                             s=150, edgecolors='black', zorder=5, label=l)
    axes[1,1].set_xlabel('$L_0$')
    axes[1,1].set_ylabel('$S_{final}$')
    axes[1,1].set_title('(e) L0 vs Final Entropy')
    axes[1,1].legend(fontsize=7)

    # (f) Summary
    valid_L0s = [L0 for L0 in L0s if 0 < L0 < n_layers * 2]
    L0_cv = np.std(valid_L0s) / (np.mean(valid_L0s) + 1e-10) if valid_L0s else 0
    summary = (
        f"Language Dependence\n\n"
        + "\n".join(f"{l}: L0={lang_results[l]['L0']:.1f} "
                    f"(S={lang_results[l]['final_S']:.2f})"
                    for l in lang_names)
        + f"\n\nL0 CV: {L0_cv:.3f}\n"
        f"Phase transition is\n"
        f"{'LANGUAGE-DEPENDENT' if L0_cv > 0.05 else 'LANGUAGE-UNIVERSAL'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 160: Language Dependence of Phase Transition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase160_language')
    plt.close()

    print(f"\n{'='*70}")
    print(f"L0 CV={L0_cv:.3f}")
    print(f"{'='*70}")

    save_results('phase160_language', {
        'experiment': 'Language Dependence',
        'results': {l: {k: v for k, v in r.items() if k not in ['eta', 'S']}
                    for l, r in lang_results.items()},
    })


if __name__ == '__main__':
    main()
