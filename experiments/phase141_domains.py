# -*- coding: utf-8 -*-
"""
Phase 141: LoRA Thermodynamics (Simplified)
Instead of actual LoRA finetuning, compare thermodynamic profiles
across different prompt CATEGORIES (math, code, natural language, etc.)
to see how "specialization" affects the energy landscape.
If different domains have different L0, the model has learned
domain-specific phase transitions.
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


DOMAINS = {
    'math': [
        "The integral of x squared from zero to one equals",
        "The derivative of sine x is cosine x because",
        "The eigenvalues of a symmetric matrix are always",
        "The Fourier transform converts signals from time domain to",
    ],
    'science': [
        "Mitochondria are known as the powerhouse of the cell because",
        "The Heisenberg uncertainty principle states that",
        "Black holes emit Hawking radiation due to quantum effects near",
        "The speed of light in vacuum is exactly",
    ],
    'code': [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) +",
        "The time complexity of binary search is O(log n) because",
        "In Python, a list comprehension is more efficient than a for loop when",
        "The difference between a stack and a queue is that",
    ],
    'language': [
        "The capital of France is Paris and it is known for",
        "Shakespeare wrote many plays including Hamlet which tells the story of",
        "The Declaration of Independence was signed in the year",
        "The largest ocean on Earth is the Pacific which covers",
    ],
    'philosophy': [
        "Descartes said I think therefore I am which means that",
        "The trolley problem asks whether it is moral to",
        "Plato's allegory of the cave suggests that reality is",
        "The meaning of existence according to existentialism is",
    ],
}


def main():
    print("=" * 70)
    print("Phase 141: Domain-Specific Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    domain_results = {}
    for domain, prompts in DOMAINS.items():
        print(f"\n  Domain: {domain}")
        all_S = [[] for _ in range(n_layers)]
        all_eta = [[] for _ in range(n_layers)]

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
                T_subset = T_vals[:li+1]
                if len(T_subset) >= 4:
                    T_hot = max(T_subset)
                    T_cold = min(T_subset[len(T_subset)//2:])
                    eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
                else:
                    eta = 0
                all_eta[li].append(eta)

        avg_eta = [np.mean(v) if v else 0 for v in all_eta]
        avg_S = [np.mean(v) if v else 0 for v in all_S]

        # Fit sigmoid
        try:
            Ls = np.arange(4, n_layers)
            popt, _ = curve_fit(sigmoid, Ls, avg_eta[4:],
                                p0=[22, 0.5, 0, 0.9], maxfev=10000)
            L0 = popt[0]
            sig_pred = sigmoid(Ls, *popt)
            r2 = 1 - np.sum((np.array(avg_eta[4:]) - sig_pred)**2) / (
                np.sum((np.array(avg_eta[4:]) - np.mean(avg_eta[4:]))**2) + 1e-10)
        except:
            L0 = 22
            r2 = 0

        domain_results[domain] = {
            'L0': float(L0),
            'L0_ratio': float(L0 / n_layers),
            'R2': float(r2),
            'eta': avg_eta,
            'S': avg_S,
            'final_S': float(avg_S[-1]),
            'final_eta': float(avg_eta[-1]),
        }
        print(f"    L0={L0:.1f}, R2={r2:.3f}, S_final={avg_S[-1]:.2f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'math': '#c0392b', 'science': '#2980b9', 'code': '#27ae60',
              'language': '#f39c12', 'philosophy': '#8e44ad'}
    layers = np.arange(n_layers)

    # (a) Eta profiles
    for domain, r in domain_results.items():
        axes[0,0].plot(layers, r['eta'], 'o-', color=colors[domain], markersize=3,
                      linewidth=2, label=f"{domain} (L0={r['L0']:.1f})")
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Domain-Specific Eta')
    axes[0,0].legend(fontsize=7)

    # (b) L0 comparison
    dom_names = list(domain_results.keys())
    L0s = [domain_results[d]['L0'] for d in dom_names]
    r2s = [domain_results[d]['R2'] for d in dom_names]
    bar_c = [colors[d] for d in dom_names]
    axes[0,1].bar(range(len(dom_names)), L0s, color=bar_c, alpha=0.8, edgecolor='black')
    axes[0,1].set_xticks(range(len(dom_names)))
    axes[0,1].set_xticklabels(dom_names, fontsize=9)
    mean_L0 = np.mean(L0s)
    axes[0,1].axhline(y=mean_L0, color='black', linestyle='--',
                      label=f'Mean={mean_L0:.1f}')
    axes[0,1].set_ylabel('$L_0$')
    axes[0,1].set_title('(b) Critical Point by Domain')
    axes[0,1].legend()

    # (c) S profiles
    for domain, r in domain_results.items():
        axes[0,2].plot(layers, r['S'], 'o-', color=colors[domain], markersize=3,
                      linewidth=2, label=domain)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$S$')
    axes[0,2].set_title('(c) Entropy Profiles')
    axes[0,2].legend(fontsize=7)

    # (d) Final S by domain
    final_S = [domain_results[d]['final_S'] for d in dom_names]
    axes[1,0].bar(range(len(dom_names)), final_S, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(dom_names)))
    axes[1,0].set_xticklabels(dom_names, fontsize=9)
    axes[1,0].set_ylabel('$S_{final}$')
    axes[1,0].set_title('(d) Final Entropy by Domain')

    # (e) L0 vs final_S scatter
    for d in dom_names:
        axes[1,1].scatter(domain_results[d]['L0'], domain_results[d]['final_S'],
                         c=colors[d], s=150, edgecolors='black', zorder=5, label=d)
    axes[1,1].set_xlabel('$L_0$')
    axes[1,1].set_ylabel('$S_{final}$')
    axes[1,1].set_title('(e) L0 vs Final Entropy')
    axes[1,1].legend(fontsize=8)

    # (f) Summary
    L0_cv = np.std(L0s) / (np.mean(L0s) + 1e-10)
    summary = (
        f"Domain-Specific Thermodynamics\n\n"
        + "\n".join(f"{d}: L0={domain_results[d]['L0']:.1f} "
                    f"(R2={domain_results[d]['R2']:.3f})"
                    for d in dom_names)
        + f"\n\nL0 CV: {L0_cv:.3f}\n"
        f"L0 range: [{min(L0s):.1f}, {max(L0s):.1f}]\n\n"
        f"Phase transition is\n"
        f"{'DOMAIN-SPECIFIC' if L0_cv > 0.05 else 'UNIVERSAL'}\n"
        f"({'varies' if L0_cv > 0.05 else 'constant'} across domains)"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 141: Domain-Specific Thermodynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase141_domains')
    plt.close()

    print(f"\n{'='*70}")
    for d in dom_names:
        print(f"  {d}: L0={domain_results[d]['L0']:.1f} (R2={domain_results[d]['R2']:.3f})")
    print(f"  L0 CV={L0_cv:.3f}")
    print(f"{'='*70}")

    save_results('phase141_domains', {
        'experiment': 'Domain-Specific Thermodynamics',
        'domain_results': {d: {k: v for k, v in r.items() if k not in ['eta', 'S']}
                           for d, r in domain_results.items()},
        'summary': {
            'L0_cv': float(L0_cv),
            'L0_range': [float(min(L0s)), float(max(L0s))],
        }
    })


if __name__ == '__main__':
    main()
