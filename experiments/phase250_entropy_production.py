# -*- coding: utf-8 -*-
"""
Phase 250: Thermodynamic Entropy Production Rate
===================================================
Measure the entropy production rate sigma = dS/dl at each layer.
According to the second law, sigma >= 0 for irreversible processes.
This quantifies HOW MUCH the arrow of time is enforced at each layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "Chemical reactions follow conservation of mass",
    "The brain contains billions of neurons",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
    "Seven abstract thoughts collided creating new",
    "The moon decided to become a dancer",
    "Random words create unpredictable sequences when",
]


def entropy_production(model, tok, device, model_name):
    """Measure entropy production rate at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_sigma = []
    all_T = []
    all_kl_adjacent = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get distributions at each layer
        dist_layers = []
        T_layers = []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            dist_layers.append(probs)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_layers.append(float(S) if not np.isnan(S) else 0)

        # Entropy production rate: dS/dl
        sigma = []
        kl_adj = []
        for i in range(len(T_layers) - 1):
            dS = T_layers[i+1] - T_layers[i]
            sigma.append(dS)
            # KL divergence between adjacent layers
            kl = torch.nn.functional.kl_div(
                torch.log(dist_layers[i] + 1e-10),
                dist_layers[i+1], reduction='sum').item()
            kl_adj.append(float(kl) if not np.isnan(kl) else 0)

        all_sigma.append(sigma)
        all_T.append(T_layers)
        all_kl_adjacent.append(kl_adj)

    n = min(len(s) for s in all_sigma)
    avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]
    std_fn = lambda d: [float(np.std([d[p][i] for p in range(len(d))])) for i in range(n)]
    
    mean_sigma = avg(all_sigma)
    std_sigma = std_fn(all_sigma)
    mean_kl = avg(all_kl_adjacent)
    mean_T = avg(all_T)

    # Fraction of negative sigma (second law violations)
    negative_frac = [sum(1 for d in all_sigma if d[i] < 0) / len(all_sigma) for i in range(n)]

    # Total entropy production
    total_sigma = sum(mean_sigma)

    # Maximum production layer
    max_sigma_layer = int(np.argmax(np.abs(mean_sigma)))

    return {
        'model': model_name,
        'mean_sigma': mean_sigma,
        'std_sigma': std_sigma,
        'mean_kl_adjacent': mean_kl,
        'mean_T': mean_T,
        'negative_frac': negative_frac,
        'total_sigma': total_sigma,
        'max_sigma_layer': max_sigma_layer,
    }


def main():
    print("=" * 70)
    print("Phase 250: Entropy Production Rate")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = entropy_production(model, tok, device, size)
        results[size] = r
        print(f"  Total sigma: {r['total_sigma']:.3f}")
        print(f"  Max |sigma| at layer: {r['max_sigma_layer']}")
        print(f"  Neg frac range: {min(r['negative_frac']):.2f} - {max(r['negative_frac']):.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) sigma(l)
    for size, r in results.items():
        c = colors[size]
        x = range(len(r['mean_sigma']))
        axes[0, 0].plot(x, r['mean_sigma'], '-', color=c, lw=2, label=size)
        axes[0, 0].fill_between(x,
                                np.array(r['mean_sigma']) - np.array(r['std_sigma']),
                                np.array(r['mean_sigma']) + np.array(r['std_sigma']),
                                color=c, alpha=0.15)
    axes[0, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 0].set_xlabel('Layer Transition')
    axes[0, 0].set_ylabel('dS/dl (entropy production)')
    axes[0, 0].set_title('(a) Entropy Production Rate')
    axes[0, 0].legend(fontsize=8)

    # (b) KL divergence between adjacent layers
    for size, r in results.items():
        axes[0, 1].plot(range(len(r['mean_kl_adjacent'])), r['mean_kl_adjacent'],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer Transition')
    axes[0, 1].set_ylabel('KL(p_l || p_{l+1})')
    axes[0, 1].set_title('(b) Adjacent KL Divergence')
    axes[0, 1].legend(fontsize=8)

    # (c) Fraction of "violations" (sigma < 0)
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['negative_frac'])), r['negative_frac'],
                       '-o', color=colors[size], lw=2, markersize=3, label=size)
    axes[0, 2].axhline(y=0.5, color='gray', ls='--', alpha=0.3, label='50%')
    axes[0, 2].set_xlabel('Layer Transition')
    axes[0, 2].set_ylabel('P(sigma < 0)')
    axes[0, 2].set_title('(c) Second Law "Violation" Rate')
    axes[0, 2].legend(fontsize=7)

    # (d) Cumulative sigma
    for size, r in results.items():
        cum = np.cumsum(r['mean_sigma'])
        axes[1, 0].plot(range(len(cum)), cum, '-', color=colors[size], lw=2, label=size)
    axes[1, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Cumulative sigma')
    axes[1, 0].set_title('(d) Cumulative Entropy Production')
    axes[1, 0].legend(fontsize=8)

    # (e) |sigma| vs KL
    for size, r in results.items():
        axes[1, 1].scatter(r['mean_kl_adjacent'], [abs(s) for s in r['mean_sigma']],
                          color=colors[size], s=30, alpha=0.7, label=size)
    axes[1, 1].set_xlabel('KL(adjacent)')
    axes[1, 1].set_ylabel('|sigma|')
    axes[1, 1].set_title('(e) |sigma| vs KL')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "ENTROPY PRODUCTION\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Total sigma: {r['total_sigma']:.3f}\n"
        summary += f"  Max at layer: {r['max_sigma_layer']}\n"
        neg_mean = np.mean(r['negative_frac'])
        summary += f"  Avg neg frac: {neg_mean:.2f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 250: Entropy Production Rate",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase250_entropy_production')
    plt.close()
    save_results('phase250_entropy_production', {
        'experiment': 'Entropy Production',
        'results': results,
    })


if __name__ == '__main__':
    main()
