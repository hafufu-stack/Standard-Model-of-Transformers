# -*- coding: utf-8 -*-
"""
Phase 21: Lyapunov Chaos and the Butterfly Effect
====================================================
Compute the Lyapunov exponent of the Transformer by measuring
how fast a tiny perturbation diverges across layers.
Uses Deep Think's safe_noise_hook to prevent fp16 NaN.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 21: Lyapunov Chaos and the Butterfly Effect")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The laws of physics are invariant under translations",
        "Deep learning models approximate complex functions",
        "The structure of DNA encodes genetic information",
        "Stars evolve through nuclear fusion reactions",
        "The brain creates consciousness from neural activity",
    ]

    sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    all_lyapunov = {}

    for sigma in sigmas:
        print(f"\n--- sigma = {sigma} ---")
        all_divergence = []

        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)

            # Baseline trajectory
            with torch.no_grad():
                base_out = model(**inp, output_hidden_states=True)
            base_hs = [h[0, -1, :].float().cpu() for h in base_out.hidden_states]

            # Perturbed trajectory: inject noise at embedding level
            with torch.no_grad():
                embeddings = model.model.embed_tokens(inp['input_ids']).float()
                noise = torch.randn_like(embeddings) * sigma
                emb_perturbed = (embeddings + noise).to(
                    next(model.model.embed_tokens.parameters()).dtype)
                emb_perturbed = torch.nan_to_num(emb_perturbed, nan=0.0,
                                                  posinf=65000.0, neginf=-65000.0)

                pert_out = model(
                    inputs_embeds=emb_perturbed,
                    attention_mask=inp.get('attention_mask'),
                    output_hidden_states=True,
                )
            pert_hs = [h[0, -1, :].float().cpu() for h in pert_out.hidden_states]

            # Measure divergence at each layer
            divergence = []
            for l in range(len(base_hs)):
                diff = (pert_hs[l] - base_hs[l]).norm().item()
                base_norm = base_hs[l].norm().item()
                # Relative divergence
                rel_div = diff / (base_norm + 1e-10)
                divergence.append(max(rel_div, 1e-15))  # floor for log

            all_divergence.append(divergence)

        # Average divergence
        avg_div = np.mean(all_divergence, axis=0)
        all_lyapunov[sigma] = avg_div

        # Fit exponential: divergence ~ exp(lambda * l)
        layers = np.arange(len(avg_div))
        try:
            log_div = np.log(avg_div + 1e-15)
            valid = np.isfinite(log_div)
            if valid.sum() > 3:
                z = np.polyfit(layers[valid], log_div[valid], 1)
                lyap = z[0]
            else:
                lyap = 0
        except Exception:
            lyap = 0

        print(f"  Lyapunov exponent lambda = {lyap:.4f} "
              f"({'CHAOS' if lyap > 0 else 'STABLE'})")
        all_lyapunov[f'lambda_{sigma}'] = lyap

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Divergence trajectories
    ax = axes[0]
    cmap = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigmas)))
    for idx, sigma in enumerate(sigmas):
        div = all_lyapunov[sigma]
        ax.semilogy(range(len(div)), div, 'o-', ms=3, color=cmap[idx],
                    label=f'sigma={sigma}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Relative Divergence')
    ax.set_title('(a) Butterfly Effect: Divergence vs Layer')
    ax.legend(fontsize=7)

    # (b) Lyapunov exponents
    ax = axes[1]
    lambdas = [all_lyapunov[f'lambda_{s}'] for s in sigmas]
    ax.bar(range(len(sigmas)), lambdas, color=cmap, alpha=0.8)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'{s}' for s in sigmas], fontsize=8)
    ax.set_xlabel('Initial Perturbation sigma')
    ax.set_ylabel('Lyapunov Exponent lambda')
    ax.axhline(y=0, color='red', ls='--', label='Chaos boundary')
    ax.set_title('(b) Lyapunov Exponents')
    ax.legend()

    # (c) Phase portrait: div at layer 5 vs div at layer 25
    ax = axes[2]
    for idx, sigma in enumerate(sigmas):
        div = all_lyapunov[sigma]
        if len(div) > 25:
            ax.scatter(div[5], div[25], s=100, color=cmap[idx], label=f'sigma={sigma}',
                       zorder=5)
    ax.plot([1e-10, 1], [1e-10, 1], 'k--', alpha=0.3, label='No amplification')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Divergence at Layer 5')
    ax.set_ylabel('Divergence at Layer 25')
    ax.set_title('(c) Amplification Factor')
    ax.legend(fontsize=7)

    mean_lambda = np.mean(lambdas)
    fig.suptitle(
        f"Phase 21: Lyapunov Chaos\n"
        f"Mean lambda = {mean_lambda:.4f} | "
        f"{'CHAOTIC (lambda > 0)' if mean_lambda > 0 else 'STABLE (lambda < 0)'}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase21_lyapunov_chaos")
    plt.close()

    if mean_lambda > 0.05:
        verdict = (f"DETERMINISTIC CHAOS: lambda={mean_lambda:.4f}. "
                   f"LLM is mathematically chaotic! "
                   f"Tiny input changes grow exponentially across layers.")
    elif mean_lambda > 0:
        verdict = (f"EDGE OF CHAOS: lambda={mean_lambda:.4f}. "
                   f"LLM operates at the boundary between order and chaos.")
    else:
        verdict = (f"STABLE DYNAMICS: lambda={mean_lambda:.4f}. "
                   f"Perturbations decay - the attractor is stable.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 21: Lyapunov Chaos',
        'summary': {'verdict': verdict, 'mean_lambda': mean_lambda,
                    'lambdas': {str(s): all_lyapunov[f'lambda_{s}'] for s in sigmas}},
    }
    save_results("phase21_lyapunov_chaos", result)
    return result


if __name__ == '__main__':
    main()
