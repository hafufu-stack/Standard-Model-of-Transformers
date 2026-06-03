# -*- coding: utf-8 -*-
"""
Phase 25: FFN Dark Energy Suppression
=======================================
Suppress FFN output to make Attention (gravity) dominant.
Does reducing "dark energy" make the model more deterministic/factual?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, measure_full_thermodynamics, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 25: FFN Dark Energy Suppression")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The speed of light in vacuum is approximately",
        "The largest planet in our solar system is",
    ]

    betas = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
    results_per_beta = {}

    for beta in betas:
        print(f"\n--- FFN suppression beta={beta} ---")
        handles = []
        if beta < 1.0:
            def make_ffn_suppress_hook(scale):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (output[0] * scale,) + output[1:]
                    return output * scale
                return hook
            for li in range(n_layers):
                h = model.model.layers[li].mlp.register_forward_hook(
                    make_ffn_suppress_hook(beta))
                handles.append(h)

        thermo_all = []
        top_tokens_all = []
        top_probs_all = []
        for p in prompts:
            thermo, out = measure_full_thermodynamics(model, tok, p, device)
            thermo_all.append(thermo)
            logits = out.logits[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top5_idx = torch.topk(probs, 5).indices
            top5_p = torch.topk(probs, 5).values
            words = [tok.decode(idx.item()) for idx in top5_idx]
            top_tokens_all.append(words)
            top_probs_all.append(top5_p[0].item())
            print(f"  '{p}' -> {words[0]} (p={top5_p[0].item():.3f})")

        for h in handles:
            h.remove()

        avg_T = np.mean([t[-1]['T'] for t in thermo_all])
        avg_U = np.mean([t[-1]['U'] for t in thermo_all])
        avg_top_p = np.mean(top_probs_all)
        results_per_beta[beta] = {
            'T': avg_T, 'U': avg_U, 'top_p': avg_top_p,
            'tokens': top_tokens_all,
        }

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    beta_list = sorted(results_per_beta.keys(), reverse=True)

    ax = axes[0]
    ax.plot(beta_list, [results_per_beta[b]['T'] for b in beta_list], 'o-', color='#e74c3c', ms=6)
    ax.set_xlabel('FFN Scale (beta)')
    ax.set_ylabel('Final Temperature T')
    ax.set_title('(a) Temperature vs FFN Suppression')
    ax.invert_xaxis()

    ax = axes[1]
    ax.plot(beta_list, [results_per_beta[b]['top_p'] for b in beta_list], 'o-', color='#3498db', ms=6)
    ax.set_xlabel('FFN Scale (beta)')
    ax.set_ylabel('Top-1 Probability')
    ax.set_title('(b) Confidence vs FFN Suppression')
    ax.invert_xaxis()

    ax = axes[2]
    ax.plot(beta_list, [results_per_beta[b]['U'] for b in beta_list], 'o-', color='#2ecc71', ms=6)
    ax.set_xlabel('FFN Scale (beta)')
    ax.set_ylabel('Internal Energy U')
    ax.set_title('(c) Energy vs FFN Suppression')
    ax.invert_xaxis()

    t_change = (results_per_beta[0.0]['T'] - results_per_beta[1.0]['T']) / (results_per_beta[1.0]['T'] + 1e-10) * 100
    p_change = (results_per_beta[0.0]['top_p'] - results_per_beta[1.0]['top_p']) / (results_per_beta[1.0]['top_p'] + 1e-10) * 100

    fig.suptitle(
        f"Phase 25: FFN Dark Energy Suppression\n"
        f"T change: {t_change:+.0f}% | Confidence change: {p_change:+.0f}%",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase25_ffn_suppression")
    plt.close()

    if results_per_beta[0.1]['top_p'] > results_per_beta[1.0]['top_p'] * 1.2:
        verdict = (f"DARK ENERGY SUPPRESSION WORKS: Confidence rises {p_change:+.0f}% "
                   f"when FFN is suppressed. Gravity dominance = determinism!")
    else:
        verdict = (f"COMPLEX INTERACTION: Confidence changes {p_change:+.0f}% with FFN suppression. "
                   f"Dark energy is necessary for coherent output.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase25_ffn_suppression", {
        'name': 'Phase 25: FFN Dark Energy Suppression',
        'summary': {'verdict': verdict, 't_change_pct': t_change, 'p_change_pct': p_change,
                    'per_beta': {str(b): {'T': results_per_beta[b]['T'], 'top_p': results_per_beta[b]['top_p']}
                                 for b in betas}},
    })


if __name__ == '__main__':
    main()
