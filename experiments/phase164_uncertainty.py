# -*- coding: utf-8 -*-
"""
Phase 164: Uncertainty Decomposition
Decompose the output entropy into aleatoric (intrinsic randomness)
and epistemic (model uncertainty) components using dropout.
Does the ratio change across the phase transition?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The capital of France is",           # factual (low uncertainty)
    "The meaning of life is",             # philosophical (high uncertainty)
    "Two plus two equals",                # math (very low)
    "The best programming language is",   # opinion (high)
    "Water freezes at zero degrees",      # factual
    "In my opinion the most beautiful",   # subjective (high)
    "The speed of light in vacuum is",    # physics (low)
    "The future of AI will be",           # speculative (high)
]

EXPECTED_UNCERTAIN = [False, True, False, True, False, True, False, True]


def main():
    print("=" * 70)
    print("Phase 164: Uncertainty Decomposition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Enable dropout for MC sampling
    def enable_dropout(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def disable_dropout(model):
        model.eval()

    n_mc = 10  # MC samples
    results = []

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)

        # Deterministic pass
        model.eval()
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Layer-by-layer analysis
        layer_results = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S_det = -(probs * torch.log(probs + 1e-10)).sum().item()
            layer_results.append({
                'S_deterministic': S_det if not np.isnan(S_det) else 0,
                'probs_det': probs.cpu(),
            })

        # MC dropout passes for final layer
        enable_dropout(model)
        mc_probs = []
        for _ in range(n_mc):
            with torch.no_grad():
                out_mc = model(**inp, output_hidden_states=True)
            hs = out_mc.hidden_states[-1]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs_mc = torch.softmax(logits, dim=-1)
            mc_probs.append(probs_mc.cpu())
        disable_dropout(model)

        mc_stack = torch.stack(mc_probs)  # (n_mc, vocab)
        mean_probs = mc_stack.mean(dim=0)

        # Total uncertainty
        S_total = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()

        # Aleatoric: average of individual entropies
        individual_S = [-(p * torch.log(p + 1e-10)).sum().item() for p in mc_probs]
        S_aleatoric = np.mean(individual_S)

        # Epistemic: total - aleatoric (mutual information)
        S_epistemic = max(0, S_total - S_aleatoric)

        is_uncertain = EXPECTED_UNCERTAIN[pi]
        tag = "UNCERTAIN" if is_uncertain else "CERTAIN"

        results.append({
            'prompt': prompt,
            'S_total': float(S_total) if not np.isnan(S_total) else 0,
            'S_aleatoric': float(S_aleatoric) if not np.isnan(S_aleatoric) else 0,
            'S_epistemic': float(S_epistemic) if not np.isnan(S_epistemic) else 0,
            'is_uncertain': is_uncertain,
            'layer_S': [r['S_deterministic'] for r in layer_results],
        })

        print(f"  [{tag}] {prompt[:40]:40s} "
              f"total={S_total:.2f} = aleat={S_aleatoric:.2f} + epist={S_epistemic:.2f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Stacked bar: aleatoric + epistemic
    names = [r['prompt'][:25] for r in results]
    al = [r['S_aleatoric'] for r in results]
    ep = [r['S_epistemic'] for r in results]
    colors_unc = ['#c0392b' if r['is_uncertain'] else '#2980b9' for r in results]
    x = range(len(results))
    axes[0,0].bar(x, al, color='#3498db', alpha=0.7, label='Aleatoric', edgecolor='black')
    axes[0,0].bar(x, ep, bottom=al, color='#e74c3c', alpha=0.7, label='Epistemic', edgecolor='black')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    axes[0,0].set_ylabel('$S$')
    axes[0,0].set_title('(a) Uncertainty Decomposition')
    axes[0,0].legend()

    # (b) Epistemic fraction
    ep_frac = [r['S_epistemic'] / (r['S_total'] + 1e-10) for r in results]
    axes[0,1].bar(x, ep_frac, color=colors_unc, alpha=0.8, edgecolor='black')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(names, fontsize=6, rotation=45, ha='right')
    axes[0,1].set_ylabel('Epistemic Fraction')
    axes[0,1].set_title('(b) Epistemic / Total')

    # (c) S profiles (certain vs uncertain)
    for r in results:
        color = '#c0392b' if r['is_uncertain'] else '#2980b9'
        alpha = 0.5
        axes[0,2].plot(range(n_layers), r['layer_S'], '-', color=color,
                      linewidth=1.5, alpha=alpha)
    axes[0,2].plot([], [], '-', color='#c0392b', label='Uncertain')
    axes[0,2].plot([], [], '-', color='#2980b9', label='Certain')
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$S$')
    axes[0,2].set_title('(c) S Profiles by Type')
    axes[0,2].legend()

    # (d) Certain vs uncertain group comparison
    cert_total = np.mean([r['S_total'] for r in results if not r['is_uncertain']])
    cert_epist = np.mean([r['S_epistemic'] for r in results if not r['is_uncertain']])
    unc_total = np.mean([r['S_total'] for r in results if r['is_uncertain']])
    unc_epist = np.mean([r['S_epistemic'] for r in results if r['is_uncertain']])
    groups = ['Certain', 'Uncertain']
    totals = [cert_total, unc_total]
    epists = [cert_epist, unc_epist]
    axes[1,0].bar(groups, totals, color=['#2980b9', '#c0392b'], alpha=0.4,
                  label='Total', edgecolor='black')
    axes[1,0].bar(groups, epists, color=['#2980b9', '#c0392b'], alpha=0.8,
                  label='Epistemic', edgecolor='black')
    axes[1,0].set_ylabel('$S$')
    axes[1,0].set_title('(d) Group Comparison')
    axes[1,0].legend()

    # (e) Scatter: total vs epistemic
    for r in results:
        color = '#c0392b' if r['is_uncertain'] else '#2980b9'
        axes[1,1].scatter(r['S_total'], r['S_epistemic'], c=color, s=100,
                         edgecolors='black', zorder=5)
    axes[1,1].set_xlabel('$S_{total}$')
    axes[1,1].set_ylabel('$S_{epistemic}$')
    axes[1,1].set_title('(e) Total vs Epistemic')

    # (f) Summary
    cert_ep_frac = np.mean([r['S_epistemic'] / (r['S_total'] + 1e-10)
                           for r in results if not r['is_uncertain']])
    unc_ep_frac = np.mean([r['S_epistemic'] / (r['S_total'] + 1e-10)
                          for r in results if r['is_uncertain']])
    summary = (
        f"Uncertainty Decomposition\n\n"
        f"Certain prompts:\n"
        f"  S_total={cert_total:.2f}\n"
        f"  Epistemic frac={cert_ep_frac:.3f}\n\n"
        f"Uncertain prompts:\n"
        f"  S_total={unc_total:.2f}\n"
        f"  Epistemic frac={unc_ep_frac:.3f}\n\n"
        f"Epistemic fraction is\n"
        f"{'HIGHER' if unc_ep_frac > cert_ep_frac else 'LOWER'}\n"
        f"for uncertain prompts"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 164: Uncertainty Decomposition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase164_uncertainty')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Certain: S_total={cert_total:.2f}, epist_frac={cert_ep_frac:.3f}")
    print(f"Uncertain: S_total={unc_total:.2f}, epist_frac={unc_ep_frac:.3f}")
    print(f"{'='*70}")

    save_results('phase164_uncertainty', {
        'experiment': 'Uncertainty Decomposition',
        'summary': {
            'certain_total': float(cert_total),
            'certain_epistemic_frac': float(cert_ep_frac),
            'uncertain_total': float(unc_total),
            'uncertain_epistemic_frac': float(unc_ep_frac),
        }
    })


if __name__ == '__main__':
    main()
