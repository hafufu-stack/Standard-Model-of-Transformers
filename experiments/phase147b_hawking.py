# -*- coding: utf-8 -*-
"""
Phase 147b: Hawking Radiation Rescue (Simplified)
When a model falls into a repetition loop (T->0, black hole),
can we rescue it by injecting directional noise at the right moment?
Simulated by feeding highly repetitive input and measuring recovery.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 147b: Hawking Radiation Rescue")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Create repetitive prompts that simulate "black hole collapse"
    # More repetition = more collapsed context
    base = "The answer is yes. "
    repetitions = [1, 2, 4, 8, 16, 32]
    suffix = "The most interesting scientific discovery is"

    # Measure: entropy, confidence, and diversity of output after repetition
    # Then inject noise at critical layers and see if we can "rescue" diversity

    results_no_noise = []
    results_with_noise = []

    for n_rep in repetitions:
        prompt = base * n_rep + suffix
        inp = tok(prompt, return_tensors='pt').to(device)

        # Baseline (no noise)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        S = -(probs * torch.log(probs + 1e-10)).sum().item()
        conf = probs.max().item()
        top5 = torch.topk(probs, 5)
        top5_tokens = [tok.decode([t]).strip() for t in top5.indices]

        # Diversity: effective number of choices (exp(S))
        diversity = np.exp(S) if not np.isnan(S) else 0

        results_no_noise.append({
            'n_rep': n_rep,
            'S': S if not np.isnan(S) else 0,
            'conf': conf,
            'diversity': diversity,
            'top5': top5_tokens,
        })

        # With noise injection at L19-L23 (critical layers)
        hooks = []
        for li in range(19, min(24, len(model.model.layers))):
            def make_hook(scale=2.0):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        noise = torch.randn_like(h) * scale
                        return (h + noise,) + output[1:]
                    return output
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_hook()))

        with torch.no_grad():
            out_n = model(**inp, output_hidden_states=True)
        for h in hooks:
            h.remove()

        logits_n = out_n.logits[0, -1, :].float()
        probs_n = torch.softmax(logits_n, dim=-1)
        S_n = -(probs_n * torch.log(probs_n + 1e-10)).sum().item()
        conf_n = probs_n.max().item()
        diversity_n = np.exp(S_n) if not np.isnan(S_n) else 0

        results_with_noise.append({
            'n_rep': n_rep,
            'S': S_n if not np.isnan(S_n) else 0,
            'conf': conf_n,
            'diversity': diversity_n,
        })

        print(f"  rep={n_rep:3d}: S={S:.2f}->{S_n:.2f}, "
              f"div={diversity:.0f}->{diversity_n:.0f}, top={top5_tokens[0]}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    reps = [r['n_rep'] for r in results_no_noise]

    # (a) Entropy vs repetition
    S_base = [r['S'] for r in results_no_noise]
    S_noise = [r['S'] for r in results_with_noise]
    axes[0,0].plot(reps, S_base, 'o-', color='#c0392b', markersize=6, linewidth=2, label='No rescue')
    axes[0,0].plot(reps, S_noise, 's-', color='#27ae60', markersize=6, linewidth=2, label='With rescue')
    axes[0,0].set_xlabel('Repetitions')
    axes[0,0].set_ylabel('$S$')
    axes[0,0].set_title('(a) Entropy vs Repetition')
    axes[0,0].legend()
    axes[0,0].set_xscale('log')

    # (b) Diversity (effective vocabulary size)
    div_base = [r['diversity'] for r in results_no_noise]
    div_noise = [r['diversity'] for r in results_with_noise]
    axes[0,1].plot(reps, div_base, 'o-', color='#c0392b', markersize=6, linewidth=2, label='No rescue')
    axes[0,1].plot(reps, div_noise, 's-', color='#27ae60', markersize=6, linewidth=2, label='With rescue')
    axes[0,1].set_xlabel('Repetitions')
    axes[0,1].set_ylabel('Diversity (exp $S$)')
    axes[0,1].set_title('(b) Output Diversity')
    axes[0,1].legend()
    axes[0,1].set_xscale('log')

    # (c) Confidence
    conf_base = [r['conf'] for r in results_no_noise]
    conf_noise = [r['conf'] for r in results_with_noise]
    axes[0,2].plot(reps, conf_base, 'o-', color='#c0392b', markersize=6, linewidth=2, label='No rescue')
    axes[0,2].plot(reps, conf_noise, 's-', color='#27ae60', markersize=6, linewidth=2, label='With rescue')
    axes[0,2].set_xlabel('Repetitions')
    axes[0,2].set_ylabel('Top-1 Confidence')
    axes[0,2].set_title('(c) Confidence vs Repetition')
    axes[0,2].legend()
    axes[0,2].set_xscale('log')

    # (d) Rescue effectiveness
    rescue_eff = [(n - b) / (b + 1e-10) * 100
                  for b, n in zip(div_base, div_noise)]
    re_colors = ['#27ae60' if r > 0 else '#c0392b' for r in rescue_eff]
    axes[1,0].bar(range(len(reps)), rescue_eff, color=re_colors, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(reps)))
    axes[1,0].set_xticklabels([str(r) for r in reps])
    axes[1,0].set_xlabel('Repetitions')
    axes[1,0].set_ylabel('Diversity Change (%)')
    axes[1,0].set_title('(d) Rescue Effectiveness')
    axes[1,0].axhline(y=0, color='black', linewidth=1)

    # (e) Black hole indicator: S approaching 0
    axes[1,1].plot(reps, S_base, 'o-', color='#c0392b', markersize=6, linewidth=2)
    axes[1,1].axhline(y=0, color='black', linewidth=1, linestyle='--', label='T=0 (Black Hole)')
    # Shade danger zone
    axes[1,1].fill_between(reps, 0, [max(S_base)*0.1]*len(reps),
                           alpha=0.2, color='black', label='Danger Zone')
    axes[1,1].set_xlabel('Repetitions')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) Black Hole Approach')
    axes[1,1].set_xscale('log')
    axes[1,1].legend()

    # (f) Summary
    collapsed = any(s < 1.0 for s in S_base)
    rescued = any(s_n > s_b * 1.2 for s_b, s_n in zip(S_base, S_noise))
    summary = (
        f"Hawking Radiation Rescue\n\n"
        f"Repetitions tested: {reps}\n\n"
        f"S range (no noise): [{min(S_base):.2f}, {max(S_base):.2f}]\n"
        f"S range (with noise): [{min(S_noise):.2f}, {max(S_noise):.2f}]\n\n"
        f"Black hole collapse: {'YES' if collapsed else 'NO'}\n"
        f"Rescue effective: {'YES' if rescued else 'NO'}\n\n"
        f"Max rescue: {max(rescue_eff):.0f}% diversity gain"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 147b: Hawking Radiation Rescue',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase147b_hawking')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Collapsed: {'YES' if collapsed else 'NO'}")
    print(f"Rescue: {'YES' if rescued else 'NO'}")
    print(f"{'='*70}")

    save_results('phase147b_hawking', {
        'experiment': 'Hawking Radiation Rescue',
        'results_no_noise': results_no_noise,
        'results_with_noise': results_with_noise,
        'summary': {
            'collapsed': collapsed,
            'rescued': rescued,
        }
    })


if __name__ == '__main__':
    main()
