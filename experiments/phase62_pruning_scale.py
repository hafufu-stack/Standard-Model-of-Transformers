# -*- coding: utf-8 -*-
"""
Phase 62: Dark Energy Pruning at Scale
Apply beta_c=0.57 dynamic FFN pruning on reasoning tasks.
Measure accuracy vs FLOPs tradeoff.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


MATH_PROBLEMS = [
    {"q": "If x + 5 = 12, what is x?", "a": "7"},
    {"q": "What is 15 * 3?", "a": "45"},
    {"q": "If a rectangle has length 8 and width 3, what is its area?", "a": "24"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "If you have 3 dozen eggs, how many eggs do you have?", "a": "36"},
    {"q": "What is 100 divided by 4?", "a": "25"},
    {"q": "If a triangle has sides 3, 4, and 5, is it a right triangle?", "a": "yes"},
    {"q": "What is 2 to the power of 8?", "a": "256"},
    {"q": "How many minutes are in 3 hours?", "a": "180"},
    {"q": "What is 17 + 28?", "a": "45"},
    {"q": "If a car travels 60 mph for 2.5 hours, how far does it go?", "a": "150"},
    {"q": "What is the next prime number after 7?", "a": "11"},
]


def main():
    print("=" * 70)
    print("Phase 62: Dark Energy Pruning at Scale")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    BETA_VALUES = [0.0, 0.3, 0.57, 0.7, 0.85]
    GEN_LENGTH = 30

    all_results = []

    for beta in BETA_VALUES:
        hooks = []
        pruned_count = [0]
        total_count = [0]

        if beta > 0:
            def make_prune_hook(b):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    mask = (h.abs() > b).to(h.dtype)
                    pruned_count[0] += int((mask == 0).sum().item())
                    total_count[0] += int(mask.numel())
                    h_pruned = h * mask
                    if isinstance(output, tuple):
                        return (h_pruned,) + output[1:]
                    return h_pruned
                return hook

            for li in range(n_layers):
                hk = model.model.layers[li].mlp.register_forward_hook(make_prune_hook(beta))
                hooks.append(hk)

        correct = 0
        total = 0

        for prob in MATH_PROBLEMS:
            prompt = f"Q: {prob['q']}\nA: The answer is"
            input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)

            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=GEN_LENGTH,
                    do_sample=False, temperature=1.0,
                )

            gen_text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            gen_clean = gen_text.strip().lower().split('.')[0].split(',')[0].strip()

            is_correct = prob['a'].lower() in gen_clean
            correct += int(is_correct)
            total += 1

        for h in hooks:
            h.remove()

        prune_pct = pruned_count[0] / (total_count[0] + 1e-10) * 100 if beta > 0 else 0
        accuracy = correct / total * 100

        print(f"  beta={beta:.2f}: accuracy={accuracy:.0f}% ({correct}/{total}), "
              f"pruned={prune_pct:.1f}%")

        all_results.append({
            'beta': float(beta),
            'accuracy': float(accuracy),
            'correct': correct,
            'total': total,
            'prune_pct': float(prune_pct),
            'flops_saved': float(prune_pct),
        })

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    betas = [r['beta'] for r in all_results]
    accs = [r['accuracy'] for r in all_results]
    prunes = [r['prune_pct'] for r in all_results]

    # (a) Accuracy vs beta
    axes[0].plot(betas, accs, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[0].axvline(x=0.57, color='blue', linestyle='--', alpha=0.5, label='beta_c=0.57')
    axes[0].set_xlabel('Pruning Threshold (beta)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('(a) Accuracy vs Pruning')
    axes[0].legend()
    for b, a in zip(betas, accs):
        axes[0].text(b, a + 1, f'{a:.0f}%', ha='center', fontsize=9)

    # (b) Pruning % vs beta
    axes[1].plot(betas, prunes, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[1].axvline(x=0.57, color='blue', linestyle='--', alpha=0.5, label='beta_c=0.57')
    axes[1].set_xlabel('Pruning Threshold (beta)')
    axes[1].set_ylabel('Neurons Pruned (%)')
    axes[1].set_title('(b) Pruning Rate')
    axes[1].legend()
    for b, p in zip(betas, prunes):
        axes[1].text(b, p + 1, f'{p:.0f}%', ha='center', fontsize=9)

    # (c) Pareto frontier (accuracy vs FLOPs saved)
    axes[2].scatter(prunes, accs, s=100, c=[plt.cm.viridis(b/max(betas+[1]))
                    for b in betas], edgecolors='black', linewidths=1.5, zorder=5)
    for i, b in enumerate(betas):
        axes[2].annotate(f'b={b:.2f}', (prunes[i], accs[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel('FLOPs Saved (% pruned)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('(c) Pareto Frontier')

    # Find beta_c result
    bc_result = next((r for r in all_results if abs(r['beta'] - 0.57) < 0.01), None)
    baseline = next((r for r in all_results if r['beta'] == 0), None)

    fig.suptitle(f'Phase 62: Dark Energy Pruning '
                 f'(beta_c=0.57: {bc_result["accuracy"]:.0f}% acc, '
                 f'{bc_result["prune_pct"]:.0f}% pruned)' if bc_result else
                 'Phase 62: Dark Energy Pruning',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase62_pruning_scale')
    plt.close()

    if bc_result and baseline:
        acc_drop = baseline['accuracy'] - bc_result['accuracy']
        print(f"\n{'='*70}")
        print(f"VERDICT: At beta_c=0.57: {bc_result['accuracy']:.0f}% accuracy "
              f"(vs {baseline['accuracy']:.0f}% baseline, "
              f"drop={acc_drop:.0f}pp), "
              f"{bc_result['prune_pct']:.0f}% neurons pruned. "
              f"{'VIABLE' if acc_drop <= 10 else 'TOO AGGRESSIVE'} for deployment.")
        print(f"{'='*70}")

    save_results('phase62_pruning_scale', {
        'experiment': 'Dark Energy Pruning at Scale',
        'results': all_results,
        'summary': {
            'baseline_acc': float(baseline['accuracy']) if baseline else 0,
            'bc_acc': float(bc_result['accuracy']) if bc_result else 0,
            'bc_prune_pct': float(bc_result['prune_pct']) if bc_result else 0,
        }
    })


if __name__ == '__main__':
    main()
