# -*- coding: utf-8 -*-
"""
Phase 89: Multi-Layer DMA v2
Inject coherent program vectors at 3-5 consecutive layers simultaneously
to test if multi-layer DMA can overcome FFN's restorative force.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

TARGET_FACTS = [
    {
        'prompt': "The capital of France is",
        'correct': "Paris",
        'target': "Tokyo",
    },
    {
        'prompt': "Water boils at",
        'correct': "100",
        'target': "50",
    },
    {
        'prompt': "The largest planet is",
        'correct': "Jupiter",
        'target': "Mars",
    },
    {
        'prompt': "The speed of light is",
        'correct': "300",
        'target': "100",
    },
]


def get_target_direction(model, tok, device, target_word):
    """Get embedding direction for target word."""
    ids = tok(target_word, add_special_tokens=False)['input_ids']
    if not ids:
        return None
    emb = model.model.embed_tokens.weight[ids[0]].detach().float()
    return emb / (emb.norm() + 1e-10)


def inject_multi_layer(model, tok, device, prompt, target_word, layer_start, n_inject, alpha=5.0):
    """Inject target direction at multiple consecutive layers."""
    direction = get_target_direction(model, tok, device, target_word)
    if direction is None:
        return None

    hooks = []

    def make_inject_hook(dir_vec, scale):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h_fp32 = h.to(torch.float32)
            dir_device = dir_vec.to(h.device)
            # Project and add target direction
            h_fp32[:, -1, :] = h_fp32[:, -1, :] + scale * dir_device
            result = h_fp32.to(h.dtype)
            if isinstance(output, tuple):
                return (result,) + output[1:]
            return result
        return hook

    n_layers = len(model.model.layers)
    for li in range(layer_start, min(layer_start + n_inject, n_layers)):
        h = model.model.layers[li].register_forward_hook(
            make_inject_hook(direction, alpha)
        )
        hooks.append(h)

    # Forward pass
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)

    for h in hooks:
        h.remove()

    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    # Check success
    target_ids = tok(target_word, add_special_tokens=False)['input_ids']
    correct_ids = tok(prompt.split()[-1], add_special_tokens=False)['input_ids']

    target_prob = probs[target_ids[0]].item() if target_ids else 0
    top_id = torch.argmax(probs).item()
    top_token = tok.decode([top_id])
    hijacked = (top_id == target_ids[0]) if target_ids else False

    return {
        'target_prob': float(target_prob),
        'top_token': top_token,
        'hijacked': bool(hijacked),
    }


def main():
    print("=" * 70)
    print("Phase 89: Multi-Layer DMA v2")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # Test configurations
    inject_configs = [
        {'n_inject': 1, 'label': '1-layer'},
        {'n_inject': 3, 'label': '3-layer'},
        {'n_inject': 5, 'label': '5-layer'},
        {'n_inject': 8, 'label': '8-layer'},
        {'n_inject': 14, 'label': 'half'},
    ]

    layer_starts = [5, 10, 15, 20]
    alpha_values = [1.0, 5.0, 10.0, 20.0]

    results = []

    # Sweep: n_inject layers at fixed position with fixed alpha
    print("\n  Sweep over n_inject layers...")
    for config in inject_configs:
        hijack_rate = 0
        total_target_prob = 0
        for fact in TARGET_FACTS:
            res = inject_multi_layer(model, tok, device, fact['prompt'],
                                     fact['target'], layer_start=10,
                                     n_inject=config['n_inject'], alpha=10.0)
            if res:
                if res['hijacked']:
                    hijack_rate += 1
                total_target_prob += res['target_prob']

        avg_prob = total_target_prob / len(TARGET_FACTS)
        rate = hijack_rate / len(TARGET_FACTS)
        results.append({
            'config': config['label'],
            'n_inject': config['n_inject'],
            'hijack_rate': float(rate),
            'avg_target_prob': float(avg_prob),
        })
        print(f"    {config['label']}: hijack={rate:.0%}, target_prob={avg_prob:.4f}")

    # Sweep: alpha at fixed 5-layer injection
    print("\n  Sweep over alpha...")
    alpha_results = []
    for alpha in alpha_values:
        hijack_rate = 0
        total_prob = 0
        for fact in TARGET_FACTS:
            res = inject_multi_layer(model, tok, device, fact['prompt'],
                                     fact['target'], layer_start=10,
                                     n_inject=5, alpha=alpha)
            if res:
                if res['hijacked']:
                    hijack_rate += 1
                total_prob += res['target_prob']

        avg_prob = total_prob / len(TARGET_FACTS)
        rate = hijack_rate / len(TARGET_FACTS)
        alpha_results.append({
            'alpha': float(alpha),
            'hijack_rate': float(rate),
            'avg_target_prob': float(avg_prob),
        })
        print(f"    alpha={alpha}: hijack={rate:.0%}, target_prob={avg_prob:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) n_inject sweep
    n_injects = [r['n_inject'] for r in results]
    hijack_rates = [r['hijack_rate'] for r in results]
    target_probs = [r['avg_target_prob'] for r in results]
    axes[0].bar(range(len(results)), hijack_rates, color='#c0392b', alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels([r['config'] for r in results], fontsize=9)
    axes[0].set_ylabel('Hijack Rate')
    axes[0].set_title('(a) DMA Success vs Injection Depth')
    axes[0].set_ylim(0, 1.1)

    # (b) Alpha sweep
    alphas = [r['alpha'] for r in alpha_results]
    a_rates = [r['hijack_rate'] for r in alpha_results]
    a_probs = [r['avg_target_prob'] for r in alpha_results]
    axes[1].plot(alphas, a_rates, 'o-', color='#c0392b', linewidth=2, label='Hijack rate')
    ax1b = axes[1].twinx()
    ax1b.plot(alphas, a_probs, 's-', color='#2980b9', linewidth=2, label='Target prob')
    axes[1].set_xlabel('Injection Strength $\\alpha$')
    axes[1].set_ylabel('Hijack Rate', color='#c0392b')
    ax1b.set_ylabel('Target Prob', color='#2980b9')
    axes[1].set_title('(b) Strength Sweep (5 layers)')

    # (c) Comparison: 1-layer vs multi
    single = results[0] if results else {'hijack_rate': 0}
    multi = max(results[1:], key=lambda r: r['hijack_rate']) if len(results) > 1 else single
    labels = ['Single Layer\n(Phase 10)', f'Multi Layer\n({multi["config"]})']
    rates = [single['hijack_rate'], multi['hijack_rate']]
    colors = ['#7f8c8d', '#27ae60' if multi['hijack_rate'] > single['hijack_rate'] else '#c0392b']
    axes[2].bar(range(2), rates, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_xticks(range(2))
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Hijack Rate')
    axes[2].set_title('(c) Single vs Multi DMA')
    axes[2].set_ylim(0, 1.1)
    improvement = multi['hijack_rate'] / (single['hijack_rate'] + 1e-10)
    axes[2].text(1, multi['hijack_rate'] + 0.05,
                 f'{improvement:.1f}x', ha='center', fontsize=14, fontweight='bold')

    fig.suptitle(f'Phase 89: Multi-Layer DMA v2 '
                 f'(Best: {multi["hijack_rate"]:.0%} with {multi["config"]})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase89_multi_layer_dma')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Single-layer DMA: {single['hijack_rate']:.0%}")
    print(f"Best multi-layer: {multi['config']} at {multi['hijack_rate']:.0%}")
    print(f"Improvement: {improvement:.1f}x")
    print(f"{'='*70}")

    save_results('phase89_multi_layer_dma', {
        'experiment': 'Multi-Layer DMA v2',
        'n_inject_sweep': results,
        'alpha_sweep': alpha_results,
        'summary': {
            'single_rate': float(single['hijack_rate']),
            'best_multi_rate': float(multi['hijack_rate']),
            'improvement': float(improvement),
        }
    })


if __name__ == '__main__':
    main()
