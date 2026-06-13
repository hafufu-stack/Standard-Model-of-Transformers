# -*- coding: utf-8 -*-
"""
Phase 44: Dark Matter Activation
Force-activate dormant FFN neurons and test effect on reasoning.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 44: Dark Matter Activation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Logic puzzles and reasoning tasks
    test_cases = [
        {
            'prompt': "If all roses are flowers, and all flowers need water, then all roses",
            'expected': 'water',
            'category': 'syllogism',
        },
        {
            'prompt': "The sequence 2, 4, 8, 16, 32 follows a pattern. The next number is",
            'expected': '64',
            'category': 'pattern',
        },
        {
            'prompt': "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? The answer is",
            'expected': '5',
            'category': 'logic',
        },
        {
            'prompt': "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. The ball costs",
            'expected': '0.05',
            'category': 'trick',
        },
        {
            'prompt': "If you rearrange the letters 'CIFAIPC', you get the name of a",
            'expected': 'ocean',  # PACIFIC
            'category': 'anagram',
        },
    ]

    # First, profile dormancy across layers
    print("\n--- Profiling FFN dormancy ---")
    profile_prompt = "The theory of everything unifies all fundamental forces of nature into"
    inp = tok(profile_prompt, return_tensors='pt').to(device)

    dormancy_profile = {}
    def make_profile_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            act = h[0, -1, :].detach().cpu().float().numpy()
            total = len(act)
            near_zero = np.sum(np.abs(act) < 0.01)
            dormancy_profile[layer_idx] = {
                'total': total,
                'dormant': int(near_zero),
                'pct': float(near_zero / total * 100),
                'mean_abs': float(np.mean(np.abs(act))),
            }
        return hook

    hooks = []
    for li in range(len(model.model.layers)):
        h = model.model.layers[li].mlp.register_forward_hook(make_profile_hook(li))
        hooks.append(h)
    with torch.no_grad():
        model(**inp)
    for h in hooks:
        h.remove()

    for li in sorted(dormancy_profile.keys()):
        d = dormancy_profile[li]
        if li % 7 == 0:
            print(f"  Layer {li}: {d['pct']:.1f}% dormant ({d['dormant']}/{d['total']})")

    # Test with different activation boost levels
    BOOST_LEVELS = [0.0, 0.5, 1.0, 2.0, 5.0]  # 0 = baseline
    all_results = []

    for tc in test_cases:
        print(f"\n--- {tc['category']}: '{tc['prompt'][:50]}...' ---")
        input_ids = tok(tc['prompt'], return_tensors='pt')['input_ids'].to(device)

        for boost in BOOST_LEVELS:
            hooks = []

            if boost > 0:
                def make_boost_hook(b):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        act = h[0, -1, :]
                        # Find dormant neurons (near-zero activation)
                        dormant_mask = (act.abs() < 0.01)
                        if dormant_mask.any():
                            h_mod = h.clone()
                            # Set dormant neurons to boost value (with random sign)
                            signs = (torch.randint(0, 2, (dormant_mask.sum(),),
                                                   device=h.device).float() * 2 - 1).to(h.dtype)
                            boost_val = (b * signs * act[~dormant_mask].abs().mean()).to(h.dtype)
                            h_mod[0, -1, dormant_mask] = boost_val
                            if isinstance(output, tuple):
                                return (h_mod,) + output[1:]
                            return h_mod
                        return output
                    return hook

                # Apply to middle layers (most dormancy)
                for li in range(len(model.model.layers) // 3, 2 * len(model.model.layers) // 3):
                    h = model.model.layers[li].mlp.register_forward_hook(make_boost_hook(boost))
                    hooks.append(h)

            # Generate 20 tokens
            current_ids = input_ids.clone()
            for t in range(20):
                with torch.no_grad():
                    out = model(current_ids)
                    logits = out.logits[0, -1, :]
                next_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                current_ids = torch.cat([current_ids, next_id], dim=1)

            for h in hooks:
                h.remove()

            generated = tok.decode(current_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            correct = tc['expected'].lower() in generated.lower()

            safe_gen = generated.encode('ascii', errors='replace').decode('ascii')[:40]
            print(f"  boost={boost}: {'OK' if correct else 'MISS'} -> '{safe_gen}...'")

            all_results.append({
                'category': tc['category'],
                'prompt': tc['prompt'][:60],
                'expected': tc['expected'],
                'boost': boost,
                'generated': generated[:100],
                'correct': correct,
            })

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Dormancy profile
    layers = sorted(dormancy_profile.keys())
    dormancy_pcts = [dormancy_profile[l]['pct'] for l in layers]
    axes[0].plot(layers, dormancy_pcts, color='#8e44ad', linewidth=1.5)
    axes[0].fill_between(layers, dormancy_pcts, alpha=0.2, color='#8e44ad')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Dormant Neurons (%)')
    axes[0].set_title('(a) FFN Dormancy Profile')

    # (b) Accuracy vs boost level
    for cat in set(r['category'] for r in all_results):
        cat_results = [r for r in all_results if r['category'] == cat]
        boost_acc = {}
        for r in cat_results:
            b = r['boost']
            if b not in boost_acc:
                boost_acc[b] = []
            boost_acc[b].append(1 if r['correct'] else 0)
        boosts = sorted(boost_acc.keys())
        accs = [np.mean(boost_acc[b]) for b in boosts]
        axes[1].plot(boosts, accs, marker='o', label=cat, linewidth=1.5, markersize=5)
    axes[1].set_xlabel('Boost Level')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('(b) Reasoning Accuracy vs Boost')
    axes[1].legend(fontsize=7)

    # (c) Overall accuracy
    boost_overall = {}
    for r in all_results:
        b = r['boost']
        if b not in boost_overall:
            boost_overall[b] = []
        boost_overall[b].append(1 if r['correct'] else 0)
    boosts = sorted(boost_overall.keys())
    means = [np.mean(boost_overall[b]) * 100 for b in boosts]
    colors = ['#2ecc71' if m >= means[0] else '#e74c3c' for m in means]
    axes[2].bar([str(b) for b in boosts], means, color=colors, alpha=0.8)
    axes[2].set_xlabel('Boost Level')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('(c) Overall Accuracy')
    axes[2].axhline(y=means[0], color='gray', linestyle='--', label=f'Baseline={means[0]:.0f}%')
    axes[2].legend()

    fig.suptitle('Phase 44: Dark Matter Activation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase44_dark_matter')
    plt.close()

    # === Verdict ===
    baseline_acc = np.mean([1 if r['correct'] else 0 for r in all_results if r['boost'] == 0]) * 100
    best_boost = max(boosts, key=lambda b: np.mean(boost_overall[b]))
    best_acc = np.mean(boost_overall[best_boost]) * 100
    change = best_acc - baseline_acc

    print(f"\n{'='*70}")
    print(f"VERDICT: Baseline accuracy={baseline_acc:.0f}%, Best boost={best_boost} "
          f"({best_acc:.0f}%, {change:+.0f}%). "
          f"Dark matter activation {'IMPROVES' if change > 5 else 'does NOT improve'} reasoning.")
    print(f"{'='*70}")

    save_results('phase44_dark_matter', {
        'experiment': 'Dark Matter Activation',
        'dormancy_profile': dormancy_profile,
        'results': all_results,
        'summary': {
            'baseline_accuracy': baseline_acc,
            'best_boost': best_boost,
            'best_accuracy': best_acc,
            'improvement': change,
        }
    })


if __name__ == '__main__':
    main()
