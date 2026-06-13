# -*- coding: utf-8 -*-
"""
Phase 143: Thermodynamic Test-Time Compute
Inject noise at the critical point to PREVENT the phase transition,
forcing the model to stay in the exploration (distributed) phase longer.
This should improve reasoning on hard problems by extending "thinking time".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

# Hard reasoning problems where the model might get wrong
HARD_PROMPTS = [
    ("If a bat and a ball cost $1.10 and the bat costs $1.00 more than the ball, how much does the ball cost? The ball costs $", "0.05"),
    ("A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? The answer is", "9"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? The answer is", "5"),
    ("There are 3 apples. You take away 2. How many do you have? You have", "2"),
    ("What is the next number in the sequence: 2, 6, 12, 20, 30, ...? The next number is", "42"),
]


def run_with_noise(model, tok, prompt, device, noise_layers, noise_scale, n_layers):
    """Run inference with noise injection at specified layers."""
    inp = tok(prompt, return_tensors='pt').to(device)

    hooks = []
    for li in noise_layers:
        if li < len(model.model.layers):
            def make_hook(scale):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        noise = torch.randn_like(h) * scale
                        return (h + noise,) + output[1:]
                    return output
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_hook(noise_scale)))

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in hooks:
        h.remove()

    # Get prediction
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(probs, 5)
    tokens = [tok.decode([t]) for t in top5.indices]
    probs_list = top5.values.tolist()

    # Get entropy profile
    S_vals = []
    for li in range(min(n_layers, len(out.hidden_states))):
        hs = out.hidden_states[li]
        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            log = model.lm_head(normed).squeeze().float()
        p = torch.softmax(log, dim=-1)
        S = -(p * torch.log(p + 1e-10)).sum().item()
        S_vals.append(S if not np.isnan(S) else 0)

    return {
        'top5_tokens': tokens,
        'top5_probs': probs_list,
        'entropy_profile': S_vals,
        'final_entropy': S_vals[-1] if S_vals else 0,
    }


def main():
    print("=" * 70)
    print("Phase 143: Thermodynamic Test-Time Compute")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Noise configurations
    # Critical layers = around L0 = 20-23
    critical_layers = list(range(19, 24))
    # Exploration layers = L5-L17 (cooling valley)
    exploration_layers = list(range(5, 18))

    noise_configs = [
        ("baseline", [], 0),
        ("critical_0.1", critical_layers, 0.1),
        ("critical_0.5", critical_layers, 0.5),
        ("critical_1.0", critical_layers, 1.0),
        ("explore_0.5", exploration_layers, 0.5),
    ]

    results = {}
    correct_counts = {name: 0 for name, _, _ in noise_configs}
    confidence_sums = {name: 0 for name, _, _ in noise_configs}

    for prompt, expected in HARD_PROMPTS:
        print(f"\n  Prompt: ...{prompt[-40:]}")
        print(f"  Expected: {expected}")
        results[prompt[:30]] = {}

        for config_name, layers, scale in noise_configs:
            r = run_with_noise(model, tok, prompt, device, layers, scale, n_layers)
            top1 = r['top5_tokens'][0].strip()
            correct = expected.strip() in top1
            if correct:
                correct_counts[config_name] += 1
            confidence_sums[config_name] += r['top5_probs'][0]

            results[prompt[:30]][config_name] = {
                'top1': top1,
                'correct': correct,
                'prob': r['top5_probs'][0],
                'final_S': r['final_entropy'],
            }
            marker = "OK" if correct else "XX"
            print(f"    {config_name}: {top1} (p={r['top5_probs'][0]:.3f}) [{marker}]")

    n_total = len(HARD_PROMPTS)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Accuracy by config
    names = [n for n, _, _ in noise_configs]
    accs = [correct_counts[n] / n_total for n in names]
    colors_a = ['#2980b9' if n == 'baseline' else '#27ae60' if a > accs[0] else '#c0392b'
                for n, a in zip(names, accs)]
    axes[0,0].bar(range(len(names)), accs, color=colors_a, alpha=0.8, edgecolor='black')
    axes[0,0].set_xticks(range(len(names)))
    axes[0,0].set_xticklabels(names, fontsize=8, rotation=20)
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('(a) Accuracy by Noise Config')

    # (b) Confidence by config
    confs = [confidence_sums[n] / n_total for n in names]
    axes[0,1].bar(range(len(names)), confs, color=colors_a, alpha=0.8, edgecolor='black')
    axes[0,1].set_xticks(range(len(names)))
    axes[0,1].set_xticklabels(names, fontsize=8, rotation=20)
    axes[0,1].set_ylabel('Mean Confidence')
    axes[0,1].set_title('(b) Confidence by Config')

    # (c) Example: first prompt entropy profiles
    first_prompt = list(results.keys())[0]
    # Re-run to get entropy profiles
    prompt_0 = HARD_PROMPTS[0][0]
    for config_name, layers, scale in noise_configs:
        r = run_with_noise(model, tok, prompt_0, device, layers, scale, n_layers)
        axes[0,2].plot(range(len(r['entropy_profile'])), r['entropy_profile'],
                      'o-', markersize=3, label=config_name)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$S$')
    axes[0,2].set_title('(c) Entropy Profiles (Prompt 1)')
    axes[0,2].legend(fontsize=7)

    # (d) Accuracy improvement
    baseline_acc = accs[0]
    improvements = [(a - baseline_acc) * 100 for a in accs]
    imp_colors = ['#27ae60' if i > 0 else '#c0392b' if i < 0 else '#7f8c8d' for i in improvements]
    axes[1,0].bar(range(len(names)), improvements, color=imp_colors, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(names)))
    axes[1,0].set_xticklabels(names, fontsize=8, rotation=20)
    axes[1,0].axhline(y=0, color='black', linewidth=1)
    axes[1,0].set_ylabel('Acc Improvement (%)')
    axes[1,0].set_title('(d) Improvement vs Baseline')

    # (e) Per-problem results matrix
    problem_names = [p[:20] + "..." for p, _ in HARD_PROMPTS]
    matrix = np.zeros((len(HARD_PROMPTS), len(names)))
    for pi, (prompt, _) in enumerate(HARD_PROMPTS):
        key = prompt[:30]
        for ci, n in enumerate(names):
            if key in results and n in results[key]:
                matrix[pi, ci] = 1 if results[key][n]['correct'] else 0

    axes[1,1].imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[1,1].set_xticks(range(len(names)))
    axes[1,1].set_xticklabels(names, fontsize=7, rotation=20)
    axes[1,1].set_yticks(range(len(problem_names)))
    axes[1,1].set_yticklabels(problem_names, fontsize=7)
    axes[1,1].set_title('(e) Correctness Matrix (green=correct)')

    # (f) Summary
    best_config = names[np.argmax(accs)]
    summary = (
        f"Thermodynamic Test-Time Compute\n\n"
        + "\n".join(f"{n}: {correct_counts[n]}/{n_total} ({a:.0%})"
                    for n, a in zip(names, accs))
        + f"\n\nBest: {best_config} ({max(accs):.0%})\n"
        f"Baseline: {baseline_acc:.0%}\n\n"
        f"Noise at critical point\n"
        f"{'IMPROVES' if max(accs) > baseline_acc else 'does NOT improve'}\n"
        f"reasoning accuracy"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 143: Thermodynamic Test-Time Compute',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase143_ttc')
    plt.close()

    print(f"\n{'='*70}")
    for n, a in zip(names, accs):
        print(f"  {n}: {a:.0%}")
    print(f"Best: {best_config}")
    print(f"{'='*70}")

    save_results('phase143_ttc', {
        'experiment': 'Thermodynamic Test-Time Compute',
        'accuracy': {n: correct_counts[n]/n_total for n in names},
        'confidence': {n: confidence_sums[n]/n_total for n in names},
        'summary': {
            'best_config': best_config,
            'best_accuracy': float(max(accs)),
            'baseline_accuracy': float(baseline_acc),
        }
    })


if __name__ == '__main__':
    main()
