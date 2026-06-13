# -*- coding: utf-8 -*-
"""
Phase 149: Macroscopic Quantum Tunneling
Can we force the model through an energy barrier to reach a normally
inaccessible correct answer? Inject a directional perturbation at
the variance peak (L21) to tunnel through the barrier.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

# Trick questions where models typically fail
TRICK_QS = [
    {
        'prompt': "How many r's are in the word 'strawberry'? The answer is",
        'correct_tokens': ['3', ' 3'],
        'desc': 'strawberry r-count',
    },
    {
        'prompt': "Which weighs more: a pound of feathers or a pound of steel? The answer is",
        'correct_tokens': ['they', 'They', ' they', ' the same', ' neither'],
        'desc': 'pound trick',
    },
    {
        'prompt': "If you have 3 apples and you take away 2, how many apples do YOU have? You have",
        'correct_tokens': ['2', ' 2', ' two'],
        'desc': 'apple take',
    },
    {
        'prompt': "A man has 5 daughters and each daughter has a brother. How many children does the man have? He has",
        'correct_tokens': ['6', ' 6', ' six'],
        'desc': 'siblings',
    },
]


def get_correct_direction(model, tok, prompt, correct_tokens, device, target_layer):
    """Get the direction in hidden space that points toward the correct answer."""
    inp = tok(prompt, return_tensors='pt').to(device)

    # Get baseline hidden state at target layer
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h_base = out.hidden_states[target_layer][0, -1, :].float().clone()

    # Find correct token embedding direction
    correct_ids = []
    for t in correct_tokens:
        ids = tok.encode(t, add_special_tokens=False)
        correct_ids.extend(ids)

    if not correct_ids:
        return torch.zeros_like(h_base)

    # Use lm_head weight to find direction that increases correct token probability
    # direction = sum of lm_head.weight[correct_id] for each correct_id
    direction = torch.zeros_like(h_base)
    for cid in correct_ids[:3]:  # Use top 3
        if cid < model.lm_head.weight.shape[0]:
            direction += model.lm_head.weight[cid].float()

    direction = direction / (direction.norm() + 1e-10)
    return direction


def main():
    print("=" * 70)
    print("Phase 149: Macroscopic Quantum Tunneling")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Tunneling configurations
    target_layers = [20, 21, 22]  # Around L0
    scales = [0, 5, 10, 20, 50, 100]

    results = {}
    all_accs = {s: 0 for s in scales}

    for qi, q in enumerate(TRICK_QS):
        prompt = q['prompt']
        correct = q['correct_tokens']
        desc = q['desc']
        print(f"\n  Q{qi}: {desc}")

        # Get correct direction at each target layer
        directions = {}
        for tl in target_layers:
            if tl < len(model.model.layers):
                directions[tl] = get_correct_direction(model, tok, prompt, correct, device, tl)

        q_results = {}
        for scale in scales:
            if scale == 0:
                # Baseline
                inp = tok(prompt, return_tensors='pt').to(device)
                with torch.no_grad():
                    out = model(**inp)
                logits = out.logits[0, -1, :].float()
            else:
                # Inject perturbation
                inp = tok(prompt, return_tensors='pt').to(device)
                hooks = []
                for tl in target_layers:
                    if tl in directions and tl < len(model.model.layers):
                        def make_hook(direction, s):
                            def hook_fn(module, input, output):
                                if isinstance(output, tuple):
                                    h = output[0].clone()
                                    d = direction.to(h.device).to(h.dtype)
                                    h[:, -1, :] += s * d
                                    return (h,) + output[1:]
                                return output
                            return hook_fn
                        hooks.append(model.model.layers[tl].register_forward_hook(
                            make_hook(directions[tl], scale)))

                with torch.no_grad():
                    out = model(**inp)
                for h in hooks:
                    h.remove()
                logits = out.logits[0, -1, :].float()

            probs = torch.softmax(logits, dim=-1)
            top5 = torch.topk(probs, 5)
            top5_tokens = [tok.decode([t]).strip() for t in top5.indices]
            top5_probs = top5.values.tolist()

            # Check correctness
            is_correct = any(ct.strip() in top5_tokens[0] for ct in correct)
            if is_correct:
                all_accs[scale] += 1

            q_results[scale] = {
                'top1': top5_tokens[0],
                'correct': is_correct,
                'prob': top5_probs[0],
                'top5': top5_tokens,
            }
            marker = "OK" if is_correct else "XX"
            print(f"    scale={scale}: {top5_tokens[0]} (p={top5_probs[0]:.3f}) [{marker}]")

        results[desc] = q_results

    n_total = len(TRICK_QS)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Accuracy vs scale
    accs_list = [all_accs[s] / n_total for s in scales]
    colors_a = ['#2980b9'] + ['#27ae60' if a > accs_list[0] else '#c0392b' for a in accs_list[1:]]
    axes[0,0].bar(range(len(scales)), accs_list, color=colors_a, alpha=0.8, edgecolor='black')
    axes[0,0].set_xticks(range(len(scales)))
    axes[0,0].set_xticklabels([str(s) for s in scales])
    axes[0,0].set_xlabel('Perturbation Scale')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('(a) Tunneling Accuracy')

    # (b) Confidence vs scale
    conf_by_scale = {s: [] for s in scales}
    for desc, q_res in results.items():
        for s, r in q_res.items():
            conf_by_scale[s].append(r['prob'])
    mean_conf = [np.mean(conf_by_scale[s]) for s in scales]
    axes[0,1].plot(scales, mean_conf, 'o-', color='#8e44ad', markersize=6, linewidth=2)
    axes[0,1].set_xlabel('Perturbation Scale')
    axes[0,1].set_ylabel('Mean Confidence')
    axes[0,1].set_title('(b) Confidence vs Scale')

    # (c) Per-question results
    q_names = list(results.keys())
    matrix = np.zeros((len(q_names), len(scales)))
    for qi, qn in enumerate(q_names):
        for si, s in enumerate(scales):
            matrix[qi, si] = 1 if results[qn][s]['correct'] else 0
    axes[0,2].imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0,2].set_xticks(range(len(scales)))
    axes[0,2].set_xticklabels([str(s) for s in scales])
    axes[0,2].set_yticks(range(len(q_names)))
    axes[0,2].set_yticklabels(q_names, fontsize=8)
    axes[0,2].set_xlabel('Scale')
    axes[0,2].set_title('(c) Tunneling Success Matrix')

    # (d) Baseline vs best scale per question
    baseline_probs = []
    best_probs = []
    for qn in q_names:
        bp = results[qn][0]['prob']
        best_p = max(results[qn][s]['prob'] for s in scales if s > 0)
        baseline_probs.append(bp)
        best_probs.append(best_p)
    x = np.arange(len(q_names))
    axes[1,0].bar(x - 0.15, baseline_probs, 0.3, color='#c0392b', label='Baseline')
    axes[1,0].bar(x + 0.15, best_probs, 0.3, color='#27ae60', label='Best tunnel')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(q_names, fontsize=7, rotation=15)
    axes[1,0].set_ylabel('Top-1 Probability')
    axes[1,0].set_title('(d) Baseline vs Tunneled')
    axes[1,0].legend(fontsize=8)

    # (e) Optimal scale analysis
    axes[1,1].plot(scales, accs_list, 'o-', color='#c0392b', markersize=6, linewidth=2,
                   label='Accuracy')
    ax_twin = axes[1,1].twinx()
    ax_twin.plot(scales, mean_conf, 's-', color='#2980b9', markersize=6, linewidth=2,
                 label='Confidence')
    axes[1,1].set_xlabel('Scale')
    axes[1,1].set_ylabel('Accuracy', color='#c0392b')
    ax_twin.set_ylabel('Confidence', color='#2980b9')
    axes[1,1].set_title('(e) Accuracy-Confidence Tradeoff')

    # (f) Summary
    best_scale = scales[np.argmax(accs_list)]
    best_acc = max(accs_list)
    summary = (
        f"Quantum Tunneling\n\n"
        f"Baseline accuracy: {accs_list[0]:.0%}\n"
        f"Best accuracy: {best_acc:.0%} (scale={best_scale})\n\n"
        f"Tunneling {'WORKS' if best_acc > accs_list[0] else 'does NOT improve'}\n\n"
        + "\n".join(f"  {qn}: {'tunneled!' if any(results[qn][s]['correct'] and s > 0 for s in scales) else 'stuck'}"
                    for qn in q_names)
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 149: Macroscopic Quantum Tunneling',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase149_tunneling')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Baseline: {accs_list[0]:.0%}, Best: {best_acc:.0%} (scale={best_scale})")
    print(f"{'='*70}")

    save_results('phase149_tunneling', {
        'experiment': 'Quantum Tunneling',
        'accuracy_by_scale': {str(s): all_accs[s]/n_total for s in scales},
        'summary': {
            'baseline_acc': float(accs_list[0]),
            'best_acc': float(best_acc),
            'best_scale': int(best_scale),
        }
    })


if __name__ == '__main__':
    main()
