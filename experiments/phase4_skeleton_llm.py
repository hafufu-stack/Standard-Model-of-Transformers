# -*- coding: utf-8 -*-
"""
Phase 4: Skeleton LLM (GlassBox Sub-graphing)
==============================================
Hypothesis: 99% of LLM parameters maintain grammar/fluency.
The actual "reasoning engine" is a tiny sub-graph at register layers.

Test: bypass all layers except register layers (L0,L2,L11,L17)
with Identity, then test arithmetic/logic accuracy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

def main():
    print("=" * 70)
    print("Phase 4: Skeleton LLM (GlassBox Sub-graphing)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # ================================================================
    # Test prompts (arithmetic + reasoning + factual)
    # ================================================================
    test_cases = [
        {"prompt": "2 + 3 =", "answer": "5", "type": "arithmetic"},
        {"prompt": "7 * 8 =", "answer": "56", "type": "arithmetic"},
        {"prompt": "15 - 9 =", "answer": "6", "type": "arithmetic"},
        {"prompt": "The capital of France is", "answer": "Paris", "type": "factual"},
        {"prompt": "Water freezes at", "answer": "0", "type": "factual"},
        {"prompt": "The opposite of up is", "answer": "down", "type": "semantic"},
        {"prompt": "1, 2, 3, 4, 5,", "answer": "6", "type": "pattern"},
        {"prompt": "If A=1 and B=2, then A+B=", "answer": "3", "type": "logic"},
    ]

    # ================================================================
    # Skeleton configurations
    # ================================================================
    register_layers = [0, 2, 11, 17]
    configs = {
        'full_model': list(range(n_layers)),          # All layers active
        'registers_only': register_layers,             # L0,L2,L11,L17
        'first_last': [0, n_layers - 1],              # L0, L27
        'every_4th': list(range(0, n_layers, 4)),     # L0,L4,L8,...
        'top_half': list(range(n_layers // 2, n_layers)),  # L14-L27
        'bottom_half': list(range(0, n_layers // 2)),      # L0-L13
        'registers_plus_output': register_layers + [22, 23, 24, 25, 26, 27],
    }

    def evaluate_skeleton(model, tok, active_layers, test_cases, device):
        """Bypass all layers NOT in active_layers with Identity hook."""
        bypassed = [l for l in range(n_layers) if l not in active_layers]
        handles = []

        for li in bypassed:
            def make_identity_hook():
                def hook(module, input, output):
                    # Return input unchanged (bypass layer computation)
                    if isinstance(output, tuple):
                        if isinstance(input, tuple) and len(input) > 0:
                            return (input[0],) + output[1:]
                        return output
                    return input[0] if isinstance(input, tuple) else input
                return hook
            h = model.model.layers[li].register_forward_hook(make_identity_hook())
            handles.append(h)

        results = []
        for tc in test_cases:
            inp = tok(tc['prompt'], return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model(**inp).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

            # Check if correct answer is in top-k
            topk = torch.topk(probs, 10)
            top_tokens = [tok.decode([idx]).strip() for idx in topk.indices.tolist()]
            top_probs = topk.values.tolist()

            correct_ids = tok.encode(tc['answer'])
            correct_prob = 0.0
            for cid in correct_ids:
                correct_prob = max(correct_prob, probs[cid].item())

            is_top1 = tc['answer'].strip() in top_tokens[0]
            is_top5 = any(tc['answer'].strip() in t for t in top_tokens[:5])

            results.append({
                'prompt': tc['prompt'],
                'answer': tc['answer'],
                'type': tc['type'],
                'correct_prob': correct_prob,
                'top1': top_tokens[0],
                'is_top1': is_top1,
                'is_top5': is_top5,
            })

        for h in handles:
            h.remove()

        accuracy = np.mean([r['is_top5'] for r in results])
        avg_prob = np.mean([r['correct_prob'] for r in results])
        return accuracy, avg_prob, results

    # ================================================================
    # Run all configurations
    # ================================================================
    config_results = {}
    for config_name, active_layers in configs.items():
        n_active = len(active_layers)
        n_params_pct = 100.0 * n_active / n_layers
        print(f"\n--- {config_name}: {n_active}/{n_layers} layers ({n_params_pct:.1f}%) ---")

        acc, avg_p, details = evaluate_skeleton(model, tok, active_layers, test_cases, device)
        config_results[config_name] = {
            'accuracy': acc,
            'avg_prob': avg_p,
            'n_active': n_active,
            'pct_params': n_params_pct,
            'details': details,
        }
        print(f"  Top-5 Accuracy: {acc:.2%}, Avg P(correct): {avg_p:.4f}")
        for d in details:
            status = "OK" if d['is_top5'] else "MISS"
            top1_safe = d['top1'].encode('ascii', 'replace').decode('ascii')
            print(f"    [{status}] {d['prompt']} -> {top1_safe} (P={d['correct_prob']:.4f})")

    # ================================================================
    # Layer ablation sweep: remove one layer at a time
    # ================================================================
    print("\n--- Layer-by-layer ablation ---")
    ablation_results = []
    for remove_layer in range(n_layers):
        active = [l for l in range(n_layers) if l != remove_layer]
        acc, avg_p, _ = evaluate_skeleton(model, tok, active, test_cases, device)
        ablation_results.append({
            'removed_layer': remove_layer,
            'accuracy': acc,
            'avg_prob': avg_p,
        })

    # ================================================================
    # Visualization
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Configuration comparison
    ax = axes[0]
    names = list(config_results.keys())
    accs = [config_results[n]['accuracy'] for n in names]
    pcts = [config_results[n]['pct_params'] for n in names]
    colors_map = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), accs, color=colors_map, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}\n({p:.0f}%)" for n, p in zip(names, pcts)], fontsize=8)
    ax.set_xlabel('Top-5 Accuracy')
    ax.set_title('(a) Skeleton Configurations')
    ax.set_xlim(0, 1.1)

    # (b) Accuracy vs parameters
    ax = axes[1]
    ax.scatter(pcts, accs, s=100, c=colors_map, edgecolors='black', zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name.replace('_', '\n'), (pcts[i], accs[i]),
                   fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('% Parameters Active')
    ax.set_ylabel('Top-5 Accuracy')
    ax.set_title('(b) Accuracy vs Model Size')
    ax.axhline(y=accs[0], color='gray', ls='--', alpha=0.5, label='Full model')
    ax.legend()

    # (c) Layer ablation
    ax = axes[2]
    layers = [r['removed_layer'] for r in ablation_results]
    abl_accs = [r['accuracy'] for r in ablation_results]
    ax.bar(layers, abl_accs, color='#3498db', alpha=0.7)
    for li in register_layers:
        ax.axvline(x=li, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Removed Layer')
    ax.set_ylabel('Accuracy After Removal')
    ax.set_title('(c) Single-Layer Ablation')

    fig.suptitle("Phase 4: Skeleton LLM\n"
                 "Does a tiny sub-graph contain the reasoning engine?",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "phase4_skeleton_llm")
    plt.close()

    # ================================================================
    # Verdict
    # ================================================================
    reg_acc = config_results['registers_only']['accuracy']
    full_acc = config_results['full_model']['accuracy']
    reg_pct = config_results['registers_only']['pct_params']

    if reg_acc >= full_acc * 0.8:
        verdict = (f"PARASITIC GLASSBOX CONFIRMED: {reg_pct:.0f}% of layers achieve "
                   f"{reg_acc:.0%} accuracy (vs full model {full_acc:.0%}). "
                   f"The reasoning engine is a tiny sub-graph.")
    elif reg_acc >= full_acc * 0.5:
        verdict = (f"PARTIAL GLASSBOX: {reg_pct:.0f}% of layers achieve "
                   f"{reg_acc:.0%} accuracy ({reg_acc/full_acc:.0%} of full). "
                   f"Register layers are important but not sufficient.")
    else:
        verdict = (f"DISTRIBUTED COMPUTATION: Registers achieve only "
                   f"{reg_acc:.0%} ({reg_pct:.0f}% params). "
                   f"Reasoning requires distributed processing.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 4: Skeleton LLM',
        'summary': {'verdict': verdict, 'register_accuracy': reg_acc,
                    'full_accuracy': full_acc, 'register_pct': reg_pct},
        'configs': {k: {'accuracy': v['accuracy'], 'avg_prob': v['avg_prob'],
                       'pct_params': v['pct_params']}
                   for k, v in config_results.items()},
        'ablation': ablation_results,
    }
    save_results("phase4_skeleton_llm", result)
    return result


if __name__ == '__main__':
    main()
