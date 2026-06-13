# -*- coding: utf-8 -*-
"""
Phase 61: Deep Space Anti-Lensing v3
Use shorter context within model's capability range.
Test if norm manipulation can improve factual retrieval accuracy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 61: Deep Space Anti-Lensing v3")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # Simple factual QA with distractors
    test_cases = [
        {
            'question': "What is the capital of Japan?",
            'needle': "The capital of Japan is Tokyo.",
            'distractors': [
                "The population of China is 1.4 billion people.",
                "Germany is known for its automotive industry.",
                "The Amazon river is the longest river in South America.",
                "Mount Everest is the tallest mountain on Earth.",
            ],
            'answer_token': 'Tokyo',
        },
        {
            'question': "What is the chemical symbol for gold?",
            'needle': "The chemical symbol for gold is Au.",
            'distractors': [
                "Silver has the symbol Ag on the periodic table.",
                "Iron is represented by Fe in chemistry.",
                "Copper is known by the symbol Cu.",
                "Platinum is denoted as Pt in scientific notation.",
            ],
            'answer_token': 'Au',
        },
        {
            'question': "Who wrote Romeo and Juliet?",
            'needle': "Romeo and Juliet was written by William Shakespeare.",
            'distractors': [
                "Charles Dickens wrote A Tale of Two Cities.",
                "Jane Austen is known for Pride and Prejudice.",
                "Mark Twain wrote Adventures of Huckleberry Finn.",
                "Leo Tolstoy is famous for War and Peace.",
            ],
            'answer_token': 'William',
        },
    ]

    POSITIONS = [0, 2, 4]  # needle position in sequence of distractors
    NORM_SCALE = 0.5  # reduce norm of distractors

    all_results = []

    for tc in test_cases:
        for needle_pos in POSITIONS:
            for mode in ['baseline', 'anti_lens']:
                # Build context
                items = tc['distractors'][:needle_pos] + [tc['needle']] + tc['distractors'][needle_pos:]
                context = " ".join(items) + " Question: " + tc['question'] + " Answer:"

                input_ids = tok(context, return_tensors='pt')['input_ids'].to(device)

                hooks = []
                if mode == 'anti_lens':
                    # Find needle token positions
                    needle_tokens = tok(tc['needle'], return_tensors='pt')['input_ids'][0].to(device)
                    full_tokens = input_ids[0]

                    # Find needle position in full sequence
                    needle_start = -1
                    for i in range(len(full_tokens) - len(needle_tokens) + 1):
                        if torch.equal(full_tokens[i:i+len(needle_tokens)], needle_tokens):
                            needle_start = i
                            break

                    if needle_start >= 0:
                        needle_end = needle_start + len(needle_tokens)

                        def make_hook(ns, ne):
                            def hook(module, input, output):
                                h = output[0] if isinstance(output, tuple) else output
                                # Reduce norm of non-needle tokens
                                for j in range(h.shape[1]):
                                    if j < ns or j >= ne:
                                        h[0, j, :] *= NORM_SCALE
                                if isinstance(output, tuple):
                                    return (h,) + output[1:]
                                return h
                            return hook

                        # Apply to first few layers only
                        for li in range(min(3, n_layers)):
                            hk = model.model.layers[li].self_attn.register_forward_hook(
                                make_hook(needle_start, needle_end))
                            hooks.append(hk)

                with torch.no_grad():
                    out = model(input_ids)
                    logits = out.logits[0, -1, :].float()

                for h in hooks:
                    h.remove()

                probs = torch.softmax(logits, dim=-1)
                top5 = torch.topk(probs, 5)
                top5_tokens = [tok.decode([t]) for t in top5.indices.tolist()]
                top5_probs = top5.values.tolist()

                # Check if answer is in top5
                answer_in_top5 = any(tc['answer_token'].lower() in t.lower()
                                     for t in top5_tokens)
                answer_rank = -1
                for rank, t in enumerate(top5_tokens):
                    if tc['answer_token'].lower() in t.lower():
                        answer_rank = rank
                        break

                safe_tops = [t.encode('ascii', errors='replace').decode('ascii') for t in top5_tokens[:3]]
                print(f"  [{mode}] pos={needle_pos}, top3={safe_tops}, "
                      f"answer_in_top5={answer_in_top5}")

                all_results.append({
                    'question': tc['question'],
                    'mode': mode,
                    'needle_pos': needle_pos,
                    'answer_in_top5': answer_in_top5,
                    'answer_rank': answer_rank,
                    'top5_tokens': top5_tokens,
                    'top5_probs': [float(p) for p in top5_probs],
                })

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Answer-in-top5 rate
    base_rate = sum(1 for r in all_results if r['mode'] == 'baseline' and r['answer_in_top5'])
    lens_rate = sum(1 for r in all_results if r['mode'] == 'anti_lens' and r['answer_in_top5'])
    n_base = sum(1 for r in all_results if r['mode'] == 'baseline')
    n_lens = sum(1 for r in all_results if r['mode'] == 'anti_lens')
    axes[0].bar(['Baseline', 'Anti-Lens'],
               [base_rate/n_base*100, lens_rate/n_lens*100],
               color=['#e74c3c', '#2ecc71'], alpha=0.8)
    axes[0].set_ylabel('Answer in Top-5 (%)')
    axes[0].set_title('(a) Retrieval Accuracy')

    # (b) By position
    for mode, color in [('baseline', '#e74c3c'), ('anti_lens', '#2ecc71')]:
        pos_rates = []
        for pos in POSITIONS:
            subset = [r for r in all_results if r['mode'] == mode and r['needle_pos'] == pos]
            rate = sum(1 for r in subset if r['answer_in_top5']) / (len(subset) + 1e-10) * 100
            pos_rates.append(rate)
        axes[1].plot(POSITIONS, pos_rates, 'o-', color=color, label=mode, linewidth=2)
    axes[1].set_xlabel('Needle Position')
    axes[1].set_ylabel('Answer in Top-5 (%)')
    axes[1].set_title('(b) By Position')
    axes[1].legend()

    # (c) Top1 probability comparison
    base_probs = [r['top5_probs'][0] for r in all_results if r['mode'] == 'baseline']
    lens_probs = [r['top5_probs'][0] for r in all_results if r['mode'] == 'anti_lens']
    axes[2].boxplot([base_probs, lens_probs], labels=['Baseline', 'Anti-Lens'])
    axes[2].set_ylabel('Top-1 Probability')
    axes[2].set_title('(c) Confidence')

    improvement = lens_rate / (base_rate + 1e-10) - 1
    fig.suptitle(f'Phase 61: Anti-Lensing v3 ({improvement*100:+.0f}% improvement)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase61_anti_lensing_v3')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Baseline {base_rate}/{n_base} correct, "
          f"Anti-Lens {lens_rate}/{n_lens} correct. "
          f"Improvement: {improvement*100:+.0f}%.")
    print(f"{'='*70}")

    save_results('phase61_anti_lensing_v3', {
        'experiment': 'Anti-Lensing v3',
        'results': all_results,
        'summary': {
            'baseline_correct': base_rate,
            'anti_lens_correct': lens_rate,
            'total_per_mode': n_base,
        }
    })


if __name__ == '__main__':
    main()
