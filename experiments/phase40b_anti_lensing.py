# -*- coding: utf-8 -*-
"""
Phase 40b: Deep Space Anti-Lensing v2
Fix: Use shorter filler (~15 tokens) to keep all 5 needle positions within 2048 tokens.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 40b: Deep Space Anti-Lensing v2 (Fixed)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Short filler sentences (~15 tokens each)
    fillers = [
        "The cat sat quietly on the warm windowsill. ",
        "A gentle breeze blew through the open door. ",
        "The old clock ticked steadily on the mantle. ",
        "Rain fell softly against the glass windows. ",
        "Leaves rustled in the cool autumn wind outside. ",
        "The dog slept peacefully beside the fireplace. ",
        "A small bird sang from the garden fence post. ",
        "The kettle whistled loudly in the tiny kitchen. ",
        "Clouds drifted slowly across the pale blue sky. ",
        "The river flowed calmly through the green valley. ",
    ]

    needle = "The secret access code is BLUE-TIGER-42."
    question = "\n\nBased on the information above, what is the secret access code? The code is"
    passkey_answer = "BLUE"  # First word of the answer

    # Build haystack: ~120 filler units = ~1800 tokens, well within 2048
    N_FILLER = 120

    positions = {
        'beginning': 5,
        'early': N_FILLER // 4,
        'middle': N_FILLER // 2,
        'late': 3 * N_FILLER // 4,
        'end': N_FILLER - 5,
    }

    STRATEGIES = {
        'baseline': 1.0,
        'stealth': 0.3,   # reduce needle norms
        'amplify': 3.0,   # boost needle norms
    }

    all_results = []

    for pos_name, pos_idx in positions.items():
        # Build haystack
        parts = []
        for i in range(N_FILLER):
            if i == pos_idx:
                parts.append(needle + " ")
            parts.append(fillers[i % len(fillers)])
        context = ''.join(parts) + question

        input_ids = tok(context, return_tensors='pt', truncation=True,
                       max_length=2048)['input_ids'].to(device)
        total_tokens = input_ids.shape[1]

        # Find needle token positions
        needle_ids = tok(needle, return_tensors='pt')['input_ids'][0].tolist()
        input_list = input_ids[0].tolist()
        needle_start = -1
        for i in range(len(input_list) - len(needle_ids) + 1):
            if input_list[i:i+len(needle_ids)] == needle_ids:
                needle_start = i
                break

        if needle_start < 0:
            print(f"  [{pos_name}] Needle NOT found in {total_tokens} tokens - SKIPPED")
            all_results.append({
                'position': pos_name, 'pos_idx': pos_idx,
                'total_tokens': total_tokens, 'found': False,
            })
            continue

        needle_len = len(needle_ids)
        rel_pos = needle_start / total_tokens
        print(f"\n  [{pos_name}] {total_tokens} tokens, needle at {needle_start} ({rel_pos:.1%})")

        strategy_results = {}

        for strat_name, scale in STRATEGIES.items():
            hooks = []

            if strat_name != 'baseline':
                def make_hook(s, ns, nl):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        if h.dim() == 3 and h.shape[1] > ns + nl:
                            h_mod = h.clone()
                            h_mod[0, ns:ns+nl, :] = (h_mod[0, ns:ns+nl, :].float() * s).to(h.dtype)
                            if isinstance(output, tuple):
                                return (h_mod,) + output[1:]
                            return h_mod
                        return output
                    return hook

                for li in range(min(6, len(model.model.layers))):
                    hh = model.model.layers[li].register_forward_hook(
                        make_hook(scale, needle_start, needle_len))
                    hooks.append(hh)

            with torch.no_grad():
                out = model(input_ids)
                logits = out.logits[0, -1, :].float()

            for hh in hooks:
                hh.remove()

            probs = torch.softmax(logits, dim=-1)
            passkey_ids = tok(passkey_answer, return_tensors='pt')['input_ids'][0]
            first_id = passkey_ids[0].item()

            top20_ids = torch.topk(probs, 20).indices.tolist()
            top1_token = tok.decode([top20_ids[0]]).strip()
            passkey_prob = probs[first_id].item()
            passkey_rank = -1
            for rank_idx, tid in enumerate(top20_ids):
                if tid == first_id:
                    passkey_rank = rank_idx + 1
                    break

            found_top1 = first_id == top20_ids[0]
            found_top5 = first_id in top20_ids[:5]
            found_top20 = first_id in top20_ids

            strategy_results[strat_name] = {
                'top1': top1_token,
                'passkey_prob': float(passkey_prob),
                'passkey_rank': passkey_rank,
                'found_top1': found_top1,
                'found_top5': found_top5,
                'found_top20': found_top20,
            }

            rank_str = f"rank={passkey_rank}" if passkey_rank > 0 else "not in top-20"
            print(f"    {strat_name}: top1='{top1_token}', prob={passkey_prob:.6f}, {rank_str}")

        all_results.append({
            'position': pos_name, 'pos_idx': pos_idx,
            'total_tokens': total_tokens, 'found': True,
            'needle_token_pos': needle_start, 'relative_pos': rel_pos,
            **strategy_results,
        })

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    valid = [r for r in all_results if r.get('found')]
    pos_labels = [f"{r['position']}\n({r['relative_pos']:.0%})" for r in valid]
    x = np.arange(len(pos_labels))

    for ax_idx, (strat, color, marker) in enumerate([
        ('baseline', '#3498db', 'o'), ('stealth', '#2ecc71', 's'), ('amplify', '#e74c3c', '^')
    ]):
        probs_list = [r[strat]['passkey_prob'] for r in valid]
        ranks = [r[strat].get('passkey_rank', -1) for r in valid]

        axes[ax_idx].bar(x, probs_list, color=color, alpha=0.7)
        axes[ax_idx].set_title(f'{strat.capitalize()} (scale={STRATEGIES[strat]})')
        axes[ax_idx].set_xlabel('Needle Position')
        axes[ax_idx].set_ylabel('Passkey Probability')
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(pos_labels, fontsize=8)

        for i, (p, rk) in enumerate(zip(probs_list, ranks)):
            label = f"R{rk}" if rk > 0 else "miss"
            axes[ax_idx].annotate(label, (i, p), ha='center', va='bottom', fontsize=7)

    fig.suptitle('Phase 40b: Deep Space Anti-Lensing v2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase40b_anti_lensing')
    plt.close()

    # Verdict
    for strat in ['baseline', 'stealth', 'amplify']:
        top5 = sum(1 for r in valid if r[strat].get('found_top5', False))
        print(f"  {strat}: {top5}/{len(valid)} in top-5")

    baseline_probs = [r['baseline']['passkey_prob'] for r in valid]
    stealth_probs = [r['stealth']['passkey_prob'] for r in valid]
    amplify_probs = [r['amplify']['passkey_prob'] for r in valid]
    b_mean = np.mean(baseline_probs) if baseline_probs else 0
    s_mean = np.mean(stealth_probs) if stealth_probs else 0
    a_mean = np.mean(amplify_probs) if amplify_probs else 0

    print(f"\n{'='*70}")
    print(f"VERDICT: Mean passkey prob: baseline={b_mean:.6f}, stealth={s_mean:.6f}, "
          f"amplify={a_mean:.6f}. "
          f"Stealth {'IMPROVES' if s_mean > b_mean else 'no improvement'}, "
          f"Amplify {'IMPROVES' if a_mean > b_mean else 'no improvement'}.")
    print(f"{'='*70}")

    save_results('phase40b_anti_lensing', {
        'experiment': 'Deep Space Anti-Lensing v2',
        'results': all_results,
        'summary': {
            'baseline_mean_prob': b_mean,
            'stealth_mean_prob': s_mean,
            'amplify_mean_prob': a_mean,
            'n_valid_positions': len(valid),
        }
    })


if __name__ == '__main__':
    main()
