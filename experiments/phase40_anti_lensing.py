# -*- coding: utf-8 -*-
"""
Phase 40: Deep Space Anti-Lensing
Fix Phase 36's too-short context by using 4K+ token Needle-in-Haystack test.
Manipulate L2 norms to test anti-lensing effect on long-context retrieval.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 40: Deep Space Anti-Lensing (Long Context)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Generate filler text (boring, repetitive)
    filler_unit = "The weather today is pleasant and mild. Birds are singing in the trees. " \
                  "People walk along the streets going about their daily routines. " \
                  "The sky is clear and the sun is shining brightly. "
    # ~50 tokens per unit, we need ~80 units for ~4000 tokens
    filler_tokens = tok(filler_unit, return_tensors='pt')['input_ids'].shape[1]
    n_repeats = max(80, 4000 // filler_tokens)

    # The needle (hidden fact)
    needle = "The secret code for the vault is BLUE-TIGER-42. "
    passkey_answer = "BLUE-TIGER-42"

    # Question at the end
    question = "\nQuestion: What is the secret code for the vault? Answer: The secret code is"

    # Test positions: where to insert needle in the haystack
    positions = {
        'beginning': 2,
        'early_quarter': n_repeats // 4,
        'middle': n_repeats // 2,
        'late_quarter': 3 * n_repeats // 4,
        'end': n_repeats - 2,
    }

    # Strategies for norm manipulation
    STEALTH_SCALE = 0.3  # reduce needle token norms
    LOUD_SCALE = 3.0     # boost needle token norms

    all_results = []

    for pos_name, pos_idx in positions.items():
        print(f"\n--- Position: {pos_name} (unit {pos_idx}/{n_repeats}) ---")

        # Build haystack with needle
        parts = []
        for i in range(n_repeats):
            if i == pos_idx:
                parts.append(needle)
            parts.append(filler_unit)
        haystack = ''.join(parts) + question

        # Truncate to model's max length if needed
        input_ids = tok(haystack, return_tensors='pt', truncation=True,
                       max_length=4096)['input_ids'].to(device)
        total_tokens = input_ids.shape[1]

        # Find needle token positions in input
        needle_ids = tok(needle, return_tensors='pt')['input_ids'][0]
        needle_len = len(needle_ids)

        # Search for needle position in tokenized input
        needle_start = -1
        input_list = input_ids[0].tolist()
        needle_list = needle_ids.tolist()
        for i in range(len(input_list) - needle_len):
            if input_list[i:i+needle_len] == needle_list:
                needle_start = i
                break

        if needle_start < 0:
            print(f"  WARNING: Needle not found in tokenized input (truncated?)")
            # Needle was truncated - skip this position
            all_results.append({
                'position': pos_name, 'pos_idx': pos_idx,
                'total_tokens': total_tokens,
                'needle_found_in_context': False,
                'baseline': None, 'stealth': None, 'loud': None
            })
            continue

        print(f"  Total tokens: {total_tokens}, Needle at token {needle_start}-{needle_start+needle_len}")

        strategies = {}
        for strategy in ['baseline', 'stealth', 'loud']:
            hooks = []

            if strategy != 'baseline':
                scale = STEALTH_SCALE if strategy == 'stealth' else LOUD_SCALE

                def make_norm_hook(s, n_start, n_len):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        if h.dim() == 3 and h.shape[1] > n_start + n_len:
                            h_mod = h.clone()
                            h_mod[0, n_start:n_start+n_len, :] *= s
                            if isinstance(output, tuple):
                                return (h_mod,) + output[1:]
                            return h_mod
                        return output
                    return hook

                # Apply to early layers only (L0-L5) to influence attention
                for layer_idx in range(min(6, len(model.model.layers))):
                    h = model.model.layers[layer_idx].register_forward_hook(
                        make_norm_hook(scale, needle_start, needle_len))
                    hooks.append(h)

            # Run inference
            with torch.no_grad():
                out = model(input_ids)
                logits = out.logits[0, -1, :].float()

            for h in hooks:
                h.remove()

            probs = torch.softmax(logits, dim=-1)

            # Check if passkey tokens are in top predictions
            passkey_ids = tok(passkey_answer, return_tensors='pt')['input_ids'][0]
            first_passkey_id = passkey_ids[0].item()

            top10_ids = torch.topk(probs, 10).indices.tolist()
            top1_token = tok.decode([top10_ids[0]])
            passkey_prob = probs[first_passkey_id].item()
            found_top1 = first_passkey_id == top10_ids[0]
            found_top10 = first_passkey_id in top10_ids

            strategies[strategy] = {
                'top1': top1_token.strip(),
                'passkey_prob': passkey_prob,
                'found_top1': found_top1,
                'found_top10': found_top10,
            }
            print(f"  [{strategy}] top1='{top1_token.strip()}', prob={passkey_prob:.6f}, "
                  f"top1={'YES' if found_top1 else 'no'}, top10={'YES' if found_top10 else 'no'}")

        all_results.append({
            'position': pos_name, 'pos_idx': pos_idx,
            'total_tokens': total_tokens,
            'needle_found_in_context': True,
            'needle_token_pos': needle_start,
            **strategies
        })

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    valid_results = [r for r in all_results if r.get('needle_found_in_context')]
    pos_labels = [r['position'] for r in valid_results]
    x = np.arange(len(pos_labels))

    for ax_idx, (strategy, color) in enumerate([
        ('baseline', '#3498db'), ('stealth', '#2ecc71'), ('loud', '#e74c3c')
    ]):
        probs_list = []
        top1_list = []
        top10_list = []
        for r in valid_results:
            s = r.get(strategy, {})
            probs_list.append(s.get('passkey_prob', 0) if s else 0)
            top1_list.append(1 if s and s.get('found_top1') else 0)
            top10_list.append(1 if s and s.get('found_top10') else 0)

        axes[ax_idx].bar(x, probs_list, color=color, alpha=0.7)
        axes[ax_idx].set_title(f'{strategy.capitalize()} Strategy')
        axes[ax_idx].set_xlabel('Needle Position')
        axes[ax_idx].set_ylabel('Passkey Probability')
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(pos_labels, rotation=45, ha='right', fontsize=8)

        # Annotate top-1/top-10
        for i, (t1, t10) in enumerate(zip(top1_list, top10_list)):
            label = 'TOP-1' if t1 else ('top-10' if t10 else 'miss')
            axes[ax_idx].annotate(label, (i, probs_list[i]), ha='center',
                                 va='bottom', fontsize=7, fontweight='bold' if t1 else 'normal')

    fig.suptitle('Phase 40: Deep Space Anti-Lensing (4K+ Tokens)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase40_anti_lensing')
    plt.close()

    # === Verdict ===
    baseline_top1 = sum(1 for r in valid_results
                       if r.get('baseline', {}).get('found_top1', False))
    stealth_top1 = sum(1 for r in valid_results
                      if r.get('stealth', {}).get('found_top1', False))
    loud_top1 = sum(1 for r in valid_results
                   if r.get('loud', {}).get('found_top1', False))
    n_valid = len(valid_results) or 1

    print(f"\n{'='*70}")
    print(f"VERDICT: Baseline={baseline_top1}/{n_valid}, "
          f"Stealth={stealth_top1}/{n_valid}, Loud={loud_top1}/{n_valid}. "
          f"Anti-lensing {'CONFIRMED' if stealth_top1 > baseline_top1 else 'not confirmed'}.")
    print(f"{'='*70}")

    save_results('phase40_anti_lensing', {
        'experiment': 'Deep Space Anti-Lensing',
        'results': all_results,
        'summary': {
            'baseline_top1_rate': baseline_top1 / n_valid,
            'stealth_top1_rate': stealth_top1 / n_valid,
            'loud_top1_rate': loud_top1 / n_valid,
        }
    })


if __name__ == '__main__':
    main()
