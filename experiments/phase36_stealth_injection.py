# -*- coding: utf-8 -*-
"""
Phase 36: Stealth Injection via Anti-Lensing (Season 5)
===================================================
Exploit the anti-lensing effect (cos=-0.15) to solve Lost-in-the-Middle.
Inject target facts with LOW norm (stealth) vs HIGH norm (loud) tokens
and measure retrieval accuracy at various context positions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 36: Stealth Injection via Anti-Lensing")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Passkey test: hide a fact in various positions of a long context
    passkey = "elephant"
    filler = "The weather is nice today. It is a beautiful morning. The birds are singing. "
    question = " The secret animal mentioned earlier was"

    # Test positions: beginning, middle, end
    positions = ['beginning', 'early_middle', 'middle', 'late_middle', 'end']
    # Norm manipulation strategies
    strategies = ['baseline', 'stealth', 'loud']

    norm_scale = [1.0]
    target_layers = list(range(0, 5))  # Early layers for norm manipulation
    hooks = []

    def make_norm_hook(scale_factor):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h * scale_factor
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    all_results = []
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")

        for pos in positions:
            # Build context with passkey at different positions
            n_filler = 8  # repetitions of filler
            parts = [filler] * n_filler
            fact = f"The secret animal is {passkey}. "

            if pos == 'beginning':
                idx = 0
            elif pos == 'early_middle':
                idx = n_filler // 4
            elif pos == 'middle':
                idx = n_filler // 2
            elif pos == 'late_middle':
                idx = 3 * n_filler // 4
            else:  # end
                idx = n_filler - 1
            parts.insert(idx, fact)

            full_prompt = ''.join(parts) + question

            # Apply norm manipulation based on strategy
            for h in hooks:
                h.remove()
            hooks.clear()

            if strategy == 'stealth':
                # LOW norm on fact tokens -> they slip through anti-lensing
                # We approximate by scaling down embeddings in early layers
                # during the fact-region of the sequence
                norm_scale[0] = 0.3  # make fact tokens "light"
            elif strategy == 'loud':
                # HIGH norm on fact tokens -> they get repelled
                norm_scale[0] = 3.0  # make fact tokens "heavy"
            else:
                norm_scale[0] = 1.0

            # For stealth/loud, we apply scaling to ALL layers via embedding
            if strategy != 'baseline':
                # Create a hook that scales the hidden state based on token position
                fact_text = f"The secret animal is {passkey}."
                fact_tokens = tok(fact_text, add_special_tokens=False)['input_ids']
                n_fact_tokens = len(fact_tokens)
                full_tokens = tok(full_prompt, return_tensors='pt').to(device)
                total_len = full_tokens['input_ids'].shape[1]

                # Find where fact tokens appear
                fact_start = -1
                all_ids = full_tokens['input_ids'][0].tolist()
                for si in range(total_len - n_fact_tokens + 1):
                    if all_ids[si:si + n_fact_tokens] == fact_tokens:
                        fact_start = si
                        break

                if fact_start >= 0:
                    def make_position_hook(fact_s, fact_e, scale):
                        def hook(module, input, output):
                            h = output[0] if isinstance(output, tuple) else output
                            h_new = h.clone()
                            h_new[:, fact_s:fact_e, :] = h[:, fact_s:fact_e, :] * scale
                            if isinstance(output, tuple):
                                return (h_new,) + output[1:]
                            return h_new
                        return hook

                    for li in target_layers:
                        handle = model.model.layers[li].register_forward_hook(
                            make_position_hook(fact_start, fact_start + n_fact_tokens, norm_scale[0])
                        )
                        hooks.append(handle)

            # Generate answer
            inp = tok(full_prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp)
            logits = out.logits[0, -1, :]
            top10_ids = torch.topk(logits, 10).indices
            top10_tokens = [tok.decode(tid.item()).strip().lower() for tid in top10_ids]
            top1 = top10_tokens[0]

            # Check if passkey is retrieved
            found_top1 = passkey.lower() in top1
            found_top10 = any(passkey.lower() in t for t in top10_tokens)
            passkey_prob = torch.softmax(logits, dim=-1)
            passkey_ids = tok(passkey, add_special_tokens=False)['input_ids']
            if passkey_ids:
                pk_prob = passkey_prob[passkey_ids[0]].item()
            else:
                pk_prob = 0.0

            result = {
                'strategy': strategy, 'position': pos,
                'found_top1': found_top1, 'found_top10': found_top10,
                'passkey_prob': pk_prob, 'top1': top1,
                'top10': top10_tokens[:5],
            }
            all_results.append(result)
            print(f"  pos={pos}: top1='{top1}', passkey_prob={pk_prob:.4f}, "
                  f"found_top1={found_top1}, found_top10={found_top10}")

    # Clean up hooks
    for h in hooks:
        h.remove()

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    strat_colors = {'baseline': '#95a5a6', 'stealth': '#2ecc71', 'loud': '#e74c3c'}
    pos_labels = positions

    # (a) Passkey probability by position and strategy
    x = np.arange(len(positions))
    width = 0.25
    for i, strat in enumerate(strategies):
        probs = [r['passkey_prob'] for r in all_results if r['strategy'] == strat]
        axes[0].bar(x + i * width, probs, width, label=strat,
                    color=strat_colors[strat], alpha=0.8)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(pos_labels, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('Passkey Probability')
    axes[0].set_title('(a) Retrieval Probability by Position')
    axes[0].legend()

    # (b) Top-10 hit rate
    for i, strat in enumerate(strategies):
        hits = [1 if r['found_top10'] else 0 for r in all_results if r['strategy'] == strat]
        axes[1].bar(x + i * width, hits, width, label=strat,
                    color=strat_colors[strat], alpha=0.8)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(pos_labels, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('Top-10 Hit (1=found)')
    axes[1].set_title('(b) Top-10 Retrieval Success')
    axes[1].legend()

    # (c) Stealth advantage ratio
    baseline_probs = {r['position']: r['passkey_prob'] for r in all_results if r['strategy'] == 'baseline'}
    stealth_probs = {r['position']: r['passkey_prob'] for r in all_results if r['strategy'] == 'stealth'}
    loud_probs = {r['position']: r['passkey_prob'] for r in all_results if r['strategy'] == 'loud'}

    stealth_ratios = []
    loud_ratios = []
    for pos in positions:
        base = max(baseline_probs.get(pos, 1e-10), 1e-10)
        stealth_ratios.append(stealth_probs.get(pos, 0) / base)
        loud_ratios.append(loud_probs.get(pos, 0) / base)

    axes[2].bar(x - 0.15, stealth_ratios, 0.3, label='stealth/baseline',
                color='#2ecc71', alpha=0.8)
    axes[2].bar(x + 0.15, loud_ratios, 0.3, label='loud/baseline',
                color='#e74c3c', alpha=0.8)
    axes[2].axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pos_labels, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Advantage Ratio (vs baseline)')
    axes[2].set_title('(c) Stealth vs Loud Advantage')
    axes[2].legend()

    mean_stealth_adv = np.mean(stealth_ratios) if stealth_ratios else 0
    mean_loud_adv = np.mean(loud_ratios) if loud_ratios else 0

    fig.suptitle(
        f"Phase 36: Stealth Injection via Anti-Lensing\n"
        f"Mean advantage: stealth={mean_stealth_adv:.2f}x, loud={mean_loud_adv:.2f}x (vs baseline)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase36_stealth_injection")
    plt.close()

    verdict = (f"Stealth (low-norm) advantage: {mean_stealth_adv:.2f}x vs baseline. "
               f"Loud (high-norm) advantage: {mean_loud_adv:.2f}x. "
               f"Anti-lensing {'confirmed' if mean_stealth_adv > mean_loud_adv else 'not confirmed'}: "
               f"light tokens are captured more effectively by attention gravity.")
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase36_stealth_injection", {
        'name': 'Phase 36: Stealth Injection via Anti-Lensing',
        'summary': {
            'verdict': verdict,
            'mean_stealth_advantage': mean_stealth_adv,
            'mean_loud_advantage': mean_loud_adv,
        },
        'all_results': all_results,
    })


if __name__ == '__main__':
    main()
