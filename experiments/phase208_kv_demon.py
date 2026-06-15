# -*- coding: utf-8 -*-
"""
Phase 208: Thermodynamic KV-Demon (Maxwell's Demon for KV Cache)
================================================================
Phase 204 showed that autoregressive generation causes "crystallization"
(eta -> 1.0, entropy -> 0). Cold tokens no longer contribute to inference.

Maxwell's Demon: instead of evicting oldest tokens (FIFO), selectively
evict only thermodynamically "cold" tokens that have crystallized.
Keep "hot" tokens that actively drive inference.

This enables infinite-context illusion with fixed VRAM by pruning dead weight.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

LONG_PROMPTS = [
    ("The history of artificial intelligence begins with ancient myths about "
     "intelligent artifacts. Modern AI research started in the 1950s when "
     "Alan Turing proposed the Turing test. Early AI programs could prove "
     "mathematical theorems and play chess. The field experienced several "
     "winters when progress stalled and funding dried up. Neural networks "
     "were proposed in the 1940s but only became practical with backpropagation "
     "in the 1980s. Deep learning revolutionized the field starting around 2012 "
     "with AlexNet winning the ImageNet competition. Transformer architecture "
     "introduced attention mechanisms that enabled large language models. "
     "Today AI systems can write code, generate images, and reason about "
     "complex problems in ways that seemed impossible just a decade ago."),
    ("The universe began approximately 13.8 billion years ago with the Big Bang. "
     "In the first few minutes, hydrogen and helium nuclei formed through "
     "nucleosynthesis. Over millions of years, gravity pulled matter into "
     "the first stars and galaxies. Heavy elements were forged in stellar "
     "cores and distributed through supernova explosions. Our solar system "
     "formed about 4.6 billion years ago from a cloud of gas and dust. "
     "Life appeared on Earth roughly 3.5 billion years ago as simple "
     "single-celled organisms. Complex multicellular life emerged much later. "
     "Humans evolved in Africa and spread across the globe. We developed "
     "agriculture, writing, science, and technology at an accelerating pace."),
]

# VRAM budget: keep only this many tokens in KV cache
KV_BUDGETS = [32, 48, 64, 96, 128]
# Eviction strategies
STRATEGIES = ['fifo', 'coldest', 'random', 'hottest_evict']


def compute_token_temperatures(model, tok, device, prompt):
    """Compute per-token temperature at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Measure T at L0=21 (the phase transition) for each token position
    L0_idx = min(21, len(out.hidden_states) - 1)
    hs_L0 = out.hidden_states[L0_idx]  # (1, seq, hidden)

    token_temps = []
    for pos in range(seq_len):
        with torch.no_grad():
            normed = norm_layer(hs_L0[:, pos:pos+1, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        token_temps.append(T if not np.isnan(T) else 0)

    # Also get baseline output (full context)
    final_logits = out.logits[0, -1, :].float()
    final_probs = torch.softmax(final_logits, dim=-1)
    baseline_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
    baseline_top1 = final_probs.max().item()
    baseline_token = tok.decode(final_logits.argmax().item())

    return {
        'token_temps': token_temps,
        'seq_len': seq_len,
        'baseline_entropy': baseline_entropy,
        'baseline_top1': baseline_top1,
        'baseline_token': baseline_token,
    }


def select_tokens_to_keep(token_temps, budget, strategy):
    """Select which token indices to keep given a budget."""
    seq_len = len(token_temps)
    if budget >= seq_len:
        return list(range(seq_len))

    # Always keep first token and last token
    must_keep = {0, seq_len - 1}
    available = list(range(1, seq_len - 1))
    n_select = budget - len(must_keep)

    if strategy == 'fifo':
        # Keep most recent tokens
        selected = available[-n_select:] if n_select > 0 else []
    elif strategy == 'coldest':
        # Demon: evict coldest, keep hottest
        temps_with_idx = [(token_temps[i], i) for i in available]
        temps_with_idx.sort(reverse=True)  # Hottest first
        selected = [idx for _, idx in temps_with_idx[:n_select]]
    elif strategy == 'random':
        np.random.seed(42)
        selected = list(np.random.choice(available, size=min(n_select, len(available)),
                                         replace=False))
    elif strategy == 'hottest_evict':
        # Anti-demon: evict hottest, keep coldest (control)
        temps_with_idx = [(token_temps[i], i) for i in available]
        temps_with_idx.sort()  # Coldest first
        selected = [idx for _, idx in temps_with_idx[:n_select]]
    else:
        selected = available[:n_select]

    keep_set = sorted(list(must_keep) + selected)
    return keep_set


def run_with_eviction(model, tok, device, prompt, budget, strategy, token_temps):
    """Re-run forward pass keeping only selected tokens."""
    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']
    seq_len = input_ids.shape[1]

    keep_indices = select_tokens_to_keep(token_temps, budget, strategy)

    # Create attention mask: 1 for kept tokens, 0 for evicted
    attention_mask = torch.zeros(1, seq_len, device=device, dtype=torch.long)
    for idx in keep_indices:
        attention_mask[0, idx] = 1

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    final_logits = out.logits[0, -1, :].float()
    final_probs = torch.softmax(final_logits, dim=-1)
    output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
    top1_prob = final_probs.max().item()
    top_token = tok.decode(final_logits.argmax().item())

    return {
        'output_entropy': output_entropy,
        'top1_prob': top1_prob,
        'top_token': top_token,
        'n_kept': len(keep_indices),
        'kept_ratio': len(keep_indices) / seq_len,
    }


def main():
    print("=" * 70)
    print("Phase 208: Thermodynamic KV-Demon")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    all_results = []
    example_temps = None

    for pi, prompt in enumerate(LONG_PROMPTS):
        print(f"\n--- Prompt {pi+1}/{len(LONG_PROMPTS)} ---")

        # Step 1: Compute per-token temperatures
        temp_data = compute_token_temperatures(model, tok, device, prompt)
        token_temps = temp_data['token_temps']
        seq_len = temp_data['seq_len']
        print(f"  Seq len: {seq_len}, Baseline entropy: "
              f"{temp_data['baseline_entropy']:.3f}")

        if pi == 0:
            example_temps = token_temps

        # Step 2: Test eviction strategies at various budgets
        for budget in KV_BUDGETS:
            if budget >= seq_len:
                continue
            for strategy in STRATEGIES:
                r = run_with_eviction(model, tok, device, prompt,
                                     budget, strategy, token_temps)
                r['prompt_idx'] = pi
                r['budget'] = budget
                r['strategy'] = strategy
                r['baseline_entropy'] = temp_data['baseline_entropy']
                r['baseline_top1'] = temp_data['baseline_top1']
                r['entropy_ratio'] = r['output_entropy'] / (temp_data['baseline_entropy'] + 1e-10)
                all_results.append(r)
                print(f"  budget={budget}, {strategy}: "
                      f"entropy={r['output_entropy']:.3f} "
                      f"(ratio={r['entropy_ratio']:.2f}), "
                      f"top1={r['top1_prob']:.4f}")

    # Aggregate by strategy and budget
    agg = {}
    for strategy in STRATEGIES:
        agg[strategy] = {}
        for budget in KV_BUDGETS:
            entries = [r for r in all_results
                       if r['strategy'] == strategy and r['budget'] == budget]
            if entries:
                agg[strategy][budget] = {
                    'entropy_ratio_mean': float(np.mean([r['entropy_ratio'] for r in entries])),
                    'top1_mean': float(np.mean([r['top1_prob'] for r in entries])),
                }

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'fifo': '#e74c3c', 'coldest': '#2ecc71',
              'random': '#3498db', 'hottest_evict': '#95a5a6'}

    # (a) Entropy ratio vs budget for each strategy
    for strategy in STRATEGIES:
        if strategy in agg:
            budgets = sorted(agg[strategy].keys())
            ratios = [agg[strategy][b]['entropy_ratio_mean'] for b in budgets]
            axes[0, 0].plot(budgets, ratios, 'o-', color=colors[strategy],
                            label=strategy, markersize=6, lw=2)
    axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('KV Budget (tokens kept)')
    axes[0, 0].set_ylabel('Entropy Ratio (vs full)')
    axes[0, 0].set_title('(a) Output Quality vs KV Budget')
    axes[0, 0].legend(fontsize=7)

    # (b) Top-1 prob vs budget
    for strategy in STRATEGIES:
        if strategy in agg:
            budgets = sorted(agg[strategy].keys())
            top1s = [agg[strategy][b]['top1_mean'] for b in budgets]
            axes[0, 1].plot(budgets, top1s, 's-', color=colors[strategy],
                            label=strategy, markersize=6, lw=2)
    axes[0, 1].set_xlabel('KV Budget (tokens kept)')
    axes[0, 1].set_ylabel('Top-1 Probability')
    axes[0, 1].set_title('(b) Confidence vs KV Budget')
    axes[0, 1].legend(fontsize=7)

    # (c) Token temperature heatmap
    if example_temps:
        n_tokens = len(example_temps)
        temp_array = np.array(example_temps).reshape(1, -1)
        im = axes[0, 2].imshow(temp_array, aspect='auto', cmap='hot',
                               interpolation='nearest')
        axes[0, 2].set_xlabel('Token Position')
        axes[0, 2].set_yticks([])
        axes[0, 2].set_title('(c) Token Temperature Map')
        plt.colorbar(im, ax=axes[0, 2], label='Temperature T')

    # (d) Temperature histogram
    if example_temps:
        axes[1, 0].hist(example_temps, bins=30, color='#e67e22', alpha=0.7,
                        edgecolor='black')
        axes[1, 0].axvline(x=np.median(example_temps), color='red',
                           linestyle='--', label=f'median={np.median(example_temps):.2f}')
        axes[1, 0].set_xlabel('Token Temperature')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('(d) Temperature Distribution')
        axes[1, 0].legend(fontsize=8)

    # (e) Demon advantage: coldest vs fifo at each budget
    if 'coldest' in agg and 'fifo' in agg:
        budgets = sorted(set(agg['coldest'].keys()) & set(agg['fifo'].keys()))
        advantages = []
        for b in budgets:
            demon_ratio = agg['coldest'][b]['entropy_ratio_mean']
            fifo_ratio = agg['fifo'][b]['entropy_ratio_mean']
            advantage = (fifo_ratio - demon_ratio) / (fifo_ratio + 1e-10) * 100
            advantages.append(advantage)
        axes[1, 1].bar(range(len(budgets)), advantages, color='#2ecc71', alpha=0.7)
        axes[1, 1].set_xticks(range(len(budgets)))
        axes[1, 1].set_xticklabels([str(b) for b in budgets])
        axes[1, 1].set_xlabel('KV Budget')
        axes[1, 1].set_ylabel('Demon Advantage (%)')
        axes[1, 1].set_title('(e) Demon vs FIFO Improvement')
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # (f) Summary
    summary_parts = ["Thermodynamic KV-Demon\n"]
    for strategy in ['fifo', 'coldest']:
        if strategy in agg and 64 in agg[strategy]:
            r = agg[strategy][64]
            summary_parts.append(
                f"{strategy} (budget=64):\n"
                f"  entropy ratio = {r['entropy_ratio_mean']:.3f}\n"
                f"  top1 = {r['top1_mean']:.4f}\n"
            )
    summary_text = '\n'.join(summary_parts)
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 208: Thermodynamic KV-Demon",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase208_kv_demon')
    plt.close()

    print(f"\n{'=' * 70}")
    if 'coldest' in agg and 'fifo' in agg and 64 in agg.get('coldest', {}) and 64 in agg.get('fifo', {}):
        print(f"Demon (budget=64): entropy ratio = {agg['coldest'][64]['entropy_ratio_mean']:.3f}")
        print(f"FIFO  (budget=64): entropy ratio = {agg['fifo'][64]['entropy_ratio_mean']:.3f}")
    print(f"{'=' * 70}")

    save_results('phase208_kv_demon', {
        'experiment': 'Thermodynamic KV-Demon',
        'aggregated': {s: {str(b): v for b, v in bv.items()}
                       for s, bv in agg.items()},
        'example_temps': [float(t) for t in example_temps] if example_temps else [],
    })


if __name__ == '__main__':
    main()
