# -*- coding: utf-8 -*-
"""
Phase 38: Thermodynamic Beam Reranking (Season 5 - Opus Original)
===================================================
Use PR*T stability as a reranking criterion during generation.
Generate multiple continuations, measure thermodynamic stability,
select the most stable. Tests whether thermodynamic stability
correlates with factual accuracy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def safe_str(s):
    """Encode string safely for cp932 console."""
    if isinstance(s, str):
        return s.encode('ascii', errors='replace').decode('ascii')
    return str(s)


def main():
    print("=" * 70)
    print("Phase 38: Thermodynamic Beam Reranking (Opus Original)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    N_BEAMS = 5
    GEN_STEPS = 30

    # Tasks with known correct answers for evaluation
    tasks = [
        ("The capital of Germany is", "Berlin"),
        ("The chemical formula for water is H2", "O"),
        ("Light travels at approximately 300000", "km"),
        ("The first president of the United States was George", "Washington"),
        ("In biology, DNA stands for deoxyribonucleic", "acid"),
        ("The smallest prime number is", "2"),
        ("The speed of sound in air is approximately 343", "m"),
        ("Mount Everest is located in the", "Him"),
    ]

    all_results = []
    for prompt, expected_substr in tasks:
        print(f"\n--- '{prompt[:45]}...' (expect: '{expected_substr}') ---")

        # Strategy 1: Greedy (baseline)
        # Strategy 2: Sampling + thermodynamic reranking
        strategies = {}

        for strategy in ['greedy', 'thermo_rerank', 'likelihood_rerank']:
            if strategy == 'greedy':
                n_candidates = 1
            else:
                n_candidates = N_BEAMS

            candidates = []
            for beam in range(n_candidates):
                inp = tok(prompt, return_tensors='pt').to(device)
                input_ids = inp['input_ids']
                past_kv = None
                generated = []
                prt_trace = []
                log_prob_sum = 0.0

                for t in range(GEN_STEPS):
                    if past_kv is None:
                        curr_input = input_ids
                    else:
                        curr_input = next_token_id

                    with torch.no_grad():
                        out = model(input_ids=curr_input, past_key_values=past_kv,
                                   use_cache=True, output_hidden_states=True)

                    past_kv = out.past_key_values
                    logits = out.logits[0, -1, :].float()
                    probs = torch.softmax(logits, dim=-1)

                    # Measure PR*T
                    PR = 1.0 / (probs ** 2).sum().item()
                    T = -(probs * torch.log(probs + 1e-10)).sum().item()
                    PRT = PR * T
                    prt_trace.append(PRT)

                    if strategy == 'greedy':
                        next_token_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
                    else:
                        # Temperature sampling for diversity
                        temp_probs = torch.softmax(logits / 1.2, dim=-1)
                        next_token_id = torch.multinomial(temp_probs, 1).unsqueeze(0)

                    token_prob = probs[next_token_id[0, 0]].item()
                    log_prob_sum += np.log(token_prob + 1e-10)
                    generated.append(tok.decode(next_token_id[0, 0].item()))

                text = ''.join(generated)
                prt_std = np.std(prt_trace) if prt_trace else 0
                prt_mean = np.mean(prt_trace) if prt_trace else 0

                candidates.append({
                    'text': text,
                    'prt_std': prt_std,
                    'prt_mean': prt_mean,
                    'log_prob': log_prob_sum,
                    'prt_trace': prt_trace,
                })

            # Reranking
            if strategy == 'thermo_rerank':
                # Select candidate with LOWEST PRT variance (most stable)
                best = min(candidates, key=lambda c: c['prt_std'])
            elif strategy == 'likelihood_rerank':
                # Select candidate with HIGHEST log probability
                best = max(candidates, key=lambda c: c['log_prob'])
            else:
                best = candidates[0]

            # Check correctness
            is_correct = expected_substr.lower() in best['text'][:20].lower()

            strategies[strategy] = {
                'text': best['text'][:40],
                'prt_std': best['prt_std'],
                'log_prob': best['log_prob'],
                'correct': is_correct,
                'n_candidates': len(candidates),
            }

            status = "OK" if is_correct else "MISS"
            print(f"  {strategy}: [{status}] prt_std={best['prt_std']:.0f}, "
                  f"text='{safe_str(best['text'][:30])}...'")

        all_results.append({
            'prompt': prompt[:45], 'expected': expected_substr,
            'strategies': strategies,
        })

    # === Compute accuracy per strategy ===
    strategy_accs = {}
    for strat in ['greedy', 'thermo_rerank', 'likelihood_rerank']:
        correct = sum(1 for r in all_results if r['strategies'][strat]['correct'])
        total = len(all_results)
        strategy_accs[strat] = correct / total

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Accuracy comparison
    strats = list(strategy_accs.keys())
    accs = [strategy_accs[s] * 100 for s in strats]
    colors_bar = ['#95a5a6', '#2ecc71', '#3498db']
    bars = axes[0].bar(range(len(strats)), accs, color=colors_bar, alpha=0.8)
    axes[0].set_xticks(range(len(strats)))
    axes[0].set_xticklabels(['Greedy', 'Thermo\nRerank', 'Likelihood\nRerank'], fontsize=10)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('(a) Accuracy by Strategy')
    axes[0].set_ylim(0, 110)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                     f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')

    # (b) PRT stability comparison
    for strat, color in zip(strats, colors_bar):
        stds = [r['strategies'][strat]['prt_std'] for r in all_results]
        axes[1].bar([f"{r['expected']}" for r in all_results],
                    stds, alpha=0.3, color=color, label=strat)
    axes[1].set_ylabel('PR*T Std Dev')
    axes[1].set_title('(b) Thermodynamic Stability')
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x', rotation=45)

    # (c) Correct vs incorrect PRT distributions
    correct_stds = []
    incorrect_stds = []
    for r in all_results:
        for strat in ['thermo_rerank', 'likelihood_rerank']:
            s = r['strategies'][strat]
            if s['correct']:
                correct_stds.append(s['prt_std'])
            else:
                incorrect_stds.append(s['prt_std'])

    if correct_stds and incorrect_stds:
        axes[2].hist(correct_stds, bins=10, alpha=0.6, color='#2ecc71', label='Correct', density=True)
        axes[2].hist(incorrect_stds, bins=10, alpha=0.6, color='#e74c3c', label='Incorrect', density=True)
    elif correct_stds:
        axes[2].hist(correct_stds, bins=10, alpha=0.6, color='#2ecc71', label='Correct', density=True)
    axes[2].set_xlabel('PR*T Std Dev')
    axes[2].set_ylabel('Density')
    axes[2].set_title('(c) Stability: Correct vs Incorrect')
    axes[2].legend()

    thermo_acc = strategy_accs['thermo_rerank'] * 100
    greedy_acc = strategy_accs['greedy'] * 100
    like_acc = strategy_accs['likelihood_rerank'] * 100
    improvement = thermo_acc - greedy_acc

    fig.suptitle(
        f"Phase 38: Thermodynamic Beam Reranking\n"
        f"Greedy={greedy_acc:.0f}%, Thermo={thermo_acc:.0f}% ({improvement:+.0f}%), "
        f"Likelihood={like_acc:.0f}%",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase38_thermo_rerank")
    plt.close()

    verdict = (
        f"Greedy: {greedy_acc:.0f}%, Thermo rerank: {thermo_acc:.0f}% "
        f"({improvement:+.0f}%), Likelihood rerank: {like_acc:.0f}%. "
        f"Thermodynamic stability {'correlates' if thermo_acc >= greedy_acc else 'does not correlate'} "
        f"with factual accuracy."
    )
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase38_thermo_rerank", {
        'name': 'Phase 38: Thermodynamic Beam Reranking',
        'summary': {
            'verdict': verdict,
            'accuracy': strategy_accs,
            'improvement_over_greedy': improvement,
        }
    })


if __name__ == '__main__':
    main()
