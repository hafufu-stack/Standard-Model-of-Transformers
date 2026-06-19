# -*- coding: utf-8 -*-
"""
Phase 269: Thermodynamic Decoding
===================================
The 6th law (P1*T ~ 0.84) implies a natural temperature for each decoding step.
Instead of fixed temperature, we dynamically set T = 0.84 / P1 at each step.

This creates a physically-grounded sampler where:
  - High confidence (P1 high) -> low T (exploit)
  - Low confidence (P1 low)  -> high T (explore)

Compared against: greedy, top-p(0.9), min-p(0.1), fixed T=0.7
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from utils import load_model, save_results, save_figure

P1T_CONSTANT = 0.84

TEST_PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics, the uncertainty principle states",
    "The most effective approach to solving climate change is",
    "Once upon a time, in a kingdom far away,",
    "The key difference between Python and Rust is",
]

MAX_NEW_TOKENS = 80
N_REPEAT = 3  # repeat each prompt for diversity stats


def decode_greedy(model, tok, input_ids, max_new=MAX_NEW_TOKENS):
    """Standard greedy decoding."""
    generated = input_ids.clone()
    p1_trace, t_trace = [], []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(generated)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        t_trace.append(t_val)
        next_id = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return generated, p1_trace, t_trace


def decode_top_p(model, tok, input_ids, top_p=0.9, temperature=1.0, max_new=MAX_NEW_TOKENS):
    """Top-p (nucleus) sampling."""
    generated = input_ids.clone()
    p1_trace, t_trace = [], []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(generated)
        logits = out.logits[0, -1, :].float() / temperature
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        t_trace.append(t_val)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum()
        next_idx = torch.multinomial(sorted_probs, 1)
        next_id = sorted_idx[next_idx].unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return generated, p1_trace, t_trace


def decode_thermodynamic(model, tok, input_ids, constant=P1T_CONSTANT, max_new=MAX_NEW_TOKENS):
    """Thermodynamic decoding: T_sample = constant / P1 at each step."""
    generated = input_ids.clone()
    p1_trace, t_trace, t_sample_trace = [], [], []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(generated)
        raw_logits = out.logits[0, -1, :].float()

        # Measure P1 from raw logits
        raw_probs = torch.softmax(raw_logits, dim=-1)
        p1 = raw_probs.max().item()
        t_natural = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        t_trace.append(t_natural)

        # Dynamic temperature from the conservation law
        t_sample = constant / max(p1, 0.01)  # avoid division by ~0
        t_sample = max(0.1, min(t_sample, 5.0))  # clamp for stability
        t_sample_trace.append(t_sample)

        # Apply temperature and sample
        scaled_logits = raw_logits / t_sample
        probs = torch.softmax(scaled_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return generated, p1_trace, t_trace, t_sample_trace


def decode_fixed_temp(model, tok, input_ids, temperature=0.7, max_new=MAX_NEW_TOKENS):
    """Fixed temperature sampling."""
    generated = input_ids.clone()
    p1_trace, t_trace = [], []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(generated)
        logits = out.logits[0, -1, :].float() / temperature
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        t_trace.append(t_val)
        next_id = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return generated, p1_trace, t_trace


def compute_metrics(tok, generated_ids, input_len):
    """Compute text quality metrics."""
    gen_ids = generated_ids[0, input_len:].tolist()
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    n_tokens = len(gen_ids)

    # Repetition rate: fraction of 3-grams that are repeated
    if n_tokens >= 3:
        trigrams = [tuple(gen_ids[i:i+3]) for i in range(n_tokens - 2)]
        counts = Counter(trigrams)
        rep_rate = sum(1 for c in counts.values() if c > 1) / max(len(trigrams), 1)
    else:
        rep_rate = 0.0

    # Unique token ratio
    unique_ratio = len(set(gen_ids)) / max(n_tokens, 1)

    return {
        'text': gen_text[:200],
        'n_tokens': n_tokens,
        'rep_rate': round(rep_rate, 4),
        'unique_ratio': round(unique_ratio, 4),
    }


def main():
    print("=" * 70)
    print("Phase 269: Thermodynamic Decoding (P1*T Feedback Sampler)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    all_results = {}
    method_metrics = {m: {'rep_rates': [], 'unique_ratios': [], 'p1_means': [], 't_means': []}
                      for m in ['greedy', 'top_p', 'fixed_t', 'thermodynamic']}

    for pi, prompt in enumerate(TEST_PROMPTS):
        print(f"\n--- Prompt {pi+1}/{len(TEST_PROMPTS)}: {prompt[:50]}... ---")
        inp = tok(prompt, return_tensors='pt').to(device)
        input_len = inp['input_ids'].shape[1]
        prompt_results = {}

        for trial in range(N_REPEAT):
            # Greedy
            gen, p1s, ts = decode_greedy(model, tok, inp['input_ids'])
            m = compute_metrics(tok, gen, input_len)
            method_metrics['greedy']['rep_rates'].append(m['rep_rate'])
            method_metrics['greedy']['unique_ratios'].append(m['unique_ratio'])
            method_metrics['greedy']['p1_means'].append(np.mean(p1s))
            method_metrics['greedy']['t_means'].append(np.mean(ts))

            # Top-p
            gen, p1s, ts = decode_top_p(model, tok, inp['input_ids'])
            m = compute_metrics(tok, gen, input_len)
            method_metrics['top_p']['rep_rates'].append(m['rep_rate'])
            method_metrics['top_p']['unique_ratios'].append(m['unique_ratio'])
            method_metrics['top_p']['p1_means'].append(np.mean(p1s))
            method_metrics['top_p']['t_means'].append(np.mean(ts))

            # Fixed T=0.7
            gen, p1s, ts = decode_fixed_temp(model, tok, inp['input_ids'])
            m = compute_metrics(tok, gen, input_len)
            method_metrics['fixed_t']['rep_rates'].append(m['rep_rate'])
            method_metrics['fixed_t']['unique_ratios'].append(m['unique_ratio'])
            method_metrics['fixed_t']['p1_means'].append(np.mean(p1s))
            method_metrics['fixed_t']['t_means'].append(np.mean(ts))

            # Thermodynamic
            gen, p1s, ts, t_samps = decode_thermodynamic(model, tok, inp['input_ids'])
            m = compute_metrics(tok, gen, input_len)
            m['t_sample_trace'] = [round(x, 4) for x in t_samps[:20]]
            method_metrics['thermodynamic']['rep_rates'].append(m['rep_rate'])
            method_metrics['thermodynamic']['unique_ratios'].append(m['unique_ratio'])
            method_metrics['thermodynamic']['p1_means'].append(np.mean(p1s))
            method_metrics['thermodynamic']['t_means'].append(np.mean(ts))

            if trial == 0:
                prompt_results = {
                    'greedy_text': compute_metrics(tok, decode_greedy(model, tok, inp['input_ids'])[0], input_len)['text'],
                    'thermo_text': m['text'],
                    'thermo_t_samples': m['t_sample_trace'],
                }

        all_results[f'prompt_{pi}'] = prompt_results

    # Summary statistics
    summary = {}
    for method, data in method_metrics.items():
        summary[method] = {
            'rep_rate_mean': round(float(np.mean(data['rep_rates'])), 4),
            'unique_ratio_mean': round(float(np.mean(data['unique_ratios'])), 4),
            'p1_mean': round(float(np.mean(data['p1_means'])), 4),
            't_mean': round(float(np.mean(data['t_means'])), 4),
        }
        print(f"\n  {method}: rep={summary[method]['rep_rate_mean']:.4f}, "
              f"unique={summary[method]['unique_ratio_mean']:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    methods = list(summary.keys())
    colors = {'greedy': '#2c3e50', 'top_p': '#3498db', 'fixed_t': '#e67e22', 'thermodynamic': '#e74c3c'}

    # (a) Repetition rate comparison
    rep_vals = [summary[m]['rep_rate_mean'] for m in methods]
    bars = axes[0, 0].bar(methods, rep_vals, color=[colors[m] for m in methods])
    axes[0, 0].set_ylabel('Repetition Rate (3-gram)')
    axes[0, 0].set_title('(a) Repetition Rate (lower = better)', fontweight='bold')
    for bar, val in zip(bars, rep_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', fontsize=9)

    # (b) Unique token ratio
    uniq_vals = [summary[m]['unique_ratio_mean'] for m in methods]
    bars = axes[0, 1].bar(methods, uniq_vals, color=[colors[m] for m in methods])
    axes[0, 1].set_ylabel('Unique Token Ratio')
    axes[0, 1].set_title('(b) Diversity (higher = better)', fontweight='bold')
    for bar, val in zip(bars, uniq_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', fontsize=9)

    # (c) P1*T product
    p1t_vals = [summary[m]['p1_mean'] * summary[m]['t_mean'] for m in methods]
    bars = axes[0, 2].bar(methods, p1t_vals, color=[colors[m] for m in methods])
    axes[0, 2].axhline(P1T_CONSTANT, color='red', ls='--', label=f'Target={P1T_CONSTANT}')
    axes[0, 2].set_ylabel('P1 * T')
    axes[0, 2].set_title('(c) P1*T Conservation', fontweight='bold')
    axes[0, 2].legend()

    # (d) Dynamic temperature trace (last prompt)
    if method_metrics['thermodynamic']['t_means']:
        # Run one more for trace visualization
        inp = tok(TEST_PROMPTS[0], return_tensors='pt').to(device)
        _, p1s, ts, t_samps = decode_thermodynamic(model, tok, inp['input_ids'])
        axes[1, 0].plot(t_samps, 'r-', lw=2, label='T_sample (dynamic)')
        axes[1, 0].axhline(0.7, color='orange', ls='--', label='Fixed T=0.7')
        axes[1, 0].set_xlabel('Generation Step')
        axes[1, 0].set_ylabel('Sampling Temperature')
        axes[1, 0].set_title('(d) Dynamic Temperature Trace', fontweight='bold')
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) P1 trace comparison
    inp = tok(TEST_PROMPTS[0], return_tensors='pt').to(device)
    _, p1_g, _ = decode_greedy(model, tok, inp['input_ids'])
    _, p1_t, _, _ = decode_thermodynamic(model, tok, inp['input_ids'])
    axes[1, 1].plot(p1_g[:50], color=colors['greedy'], lw=1.5, label='Greedy')
    axes[1, 1].plot(p1_t[:50], color=colors['thermodynamic'], lw=1.5, label='Thermodynamic')
    axes[1, 1].set_xlabel('Generation Step')
    axes[1, 1].set_ylabel('P1 (max probability)')
    axes[1, 1].set_title('(e) P1 Trajectory Comparison', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary text
    txt = "THERMODYNAMIC DECODING\n"
    txt += f"T = {P1T_CONSTANT} / P1\n\n"
    for m in methods:
        s = summary[m]
        txt += f"{m}:\n"
        txt += f"  Rep: {s['rep_rate_mean']:.3f}  Uniq: {s['unique_ratio_mean']:.3f}\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 269: Thermodynamic Decoding -- P1*T Feedback Sampler",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase269_thermodynamic_decoding')
    plt.close()

    save_results('phase269_thermodynamic_decoding', {
        'experiment': 'Thermodynamic Decoding',
        'P1T_constant': P1T_CONSTANT,
        'n_prompts': len(TEST_PROMPTS),
        'n_repeat': N_REPEAT,
        'max_new_tokens': MAX_NEW_TOKENS,
        'summary': summary,
        'sample_outputs': all_results,
    })

    del model, tok
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
