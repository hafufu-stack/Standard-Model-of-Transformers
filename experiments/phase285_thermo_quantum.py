# -*- coding: utf-8 -*-
"""
Phase 285: Thermodynamic Decoding x Quantum Advantage
======================================================
S-Qubit Q295 found quantum advantage scores (avg 2202x, 3 infinite).
Test: does Thermodynamic Decoding produce outputs that score higher
on quantum-information metrics vs conventional decoding?
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
MAX_NEW_TOKENS = 60


def decode_thermodynamic(model, tok, input_ids, constant=P1T_CONSTANT):
    """Thermodynamic decoding: T_sample = constant / P1."""
    generated = input_ids.clone()
    p1_trace, entropy_trace = [], []
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(generated)
        raw_logits = out.logits[0, -1, :].float()
        raw_probs = torch.softmax(raw_logits, dim=-1)
        p1 = raw_probs.max().item()
        entropy = -(raw_probs * torch.log(raw_probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        entropy_trace.append(entropy)

        t_sample = constant / max(p1, 0.01)
        t_sample = max(0.1, min(t_sample, 5.0))
        scaled_logits = raw_logits / t_sample
        probs = torch.softmax(scaled_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return generated, p1_trace, entropy_trace


def decode_top_p(model, tok, input_ids, top_p=0.9):
    """Standard top-p sampling."""
    generated = input_ids.clone()
    p1_trace, entropy_trace = [], []
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(generated)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        p1_trace.append(p1)
        entropy_trace.append(entropy)

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
    return generated, p1_trace, entropy_trace


def quantum_metrics(gen_ids, p1_trace, entropy_trace):
    """Compute quantum-inspired metrics on generated text."""
    n = len(gen_ids)
    if n < 3:
        return {'superposition': 0, 'entanglement': 0, 'coherence': 0, 'advantage': 0}

    # Superposition: how many tokens have p1 < 0.5 (multiple states active)
    superposition = sum(1 for p in p1_trace if p < 0.5) / len(p1_trace) if p1_trace else 0

    # Entanglement: token-token correlation (bigram predictability)
    bigrams = [tuple(gen_ids[i:i+2]) for i in range(n-1)]
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(gen_ids)
    # Mutual information proxy
    mi = 0.0
    for (a, b), count_ab in bigram_counts.items():
        p_ab = count_ab / max(n-1, 1)
        p_a = unigram_counts[a] / n
        p_b = unigram_counts[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
    entanglement = float(mi)

    # Coherence: entropy stability (low variance = coherent)
    coherence = 1.0 / (np.std(entropy_trace) + 1e-3) if entropy_trace else 0

    # Advantage: superposition * diversity (unique ratio)
    unique_ratio = len(set(gen_ids)) / max(n, 1)
    advantage = superposition * unique_ratio * 100

    return {
        'superposition': round(superposition, 4),
        'entanglement': round(entanglement, 4),
        'coherence': round(coherence, 4),
        'advantage': round(advantage, 4),
        'unique_ratio': round(unique_ratio, 4),
    }


PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "The most effective approach to solving climate change is",
    "Once upon a time in a kingdom far away",
    "Machine learning algorithms can classify data by",
]


def main():
    print("=" * 70)
    print("Phase 285: Thermodynamic Decoding x Quantum Advantage")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    all_results = {'thermodynamic': [], 'top_p': []}

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {pi+1}: {prompt[:40]}... ---")
        inp = tok(prompt, return_tensors='pt').to(device)
        input_len = inp['input_ids'].shape[1]

        for trial in range(3):
            # Thermodynamic
            gen_t, p1_t, ent_t = decode_thermodynamic(model, tok, inp['input_ids'])
            gen_ids_t = gen_t[0, input_len:].tolist()
            qm_t = quantum_metrics(gen_ids_t, p1_t, ent_t)
            qm_t['method'] = 'thermodynamic'
            qm_t['text'] = tok.decode(gen_ids_t, skip_special_tokens=True)[:100]
            all_results['thermodynamic'].append(qm_t)

            # Top-p
            gen_p, p1_p, ent_p = decode_top_p(model, tok, inp['input_ids'])
            gen_ids_p = gen_p[0, input_len:].tolist()
            qm_p = quantum_metrics(gen_ids_p, p1_p, ent_p)
            qm_p['method'] = 'top_p'
            qm_p['text'] = tok.decode(gen_ids_p, skip_special_tokens=True)[:100]
            all_results['top_p'].append(qm_p)

        print(f"  Thermo advantage: {np.mean([r['advantage'] for r in all_results['thermodynamic'][-3:]]):.2f}")
        print(f"  Top-p advantage:  {np.mean([r['advantage'] for r in all_results['top_p'][-3:]]):.2f}")

    # Summary
    summary = {}
    for method in ['thermodynamic', 'top_p']:
        data = all_results[method]
        summary[method] = {
            'avg_superposition': round(float(np.mean([d['superposition'] for d in data])), 4),
            'avg_entanglement': round(float(np.mean([d['entanglement'] for d in data])), 4),
            'avg_coherence': round(float(np.mean([d['coherence'] for d in data])), 4),
            'avg_advantage': round(float(np.mean([d['advantage'] for d in data])), 4),
            'avg_unique_ratio': round(float(np.mean([d['unique_ratio'] for d in data])), 4),
        }

    ratio = summary['thermodynamic']['avg_advantage'] / max(summary['top_p']['avg_advantage'], 1e-10)
    print(f"\n  Thermodynamic/Top-p advantage ratio: {ratio:.2f}x")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_method = {'thermodynamic': '#e74c3c', 'top_p': '#3498db'}

    metrics = ['superposition', 'entanglement', 'coherence', 'advantage', 'unique_ratio']
    titles = ['Superposition', 'Entanglement', 'Coherence', 'Advantage', 'Unique Ratio']

    for i, (metric, title) in enumerate(zip(metrics[:3], titles[:3])):
        ax = axes[0, i]
        for method in ['thermodynamic', 'top_p']:
            vals = [d[metric] for d in all_results[method]]
            ax.hist(vals, bins=10, alpha=0.6, color=colors_method[method], label=method)
        ax.set_xlabel(metric)
        ax.set_title(f'({chr(97+i)}) {title}', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)

    # (d) Advantage comparison
    for method in ['thermodynamic', 'top_p']:
        vals = [d['advantage'] for d in all_results[method]]
        axes[1, 0].hist(vals, bins=10, alpha=0.6, color=colors_method[method], label=method)
    axes[1, 0].set_xlabel('Advantage Score')
    axes[1, 0].set_title('(d) Quantum Advantage', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Spider/radar comparison
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for method in ['thermodynamic', 'top_p']:
        vals = [summary[method][f'avg_{m}'] for m in metrics]
        # Normalize
        max_vals = [max(summary['thermodynamic'][f'avg_{m}'],
                       summary['top_p'][f'avg_{m}'], 1e-10) for m in metrics]
        vals_norm = [v/mv for v, mv in zip(vals, max_vals)]
        vals_norm += vals_norm[:1]
        axes[1, 1].plot(angles, vals_norm, '-o', color=colors_method[method],
                       label=method, lw=2)
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels([m[:5] for m in metrics], fontsize=8)
    axes[1, 1].set_title('(e) Quantum Metrics Radar', fontweight='bold')
    axes[1, 1].legend()

    # (f) Summary
    txt = "THERMO DECODING x QUANTUM ADVANTAGE\n\n"
    for method, s in summary.items():
        txt += f"{method}:\n"
        txt += f"  Superpos: {s['avg_superposition']:.3f}\n"
        txt += f"  Entangle: {s['avg_entanglement']:.3f}\n"
        txt += f"  Coherent: {s['avg_coherence']:.1f}\n"
        txt += f"  Advantage: {s['avg_advantage']:.1f}\n\n"
    txt += f"Thermo/Top-p ratio: {ratio:.2f}x"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 285: Thermodynamic Decoding x Quantum Advantage",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase285_thermo_quantum')
    plt.close()

    save_results('phase285_thermo_quantum', {
        'experiment': 'Thermodynamic Decoding x Quantum Advantage',
        'summary': summary,
        'advantage_ratio': round(ratio, 4),
        'results': all_results,
    })

    del model, tok
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
