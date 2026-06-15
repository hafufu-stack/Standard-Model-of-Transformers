# -*- coding: utf-8 -*-
"""
Phase 211: Anti-Crystallization Generator
==========================================
Phase 204 discovered that autoregressive generation "crystallizes":
eta -> 1.0, output_entropy -> 0, the model locks onto high-confidence
predictions and loses diversity.

Anti-Crystallization: monitor eta during generation and inject
stochastic resonance noise when crystallization is detected,
breaking the crystal and restoring diverse, creative output.

Added by Opus: The crystallization phenomenon is essentially the
thermodynamic death of creativity. This phase engineers a cure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The meaning of life is",
    "Once upon a time in a distant galaxy",
    "The future of artificial intelligence will",
    "In the deepest ocean there exists",
    "A revolutionary scientific discovery revealed that",
]

GEN_LENGTH = 200
CRYSTAL_THRESHOLD = 0.98  # eta above this = crystallized
CRYSTAL_WINDOW = 3        # N consecutive crystallized steps to trigger
NOISE_SIGMA = 0.15        # Stochastic resonance optimal (from Phase 179)


def generate_with_monitoring(model, tok, device, prompt, mode='greedy',
                             anti_crystal=False):
    """Generate tokens while monitoring thermodynamic state.
    
    mode: 'greedy' | 'sample_t08'
    anti_crystal: if True, inject noise when crystallization detected
    """
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']
    generated_ids = input_ids.clone()

    eta_trace = []
    entropy_trace = []
    top1_trace = []
    crystal_events = []
    crystal_count = 0

    with torch.no_grad():
        for step in range(GEN_LENGTH):
            out = model(input_ids=generated_ids, output_hidden_states=True)

            # Measure thermodynamics at last token
            T_list = []
            for hs in out.hidden_states:
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_list.append(T if not np.isnan(T) else 0)

            T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
            T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
            eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0
            eta_trace.append(eta)

            # Check for crystallization
            is_crystallized = False
            if len(eta_trace) >= CRYSTAL_WINDOW:
                recent = eta_trace[-CRYSTAL_WINDOW:]
                if all(e > CRYSTAL_THRESHOLD for e in recent):
                    is_crystallized = True
                    crystal_count += 1

            # Get final logits
            final_logits = out.logits[0, -1, :].float()

            # Anti-crystallization: inject noise into logits when crystallized
            if anti_crystal and is_crystallized:
                noise = torch.randn_like(final_logits) * NOISE_SIGMA * final_logits.std()
                final_logits = final_logits + noise
                crystal_events.append(step)

            final_probs = torch.softmax(final_logits, dim=-1)
            output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
            entropy_trace.append(output_entropy)
            top1_trace.append(final_probs.max().item())

            # Select next token
            if mode == 'greedy':
                next_id = final_logits.argmax().unsqueeze(0).unsqueeze(0)
            elif mode == 'sample_t08':
                probs_t = torch.softmax(final_logits / 0.8, dim=-1)
                next_id = torch.multinomial(probs_t, 1).unsqueeze(0)
            else:
                next_id = final_logits.argmax().unsqueeze(0).unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_id], dim=1)

            # Truncate to prevent OOM for very long sequences
            if generated_ids.shape[1] > 512:
                generated_ids = generated_ids[:, -512:]

    # Decode generated text
    gen_text = tok.decode(generated_ids[0, inp['input_ids'].shape[1]:],
                          skip_special_tokens=True)

    # Diversity metrics
    tokens = gen_text.split()
    unique_1gram = len(set(tokens)) / (len(tokens) + 1e-10)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    unique_2gram = len(set(bigrams)) / (len(bigrams) + 1e-10) if bigrams else 0

    # Repetition detection
    rep_score = 0
    for n in [3, 4, 5]:
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if ngrams:
            rep_score += 1 - len(set(ngrams)) / len(ngrams)
    rep_score /= 3

    return {
        'eta_trace': eta_trace,
        'entropy_trace': entropy_trace,
        'top1_trace': top1_trace,
        'crystal_events': crystal_events,
        'crystal_count': crystal_count,
        'gen_text': gen_text[:500],
        'unique_1gram': unique_1gram,
        'unique_2gram': unique_2gram,
        'rep_score': rep_score,
        'mean_entropy': float(np.mean(entropy_trace)),
        'mean_eta': float(np.mean(eta_trace)),
    }


def main():
    print("=" * 70)
    print("Phase 211: Anti-Crystallization Generator")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    modes = [
        ('greedy', False, 'Greedy (baseline)'),
        ('greedy', True, 'Greedy + Anti-Crystal'),
        ('sample_t08', False, 'Sampling (T=0.8)'),
    ]

    all_results = {}
    example_traces = {}

    for mode, anti_crystal, desc in modes:
        label = f"{mode}_{'ac' if anti_crystal else 'noac'}"
        print(f"\n=== {desc} ===")
        mode_results = []

        for pi, prompt in enumerate(PROMPTS):
            print(f"  Prompt {pi+1}/{len(PROMPTS)}: ", end='', flush=True)
            r = generate_with_monitoring(model, tok, device, prompt,
                                         mode=mode, anti_crystal=anti_crystal)
            mode_results.append(r)
            print(f"entropy={r['mean_entropy']:.3f}, "
                  f"crystals={r['crystal_count']}, "
                  f"diversity={r['unique_1gram']:.3f}")

            if pi == 0:
                example_traces[label] = r

        all_results[label] = {
            'description': desc,
            'mean_entropy': float(np.mean([r['mean_entropy'] for r in mode_results])),
            'mean_eta': float(np.mean([r['mean_eta'] for r in mode_results])),
            'mean_diversity_1gram': float(np.mean([r['unique_1gram'] for r in mode_results])),
            'mean_diversity_2gram': float(np.mean([r['unique_2gram'] for r in mode_results])),
            'mean_rep_score': float(np.mean([r['rep_score'] for r in mode_results])),
            'total_crystals': sum(r['crystal_count'] for r in mode_results),
            'example_text': mode_results[0]['gen_text'][:300],
        }

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    trace_colors = {
        'greedy_noac': '#e74c3c',
        'greedy_ac': '#2ecc71',
        'sample_t08_noac': '#3498db',
    }

    # (a) eta trace comparison
    for label, r in example_traces.items():
        axes[0, 0].plot(r['eta_trace'], '-', color=trace_colors.get(label, 'gray'),
                        lw=1.5, label=label.replace('_', ' '), alpha=0.8)
        if r['crystal_events']:
            axes[0, 0].scatter(r['crystal_events'],
                               [r['eta_trace'][e] for e in r['crystal_events']
                                if e < len(r['eta_trace'])],
                               color='gold', marker='*', s=50, zorder=5)
    axes[0, 0].axhline(y=CRYSTAL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Generation Step')
    axes[0, 0].set_ylabel('eta')
    axes[0, 0].set_title('(a) Efficiency Traces')
    axes[0, 0].legend(fontsize=7)

    # (b) Entropy trace comparison
    for label, r in example_traces.items():
        axes[0, 1].plot(r['entropy_trace'], '-', color=trace_colors.get(label, 'gray'),
                        lw=1.5, label=label.replace('_', ' '), alpha=0.8)
    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('Output Entropy (nats)')
    axes[0, 1].set_title('(b) Entropy Traces')
    axes[0, 1].legend(fontsize=7)

    # (c) Top-1 probability trace
    for label, r in example_traces.items():
        axes[0, 2].plot(r['top1_trace'], '-', color=trace_colors.get(label, 'gray'),
                        lw=1.5, label=label.replace('_', ' '), alpha=0.8)
    axes[0, 2].set_xlabel('Generation Step')
    axes[0, 2].set_ylabel('Top-1 Probability')
    axes[0, 2].set_title('(c) Confidence Traces')
    axes[0, 2].legend(fontsize=7)

    # (d) Diversity comparison bar chart
    methods = [r['description'] for r in all_results.values()]
    div_1gram = [r['mean_diversity_1gram'] for r in all_results.values()]
    div_2gram = [r['mean_diversity_2gram'] for r in all_results.values()]
    x = np.arange(len(methods))
    w = 0.35
    axes[1, 0].bar(x - w/2, div_1gram, w, label='1-gram', color='#e74c3c', alpha=0.7)
    axes[1, 0].bar(x + w/2, div_2gram, w, label='2-gram', color='#3498db', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, fontsize=7, rotation=15)
    axes[1, 0].set_ylabel('Unique N-gram Ratio')
    axes[1, 0].set_title('(d) Text Diversity')
    axes[1, 0].legend(fontsize=8)

    # (e) Repetition score
    rep_scores = [r['mean_rep_score'] for r in all_results.values()]
    axes[1, 1].bar(methods, rep_scores, color='#e67e22', alpha=0.7)
    axes[1, 1].set_ylabel('Repetition Score (lower=better)')
    axes[1, 1].set_title('(e) Repetition')
    axes[1, 1].tick_params(axis='x', labelsize=7, rotation=15)

    # (f) Summary
    greedy = all_results.get('greedy_noac', {})
    ac = all_results.get('greedy_ac', {})
    sampling = all_results.get('sample_t08_noac', {})
    summary_text = (
        f"Anti-Crystallization\n\n"
        f"Greedy baseline:\n"
        f"  diversity = {greedy.get('mean_diversity_1gram', 0):.3f}\n"
        f"  repetition = {greedy.get('mean_rep_score', 0):.3f}\n\n"
        f"Greedy + Anti-Crystal:\n"
        f"  diversity = {ac.get('mean_diversity_1gram', 0):.3f}\n"
        f"  repetition = {ac.get('mean_rep_score', 0):.3f}\n\n"
        f"Sampling (T=0.8):\n"
        f"  diversity = {sampling.get('mean_diversity_1gram', 0):.3f}\n"
        f"  repetition = {sampling.get('mean_rep_score', 0):.3f}\n"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 211: Anti-Crystallization Generator",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase211_anti_crystallization')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Greedy diversity: {greedy.get('mean_diversity_1gram', 0):.3f}")
    print(f"Anti-Crystal diversity: {ac.get('mean_diversity_1gram', 0):.3f}")
    print(f"Sampling diversity: {sampling.get('mean_diversity_1gram', 0):.3f}")
    print(f"{'=' * 70}")

    save_results('phase211_anti_crystallization', {
        'experiment': 'Anti-Crystallization Generator',
        'results': all_results,
    })


if __name__ == '__main__':
    main()
