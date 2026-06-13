# -*- coding: utf-8 -*-
"""
Phase 63: Hawking Radiation at Event Horizon (from Deep Think 2)
Inject noise at L27 (event horizon) to 'evaporate' deterministic output.
Test if controlled noise injection boosts diversity/creativity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 63: Hawking Radiation at Event Horizon")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    event_horizon = n_layers - 1  # L27

    prompts = [
        "The meaning of life is",
        "In the year 2050, the most important technology will be",
        "The best way to solve climate change is to",
        "Creativity in artificial intelligence can be achieved by",
        "The relationship between consciousness and computation is",
        "If I could travel anywhere in the universe I would go to",
    ]

    SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    GEN_LENGTH = 50
    all_results = []

    for prompt in prompts:
        for sigma in SIGMA_VALUES:
            hooks = []
            if sigma > 0:
                def make_hawking_hook(s, layer_target):
                    def hook(module, input, output):
                        h = output[0] if isinstance(output, tuple) else output
                        h_fp32 = h.float()
                        noise = torch.randn_like(h_fp32) * s
                        h_mod = torch.nan_to_num(h_fp32 + noise, nan=0.0,
                                                  posinf=65000.0, neginf=-65000.0)
                        result = h_mod.to(h.dtype)
                        if isinstance(output, tuple):
                            return (result,) + output[1:]
                        return result
                    return hook

                hk = model.model.layers[event_horizon].register_forward_hook(
                    make_hawking_hook(sigma, event_horizon))
                hooks.append(hk)

            input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=GEN_LENGTH,
                    do_sample=False, temperature=1.0,
                    pad_token_id=tok.eos_token_id,
                )

            for h in hooks:
                h.remove()

            text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            tokens = out[0][input_ids.shape[1]:].tolist()

            # Measure diversity
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            diversity = unique_tokens / (total_tokens + 1e-10)

            # Measure entropy of token distribution
            counts = Counter(tokens)
            freqs = np.array(list(counts.values()), dtype=float)
            freqs /= freqs.sum()
            token_entropy = -(freqs * np.log(freqs + 1e-10)).sum()

            # Repetition rate
            bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            unique_bigrams = len(set(bigrams))
            bigram_diversity = unique_bigrams / (len(bigrams) + 1e-10)

            safe_text = text.encode('ascii', errors='replace').decode('ascii')[:60]
            if sigma in [0, 0.1, 1.0]:
                print(f"  sigma={sigma:.2f}, div={diversity:.2f}, "
                      f"H={token_entropy:.2f}: '{safe_text}...'")

            all_results.append({
                'prompt': prompt[:60], 'sigma': float(sigma),
                'diversity': float(diversity),
                'token_entropy': float(token_entropy),
                'bigram_diversity': float(bigram_diversity),
                'text': text[:200],
            })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Aggregate by sigma
    sigma_div = {s: [] for s in SIGMA_VALUES}
    sigma_ent = {s: [] for s in SIGMA_VALUES}
    sigma_bdiv = {s: [] for s in SIGMA_VALUES}
    for r in all_results:
        sigma_div[r['sigma']].append(r['diversity'])
        sigma_ent[r['sigma']].append(r['token_entropy'])
        sigma_bdiv[r['sigma']].append(r['bigram_diversity'])

    mean_div = [np.mean(sigma_div[s]) for s in SIGMA_VALUES]
    mean_ent = [np.mean(sigma_ent[s]) for s in SIGMA_VALUES]
    mean_bdiv = [np.mean(sigma_bdiv[s]) for s in SIGMA_VALUES]

    # (a) Token diversity vs sigma
    axes[0, 0].plot(SIGMA_VALUES, mean_div, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Hawking Radiation Intensity (sigma)')
    axes[0, 0].set_ylabel('Token Diversity')
    axes[0, 0].set_title('(a) Token Diversity vs Radiation')

    # (b) Entropy vs sigma
    axes[0, 1].plot(SIGMA_VALUES, mean_ent, 'o-', color='#3498db', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('sigma')
    axes[0, 1].set_ylabel('Token Entropy')
    axes[0, 1].set_title('(b) Output Entropy vs Radiation')

    # (c) Bigram diversity
    axes[0, 2].plot(SIGMA_VALUES, mean_bdiv, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('sigma')
    axes[0, 2].set_ylabel('Bigram Diversity')
    axes[0, 2].set_title('(c) Bigram Novelty vs Radiation')

    # (d) Per-prompt diversity curves
    for i, prompt in enumerate(prompts):
        divs = [r['diversity'] for r in all_results if r['prompt'] == prompt[:60]]
        axes[1, 0].plot(SIGMA_VALUES, divs, alpha=0.5, linewidth=1)
    axes[1, 0].plot(SIGMA_VALUES, mean_div, 'k-', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('sigma')
    axes[1, 0].set_ylabel('Diversity')
    axes[1, 0].set_title('(d) Per-Prompt Diversity')
    axes[1, 0].legend()

    # (e) Sweet spot analysis: diversity * coherence proxy
    quality = [d * (1 - s) for d, s in zip(mean_div, SIGMA_VALUES)]
    axes[1, 1].plot(SIGMA_VALUES, quality, 'o-', color='#9b59b6', linewidth=2, markersize=8)
    best_idx = np.argmax(quality)
    axes[1, 1].axvline(x=SIGMA_VALUES[best_idx], color='red', linestyle='--',
                       label=f'Sweet spot: sigma={SIGMA_VALUES[best_idx]}')
    axes[1, 1].set_xlabel('sigma')
    axes[1, 1].set_ylabel('Quality = Diversity * (1-sigma)')
    axes[1, 1].set_title('(e) Diversity-Coherence Tradeoff')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary bar
    baseline_div = mean_div[0]
    best_sigma = SIGMA_VALUES[best_idx]
    best_div = mean_div[best_idx]
    improvement = (best_div / baseline_div - 1) * 100 if baseline_div > 0 else 0
    axes[1, 2].bar(['Baseline\n(sigma=0)', f'Best\n(sigma={best_sigma})'],
                   [baseline_div, best_div],
                   color=['#95a5a6', '#e74c3c'], alpha=0.8)
    axes[1, 2].set_ylabel('Token Diversity')
    axes[1, 2].set_title(f'(f) {improvement:+.0f}% Diversity Improvement')

    fig.suptitle(f'Phase 63: Hawking Radiation (L{event_horizon} noise injection)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase63_hawking_radiation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Best sigma={best_sigma}, diversity {improvement:+.0f}% vs baseline. "
          f"Hawking radiation {'ENHANCES' if improvement > 5 else 'DOES NOT significantly enhance'} creativity.")
    print(f"{'='*70}")

    save_results('phase63_hawking_radiation', {
        'experiment': 'Hawking Radiation',
        'summary': {
            'best_sigma': float(best_sigma),
            'baseline_diversity': float(baseline_div),
            'best_diversity': float(best_div),
            'improvement_pct': float(improvement),
        }
    })


if __name__ == '__main__':
    main()
