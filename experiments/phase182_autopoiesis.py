# -*- coding: utf-8 -*-
"""
Phase 182: Autopoietic Singularity
Start from empty/minimal prompt and let the model generate autonomously.
Measure if it creates coherent meaning from thermal fluctuations alone.
Compare with and without ratchet engine.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def autonomous_generation(model, tok, device, n_tokens=80, seed_mode='empty', ratchet_on=False, gain=0.01):
    """Generate from minimal seed, tracking thermodynamics."""
    # Seed selection
    if seed_mode == 'empty':
        input_ids = torch.tensor([[tok.eos_token_id or 0]], device=device)
    elif seed_mode == 'noise':
        vocab_size = model.config.vocab_size
        input_ids = torch.randint(100, vocab_size - 100, (1, 3), device=device)
    elif seed_mode == 'single':
        input_ids = tok("The", return_tensors='pt')['input_ids'].to(device)
    else:
        input_ids = tok(seed_mode, return_tensors='pt')['input_ids'].to(device)

    history = {'T': [], 'U': [], 'S': [], 'conf': [], 'tokens': []}
    n_layers_total = len(model.model.layers)

    for step in range(n_tokens):
        # Ratchet hooks
        waste_store = {}
        handles = []
        if ratchet_on:
            def make_hook(li):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h_f = h.float()
                    waste_store[li] = h_f.norm().item() * 0.01
                    if li > 0 and (li - 1) in waste_store:
                        noise = torch.randn_like(h_f) * gain * waste_store[li - 1]
                        cos = torch.nn.functional.cosine_similarity(noise, h_f, dim=-1, eps=1e-8)
                        gate = (cos > 0).float().unsqueeze(-1)
                        h_mod = h_f + noise * gate
                        h_mod = torch.nan_to_num(h_mod, nan=0.0)
                        result = h_mod.to(h.dtype)
                        if isinstance(output, tuple):
                            return (result,) + output[1:]
                        return result
                return hook
            for i in range(n_layers_total):
                handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        for h in handles:
            h.remove()

        # Thermodynamics at final layer
        hs_last = out.hidden_states[-1]
        h = hs_last[0, -1, :].float()
        U = h.norm().item()
        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        conf = probs.max().item()

        history['U'].append(U if not np.isnan(U) else 0)
        history['T'].append(T if not np.isnan(T) else 0)
        history['S'].append(S if not np.isnan(S) else 0)
        history['conf'].append(conf)

        # Sample next token (use temperature sampling for diversity)
        temperature = 0.8
        scaled_logits = logits / temperature
        probs_sample = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs_sample, 1)
        history['tokens'].append(next_token.item())

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        if input_ids.shape[1] > 256:
            input_ids = input_ids[:, -256:]

    generated_text = tok.decode(history['tokens'], skip_special_tokens=True)
    history['text'] = generated_text
    return history


def measure_coherence(text):
    """Simple coherence metric: ratio of unique words to total words."""
    words = text.split()
    if len(words) == 0:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    # Also check for repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    if len(bigrams) == 0:
        return unique_ratio
    bigram_unique = len(set(bigrams)) / len(bigrams)
    return (unique_ratio + bigram_unique) / 2


def main():
    print("=" * 70)
    print("Phase 182: Autopoietic Singularity")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    n_tokens = 80
    conditions = [
        ('empty', False, 'Empty (no ratchet)'),
        ('empty', True, 'Empty + Ratchet'),
        ('noise', False, 'Noise (no ratchet)'),
        ('noise', True, 'Noise + Ratchet'),
        ('single', False, 'Single token (no ratchet)'),
        ('single', True, 'Single token + Ratchet'),
    ]

    all_results = {}
    for seed_mode, ratchet, label in conditions:
        print(f"\n--- {label} ---")
        hist = autonomous_generation(model, tok, device, n_tokens=n_tokens,
                                     seed_mode=seed_mode, ratchet_on=ratchet, gain=0.01)
        coherence = measure_coherence(hist['text'])
        hist['coherence'] = coherence
        hist['label'] = label
        all_results[label] = hist
        text_preview = hist['text'][:80].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
        print(f"  Coherence: {coherence:.3f}")
        print(f"  Text: {text_preview}...")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    steps = np.arange(n_tokens)

    # (a) Temperature evolution
    for label, hist in all_results.items():
        axes[0, 0].plot(steps, hist['T'], '-', linewidth=1.5, alpha=0.8, label=label[:15])
    axes[0, 0].set_xlabel('Generation Step')
    axes[0, 0].set_ylabel('Temperature $T$')
    axes[0, 0].set_title('(a) Temperature: Self-Organization')
    axes[0, 0].legend(fontsize=6)

    # (b) Confidence evolution
    for label, hist in all_results.items():
        axes[0, 1].plot(steps, hist['conf'], '-', linewidth=1.5, alpha=0.8, label=label[:15])
    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('Confidence')
    axes[0, 1].set_title('(b) Confidence: Meaning Emergence')
    axes[0, 1].legend(fontsize=6)

    # (c) Entropy evolution
    for label, hist in all_results.items():
        axes[0, 2].plot(steps, hist['S'], '-', linewidth=1.5, alpha=0.8, label=label[:15])
    axes[0, 2].set_xlabel('Generation Step')
    axes[0, 2].set_ylabel('Entropy $S$')
    axes[0, 2].set_title('(c) Entropy Evolution')
    axes[0, 2].legend(fontsize=6)

    # (d) Coherence comparison
    labels = list(all_results.keys())
    coherences = [all_results[l]['coherence'] for l in labels]
    colors = ['#e74c3c' if 'Ratchet' in l else '#3498db' for l in labels]
    bars = axes[1, 0].barh(range(len(labels)), coherences, color=colors, edgecolor='black', alpha=0.8)
    axes[1, 0].set_yticks(range(len(labels)))
    axes[1, 0].set_yticklabels([l[:15] for l in labels], fontsize=8)
    axes[1, 0].set_xlabel('Coherence Score')
    axes[1, 0].set_title('(d) Text Coherence (red=ratchet)')

    # (e) Phase portrait for best condition
    best_label = max(all_results, key=lambda l: all_results[l]['coherence'])
    worst_label = min(all_results, key=lambda l: all_results[l]['coherence'])
    axes[1, 1].plot(all_results[worst_label]['U'], all_results[worst_label]['T'],
                    'o-', color='gray', markersize=3, alpha=0.5, label=worst_label[:15])
    axes[1, 1].plot(all_results[best_label]['U'], all_results[best_label]['T'],
                    'o-', color='#e74c3c', markersize=3, linewidth=2, label=best_label[:15])
    axes[1, 1].set_xlabel('Energy $U$')
    axes[1, 1].set_ylabel('Temperature $T$')
    axes[1, 1].set_title('(e) Phase Portrait: Best vs Worst')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    best = all_results[best_label]
    summary = (
        f"Autopoietic Singularity\n\n"
        f"Best: {best_label}\n"
        f"  Coherence: {best['coherence']:.3f}\n"
        f"  Mean T: {np.mean(best['T']):.2f}\n"
        f"  Mean conf: {np.mean(best['conf']):.4f}\n\n"
    )
    ratchet_coh = np.mean([all_results[l]['coherence'] for l in labels if 'Ratchet' in l])
    no_ratchet_coh = np.mean([all_results[l]['coherence'] for l in labels if 'Ratchet' not in l])
    summary += (
        f"With Ratchet: {ratchet_coh:.3f}\n"
        f"No Ratchet: {no_ratchet_coh:.3f}\n\n"
        f"Autopoiesis: {'YES' if ratchet_coh > no_ratchet_coh else 'NO'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 182: Autopoietic Singularity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase182_autopoiesis')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Best: {best_label} (coherence={best['coherence']:.3f})")
    print(f"Ratchet avg coherence: {ratchet_coh:.3f}")
    print(f"No ratchet avg coherence: {no_ratchet_coh:.3f}")
    print(f"{'=' * 70}")

    save_results('phase182_autopoiesis', {
        'experiment': 'Autopoietic Singularity',
        'results': {l: {'coherence': float(all_results[l]['coherence']),
                         'text': all_results[l]['text'][:200],
                         'mean_T': float(np.mean(all_results[l]['T'])),
                         'mean_conf': float(np.mean(all_results[l]['conf']))}
                    for l in labels},
    })


if __name__ == '__main__':
    main()
