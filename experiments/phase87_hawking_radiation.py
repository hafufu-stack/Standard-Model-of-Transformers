# -*- coding: utf-8 -*-
"""
Phase 87: Hawking Radiation at Collapse Boundary
Observe whether T spikes (Hawking radiation) occur just before black hole
collapse (T->0), and test if noise injection can rescue collapsed prompts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

COLLAPSE_PROMPTS = [
    "The history of mathematics spans thousands of years",
    "Quantum computing promises to revolutionize information",
    "The Amazon rainforest contains the greatest biodiversity",
    "The number one is followed by one which is followed by",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
    "This sentence is a sentence that is a sentence that is",
]


def iterative_feed(model, tok, device, prompt, n_iters=100):
    """Feed output back as input iteratively, tracking T at each step."""
    t_trace = []
    current_text = prompt

    for i in range(n_iters):
        inp = tok(current_text, return_tensors='pt', truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(T):
            T = 0.0
        t_trace.append(T)

        # Generate next token
        next_id = torch.argmax(logits).item()
        next_tok = tok.decode([next_id])
        current_text = current_text + next_tok

    return t_trace


def detect_hawking_radiation(t_trace, window=5):
    """Detect T spikes (radiation) before collapse regions."""
    spikes = []
    for i in range(window, len(t_trace) - 1):
        local_mean = np.mean(t_trace[max(0, i-window):i])
        # Spike = T suddenly increases before dropping to near-zero
        if t_trace[i] > local_mean * 2.0 and t_trace[i] > 0.5:
            # Check if followed by collapse (T -> small)
            future = t_trace[i+1:min(i+6, len(t_trace))]
            if future and min(future) < 0.1:
                spikes.append({
                    'iteration': i,
                    'T_spike': float(t_trace[i]),
                    'T_before': float(local_mean),
                    'T_after_min': float(min(future)),
                    'ratio': float(t_trace[i] / (local_mean + 1e-10)),
                })
    return spikes


def noise_rescue_test(model, tok, device, prompt, noise_layer=20, sigma=0.5, n_iters=50):
    """Try to rescue a collapsed prompt by injecting noise at a specific layer."""
    # First, get collapse trajectory (no noise)
    t_baseline = iterative_feed(model, tok, device, prompt, n_iters=n_iters)

    # Then with noise injection
    hook_handle = None

    def noise_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h_fp32 = h.to(torch.float32)
        noise = torch.randn_like(h_fp32) * sigma
        h_mod = h_fp32 + noise
        h_mod = torch.nan_to_num(h_mod, nan=0.0, posinf=65000.0, neginf=-65000.0)
        result = h_mod.to(h.dtype)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    # Install hook
    if noise_layer < len(model.model.layers):
        hook_handle = model.model.layers[noise_layer].register_forward_hook(noise_hook)

    t_noisy = iterative_feed(model, tok, device, prompt, n_iters=n_iters)

    if hook_handle:
        hook_handle.remove()

    return t_baseline, t_noisy


def main():
    print("=" * 70)
    print("Phase 87: Hawking Radiation at Collapse Boundary")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    results = []
    all_spikes = []

    for prompt in COLLAPSE_PROMPTS:
        print(f"  Testing: {prompt[:40]}...")
        t_trace = iterative_feed(model, tok, device, prompt, n_iters=100)

        # Detect collapse
        collapsed = any(t < 0.01 for t in t_trace[-20:])

        # Detect Hawking radiation spikes
        spikes = detect_hawking_radiation(t_trace)

        # Noise rescue test (shorter)
        t_base, t_noisy = noise_rescue_test(model, tok, device, prompt,
                                             noise_layer=20, sigma=0.3, n_iters=30)
        base_final_T = np.mean(t_base[-5:]) if t_base else 0
        noisy_final_T = np.mean(t_noisy[-5:]) if t_noisy else 0
        rescue_improvement = float(noisy_final_T - base_final_T)

        results.append({
            'prompt': prompt[:50],
            't_trace': [float(t) for t in t_trace],
            'collapsed': collapsed,
            'n_spikes': len(spikes),
            'spikes': spikes,
            'rescue_improvement': rescue_improvement,
            'base_final_T': float(base_final_T),
            'noisy_final_T': float(noisy_final_T),
        })
        all_spikes.extend(spikes)

        status = 'COLLAPSE' if collapsed else 'STABLE'
        print(f"    {status}, {len(spikes)} Hawking spikes, "
              f"rescue: {'+' if rescue_improvement > 0 else ''}{rescue_improvement:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) T traces with spike markers
    styles = [
        {'color': '#c0392b', 'ls': '-'},
        {'color': '#2980b9', 'ls': '--'},
        {'color': '#27ae60', 'ls': '-.'},
        {'color': '#2c3e50', 'ls': ':'},
        {'color': '#8e44ad', 'ls': '-'},
        {'color': '#e67e22', 'ls': '--'},
    ]
    for i, r in enumerate(results):
        s = styles[i % len(styles)]
        axes[0].plot(r['t_trace'], color=s['color'], linestyle=s['ls'],
                     alpha=0.7, linewidth=1.5, label=r['prompt'][:20] + '...')
        # Mark spikes
        for spike in r['spikes']:
            axes[0].scatter(spike['iteration'], spike['T_spike'],
                           s=80, c='gold', edgecolors='black', zorder=5, marker='*')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Temperature $T$')
    axes[0].set_title('(a) Collapse + Hawking Spikes')
    axes[0].legend(fontsize=6, loc='upper right')

    # (b) Spike statistics
    if all_spikes:
        spike_ratios = [s['ratio'] for s in all_spikes]
        axes[1].hist(spike_ratios, bins=15, color='#f39c12', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Spike Ratio ($T_{spike}/T_{local}$)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'(b) Hawking Spike Distribution (n={len(all_spikes)})')
    else:
        axes[1].text(0.5, 0.5, 'No Hawking\nspikes detected', ha='center', va='center',
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('(b) No Hawking Radiation')

    # (c) Noise rescue results
    prompts_short = [r['prompt'][:15] for r in results]
    improvements = [r['rescue_improvement'] for r in results]
    colors_bar = ['#27ae60' if imp > 0 else '#c0392b' for imp in improvements]
    axes[2].bar(range(len(results)), improvements, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[2].set_xticks(range(len(results)))
    axes[2].set_xticklabels(prompts_short, rotation=45, ha='right', fontsize=7)
    axes[2].axhline(y=0, color='black', linewidth=1)
    axes[2].set_ylabel('$\\Delta T$ (noisy - baseline)')
    axes[2].set_title('(c) Noise Rescue Attempt')

    n_collapsed = sum(1 for r in results if r['collapsed'])
    fig.suptitle(f'Phase 87: Hawking Radiation ({len(all_spikes)} spikes, '
                 f'{n_collapsed}/{len(results)} collapsed)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase87_hawking_radiation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Total Hawking spikes: {len(all_spikes)}")
    print(f"Collapsed prompts: {n_collapsed}/{len(results)}")
    rescued = sum(1 for r in results if r['rescue_improvement'] > 0.1)
    print(f"Successfully rescued: {rescued}/{len(results)}")
    print(f"{'='*70}")

    save_results('phase87_hawking_radiation', {
        'experiment': 'Hawking Radiation at Collapse Boundary',
        'results': results,
        'summary': {
            'total_spikes': len(all_spikes),
            'n_collapsed': n_collapsed,
            'n_rescued': rescued,
            'mean_rescue_delta': float(np.mean(improvements)),
        }
    })


if __name__ == '__main__':
    main()
