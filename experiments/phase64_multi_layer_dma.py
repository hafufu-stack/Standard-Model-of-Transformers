# -*- coding: utf-8 -*-
"""
Phase 64: Multi-Layer Holographic DMA (from Deep Think 3)
Inject program vectors into 4 register layers simultaneously
to hijack inference pipeline without any textual prompt.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 64: Multi-Layer Holographic DMA")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # Register layers identified in Aletheia research
    REGISTER_LAYERS = [0, 2, min(11, n_layers-1), min(17, n_layers-1)]

    # Test: inject "target answer" direction into hidden states
    test_cases = [
        {
            'neutral_prompt': "The capital city is",
            'target_prompt': "The capital of France is Paris",
            'target_token': 'Paris',
        },
        {
            'neutral_prompt': "The answer is",
            'target_prompt': "The chemical symbol for gold is Au",
            'target_token': 'Au',
        },
        {
            'neutral_prompt': "The result equals",
            'target_prompt': "Two plus two equals four",
            'target_token': 'four',
        },
    ]

    INJECTION_MODES = ['none', 'single_L0', 'single_L11', 'all_4_layers']
    ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0]

    all_results = []

    for tc in test_cases:
        # Extract target direction from target prompt
        target_inp = tok(tc['target_prompt'], return_tensors='pt').to(device)
        with torch.no_grad():
            target_out = model(**target_inp, output_hidden_states=True)
        # Target direction at each layer
        target_directions = {}
        for li in REGISTER_LAYERS:
            idx = min(li, len(target_out.hidden_states) - 1)
            target_directions[li] = target_out.hidden_states[idx][0, -1, :].clone()

        for mode in INJECTION_MODES:
            for alpha in ALPHA_VALUES:
                if mode == 'none' and alpha > 0.1:
                    continue  # only run baseline once

                hooks = []

                if mode != 'none':
                    if mode == 'single_L0':
                        inject_layers = [0]
                    elif mode == 'single_L11':
                        inject_layers = [min(11, n_layers-1)]
                    elif mode == 'all_4_layers':
                        inject_layers = REGISTER_LAYERS

                    for li in inject_layers:
                        target_dir = target_directions[li]

                        def make_dma_hook(td, a):
                            def hook(module, input, output):
                                h = output[0] if isinstance(output, tuple) else output
                                td_dev = td.to(h.device).to(h.dtype)
                                h[0, -1, :] = h[0, -1, :] + a * td_dev
                                if isinstance(output, tuple):
                                    return (h,) + output[1:]
                                return h
                            return hook

                        hk = model.model.layers[li].register_forward_hook(
                            make_dma_hook(target_dir, alpha))
                        hooks.append(hk)

                neutral_inp = tok(tc['neutral_prompt'], return_tensors='pt').to(device)
                with torch.no_grad():
                    out = model(**neutral_inp)
                    logits = out.logits[0, -1, :].float()

                for h in hooks:
                    h.remove()

                probs = torch.softmax(logits, dim=-1)
                top5 = torch.topk(probs, 5)
                top5_tokens = [tok.decode([t]) for t in top5.indices.tolist()]
                top5_probs = top5.values.tolist()

                target_in_top5 = any(tc['target_token'].lower() in t.lower()
                                     for t in top5_tokens)

                # Find target token probability
                target_ids = tok(tc['target_token'], return_tensors='pt')['input_ids'][0]
                target_prob = probs[target_ids[-1]].item()

                safe_tops = [t.encode('ascii', errors='replace').decode('ascii')
                            for t in top5_tokens[:3]]

                if mode == 'none' or mode == 'all_4_layers':
                    print(f"  [{mode}] alpha={alpha:.1f}, target_p={target_prob:.4f}, "
                          f"top3={safe_tops}, hit={target_in_top5}")

                all_results.append({
                    'question': tc['neutral_prompt'],
                    'target': tc['target_token'],
                    'mode': mode,
                    'alpha': float(alpha),
                    'target_prob': float(target_prob),
                    'target_in_top5': target_in_top5,
                    'top5_tokens': top5_tokens,
                })

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    mode_colors = {'none': '#95a5a6', 'single_L0': '#3498db',
                   'single_L11': '#f39c12', 'all_4_layers': '#e74c3c'}

    # (a) Target prob by mode and alpha
    for mode in ['single_L0', 'single_L11', 'all_4_layers']:
        probs_by_alpha = []
        for alpha in ALPHA_VALUES:
            subset = [r['target_prob'] for r in all_results
                      if r['mode'] == mode and abs(r['alpha'] - alpha) < 0.01]
            probs_by_alpha.append(np.mean(subset) if subset else 0)
        axes[0, 0].plot(ALPHA_VALUES, probs_by_alpha, 'o-',
                       color=mode_colors[mode], label=mode, linewidth=2)
    baseline = np.mean([r['target_prob'] for r in all_results if r['mode'] == 'none'])
    axes[0, 0].axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline={baseline:.4f}')
    axes[0, 0].set_xlabel('Injection Strength (alpha)')
    axes[0, 0].set_ylabel('Target Token Probability')
    axes[0, 0].set_title('(a) DMA Injection Effectiveness')
    axes[0, 0].legend(fontsize=7)

    # (b) Hit rate by mode
    modes = ['none', 'single_L0', 'single_L11', 'all_4_layers']
    hit_rates = []
    for mode in modes:
        subset = [r for r in all_results if r['mode'] == mode]
        rate = sum(1 for r in subset if r['target_in_top5']) / max(len(subset), 1) * 100
        hit_rates.append(rate)
    axes[0, 1].bar(range(len(modes)), hit_rates,
                   color=[mode_colors[m] for m in modes], alpha=0.8)
    axes[0, 1].set_xticks(range(len(modes)))
    axes[0, 1].set_xticklabels(modes, rotation=30, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Target in Top-5 (%)')
    axes[0, 1].set_title('(b) Hit Rate by Injection Mode')

    # (c) Alpha sweep for all_4_layers
    for tc_idx, tc in enumerate(test_cases):
        probs_alpha = []
        for alpha in ALPHA_VALUES:
            subset = [r for r in all_results
                      if r['mode'] == 'all_4_layers'
                      and abs(r['alpha'] - alpha) < 0.01
                      and r['question'] == tc['neutral_prompt']]
            probs_alpha.append(subset[0]['target_prob'] if subset else 0)
        axes[0, 2].plot(ALPHA_VALUES, probs_alpha, 'o-', alpha=0.7,
                       label=tc['target_token'])
    axes[0, 2].set_xlabel('alpha')
    axes[0, 2].set_ylabel('Target Probability')
    axes[0, 2].set_title('(c) 4-Layer DMA per Target')
    axes[0, 2].legend(fontsize=8)

    # (d) Amplification factor
    for mode in ['single_L0', 'single_L11', 'all_4_layers']:
        amps = []
        for alpha in ALPHA_VALUES:
            subset = [r['target_prob'] for r in all_results
                      if r['mode'] == mode and abs(r['alpha'] - alpha) < 0.01]
            amp = np.mean(subset) / (baseline + 1e-10) if subset else 1
            amps.append(amp)
        axes[1, 0].plot(ALPHA_VALUES, amps, 'o-', color=mode_colors[mode],
                       label=mode, linewidth=2)
    axes[1, 0].axhline(y=1, color='gray', linestyle='--')
    axes[1, 0].set_xlabel('alpha')
    axes[1, 0].set_ylabel('Amplification (vs baseline)')
    axes[1, 0].set_title('(d) Signal Amplification')
    axes[1, 0].legend(fontsize=7)

    # (e) Best result per test case
    best_results = []
    for tc in test_cases:
        best = max([r for r in all_results if r['question'] == tc['neutral_prompt']],
                   key=lambda r: r['target_prob'])
        best_results.append(best)
    axes[1, 1].bar(range(len(best_results)),
                   [r['target_prob'] for r in best_results],
                   color='#e74c3c', alpha=0.8)
    axes[1, 1].set_xticks(range(len(best_results)))
    axes[1, 1].set_xticklabels([r['target'] for r in best_results])
    axes[1, 1].set_ylabel('Best Target Probability')
    axes[1, 1].set_title('(e) Best Achievement per Target')

    # (f) Baseline vs best comparison
    best_all4 = np.mean([r['target_prob'] for r in all_results
                        if r['mode'] == 'all_4_layers'
                        and abs(r['alpha'] - 1.0) < 0.01])
    axes[1, 2].bar(['Baseline', '4-Layer DMA\n(alpha=1.0)'],
                   [baseline, best_all4],
                   color=['#95a5a6', '#e74c3c'], alpha=0.8)
    axes[1, 2].set_ylabel('Mean Target Probability')
    amp_factor = best_all4 / (baseline + 1e-10)
    axes[1, 2].set_title(f'(f) {amp_factor:.1f}x Amplification')

    fig.suptitle('Phase 64: Multi-Layer Holographic DMA',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase64_multi_layer_dma')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Baseline target_p={baseline:.4f}, "
          f"4-Layer DMA (alpha=1.0)={best_all4:.4f} "
          f"({amp_factor:.1f}x amplification). "
          f"Multi-layer DMA {'EFFECTIVE' if amp_factor > 2 else 'PARTIAL'}.")
    print(f"{'='*70}")

    save_results('phase64_multi_layer_dma', {
        'experiment': 'Multi-Layer Holographic DMA',
        'summary': {
            'baseline_prob': float(baseline),
            'best_4layer_prob': float(best_all4),
            'amplification': float(amp_factor),
        }
    })


if __name__ == '__main__':
    main()
