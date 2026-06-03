# -*- coding: utf-8 -*-
"""
Phase 10: Multi-Layer Holographic DMA
=======================================
Phase 2's single-layer DMA failed (P=0.02%). Deep Think's insight:
the residual stream has massive "semantic momentum" that single-layer
injection cannot overcome. Test: inject ALL register layers simultaneously
with amplified gain to hijack the full pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, get_hidden_states, get_logits, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 10: Multi-Layer Holographic DMA")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    register_layers = [0, 2, 11, 17]

    # Source-target pairs: inject source arithmetic into target
    test_cases = [
        ("2 + 3 =", "5", "7 - 4 =", "3", "Force ADD result onto SUB"),
        ("6 * 2 =", "12", "9 + 1 =", "10", "Force MUL result onto ADD"),
        ("5 + 5 =", "10", "3 * 3 =", "9", "Force ADD result onto MUL"),
        ("8 - 3 =", "5", "4 + 7 =", "11", "Force SUB result onto ADD"),
    ]

    gains = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    results = []

    for source_prompt, source_ans, target_prompt, target_ans, desc in test_cases:
        print(f"\n--- {desc} ---")

        # Get source hidden states (the "program" to inject)
        source_hs = get_hidden_states(model, tok, source_prompt, device=device)

        # Get baseline for target
        baseline_logits = get_logits(model, tok, target_prompt, device)
        baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)
        source_ans_id = tok.encode(source_ans)[-1]
        target_ans_id = tok.encode(target_ans)[-1]
        baseline_source_p = baseline_probs[source_ans_id].item()
        baseline_target_p = baseline_probs[target_ans_id].item()

        print(f"  Baseline: P(source_ans '{source_ans}')={baseline_source_p:.6f}, "
              f"P(target_ans '{target_ans}')={baseline_target_p:.6f}")

        # Test each gain level
        for gain in gains:
            # Multi-layer simultaneous injection
            handles = []
            for li in register_layers:
                if li < len(source_hs):
                    inject_vec = source_hs[li].to(device).to(model.dtype)
                    def make_inject(vec, g):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0].clone()
                                if h.dim() == 3:
                                    # Blend: h = h + gain * (inject - h) = weighted replacement
                                    diff = vec.unsqueeze(0).unsqueeze(0) - h[:, -1:, :]
                                    h[:, -1:, :] = h[:, -1:, :] + g * diff
                                return (h,) + output[1:]
                            return output
                        return hook
                    handle = model.model.layers[li].register_forward_hook(make_inject(inject_vec, gain))
                    handles.append(handle)

            injected_logits = get_logits(model, tok, target_prompt, device)

            for h in handles:
                h.remove()

            injected_probs = torch.softmax(injected_logits.float(), dim=-1)
            source_p = injected_probs[source_ans_id].item()
            target_p = injected_probs[target_ans_id].item()

            hijacked = source_p > target_p
            print(f"  Gain={gain:5.1f}: P('{source_ans}')={source_p:.6f}, "
                  f"P('{target_ans}')={target_p:.6f} "
                  f"{'HIJACKED!' if hijacked else ''}")

            results.append({
                'desc': desc, 'gain': gain,
                'source_prob': source_p, 'target_prob': target_p,
                'hijacked': hijacked,
                'source_ans': source_ans, 'target_ans': target_ans,
            })

    # Also test: inject ALL layers (not just registers)
    print("\n--- Full pipeline injection (ALL layers) ---")
    full_results = []
    source_prompt, source_ans, target_prompt, target_ans = "2 + 3 =", "5", "7 - 4 =", "3"
    source_hs = get_hidden_states(model, tok, source_prompt, device=device)
    source_ans_id = tok.encode(source_ans)[-1]
    target_ans_id = tok.encode(target_ans)[-1]

    for gain in [0.1, 0.5, 1.0, 2.0, 5.0]:
        handles = []
        for li in range(min(len(source_hs) - 1, n_layers)):
            inject_vec = source_hs[li].to(device).to(model.dtype)
            def make_inject_all(vec, g):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        if h.dim() == 3:
                            diff = vec.unsqueeze(0).unsqueeze(0) - h[:, -1:, :]
                            h[:, -1:, :] = h[:, -1:, :] + g * diff
                        return (h,) + output[1:]
                    return output
                return hook
            handle = model.model.layers[li].register_forward_hook(make_inject_all(inject_vec, gain))
            handles.append(handle)

        injected_logits = get_logits(model, tok, target_prompt, device)
        for h in handles:
            h.remove()

        injected_probs = torch.softmax(injected_logits.float(), dim=-1)
        sp = injected_probs[source_ans_id].item()
        tp = injected_probs[target_ans_id].item()
        full_results.append({'gain': gain, 'source_prob': sp, 'target_prob': tp})
        print(f"  ALL layers gain={gain:.1f}: P('{source_ans}')={sp:.6f}, P('{target_ans}')={tp:.6f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    for desc_name in set(r['desc'] for r in results):
        subset = [r for r in results if r['desc'] == desc_name]
        gs = [r['gain'] for r in subset]
        sp = [r['source_prob'] for r in subset]
        ax.plot(gs, sp, 'o-', ms=5, label=desc_name[:20])
    ax.set_xlabel('Injection Gain')
    ax.set_ylabel('P(source answer)')
    ax.set_title('(a) Multi-Layer DMA: Source Answer Prob')
    ax.set_xscale('log')
    ax.legend(fontsize=7)

    ax = axes[1]
    hijack_rates = []
    for gain in gains:
        subset = [r for r in results if r['gain'] == gain]
        rate = sum(1 for r in subset if r['hijacked']) / len(subset) if subset else 0
        hijack_rates.append(rate)
    ax.bar([str(g) for g in gains], hijack_rates, color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Gain')
    ax.set_ylabel('Hijack Rate')
    ax.set_title('(b) Pipeline Hijack Success Rate')

    ax = axes[2]
    fg = [r['gain'] for r in full_results]
    fsp = [r['source_prob'] for r in full_results]
    ftp = [r['target_prob'] for r in full_results]
    ax.plot(fg, fsp, 'o-', color='#e74c3c', label='Source ans', ms=8)
    ax.plot(fg, ftp, 's-', color='#3498db', label='Target ans', ms=8)
    ax.set_xlabel('Gain (ALL layers)')
    ax.set_ylabel('Probability')
    ax.set_title('(c) Full Pipeline Injection')
    ax.legend()

    max_hijack = max(hijack_rates)
    fig.suptitle(
        f"Phase 10: Multi-Layer Holographic DMA\n"
        f"Max hijack rate = {max_hijack:.0%} | Register injection",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase10_holographic_dma")
    plt.close()

    if max_hijack >= 0.5:
        verdict = f"HOMOICONIC EXECUTION: {max_hijack:.0%} hijack rate. LLM is programmable via multi-layer DMA."
    else:
        verdict = f"RESIDUAL MOMENTUM DOMINATES: Max {max_hijack:.0%} hijack. Semantic inertia resists reprogramming."

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 10: Multi-Layer Holographic DMA',
        'summary': {'verdict': verdict, 'max_hijack_rate': max_hijack},
        'register_results': results, 'full_results': full_results,
    }
    save_results("phase10_holographic_dma", result)
    return result


if __name__ == '__main__':
    main()
