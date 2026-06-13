# -*- coding: utf-8 -*-
"""
Phase 88b: Thermodynamic Token Routing (Fixed)
Fixed evaluation: use generation quality (perplexity on continuation)
instead of exact token match which fails for Qwen2.5-1.5B.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

EVAL_PROMPTS = [
    "The capital of France is",
    "Water boils at one hundred degrees",
    "The speed of light is approximately three hundred thousand",
    "The largest planet in our solar system is",
    "Photosynthesis converts sunlight into chemical",
    "The chemical symbol for gold is",
    "Einstein is famous for the theory of",
    "The human heart has four",
    "DNA has a double helix",
    "Gravity pulls objects toward the center of the",
]


def measure_generation_quality(model, tok, device, prompt, max_new=20):
    """Measure generation quality by checking if output is coherent."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)

    # Measure perplexity of generated continuation
    full_text = prompt + generated
    full_inp = tok(full_text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**full_inp, labels=full_inp['input_ids'])
    ppl = torch.exp(out.loss).item()
    if np.isnan(ppl) or np.isinf(ppl):
        ppl = 1000.0

    return {
        'text': generated.strip()[:50],
        'perplexity': float(ppl),
    }


def measure_per_layer_thermodynamics(model, tok, device, prompt):
    """Get thermodynamic profile (dU/dT) at each layer for last token."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    Us = []
    Ts = []
    for li, hs in enumerate(out.hidden_states):
        h = hs[0, -1, :].float()
        U = h.norm().item()
        Us.append(U)

        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        if np.isnan(T):
            T = 0.0
        Ts.append(T)

    dUdTs = []
    for i in range(1, len(Us)):
        dU = Us[i] - Us[i-1]
        dT = Ts[i] - Ts[i-1]
        dUdTs.append(dU / dT if abs(dT) > 1e-6 else 0.0)

    return Us, Ts, dUdTs


def run_with_ffn_skip(model, tok, device, prompt, skip_layers, max_new=20):
    """Generate with FFN zeroed at skip_layers."""
    hooks = []
    for li in skip_layers:
        if li < len(model.model.layers):
            h = model.model.layers[li].mlp.register_forward_hook(
                lambda m, i, o: o * 0.0)
            hooks.append(h)

    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)

    # PPL
    full_text = prompt + generated
    full_inp = tok(full_text, return_tensors='pt').to(device)
    with torch.no_grad():
        out2 = model(**full_inp, labels=full_inp['input_ids'])
    ppl = torch.exp(out2.loss).item()
    if np.isnan(ppl) or np.isinf(ppl):
        ppl = 1000.0

    for h in hooks:
        h.remove()

    return {
        'text': generated.strip()[:50],
        'perplexity': float(ppl),
    }


def main():
    print("=" * 70)
    print("Phase 88b: Thermodynamic Token Routing (Fixed)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    # === Baseline ===
    print("  Measuring baseline...")
    baseline_ppls = []
    for prompt in EVAL_PROMPTS:
        result = measure_generation_quality(model, tok, device, prompt)
        baseline_ppls.append(result['perplexity'])
        print(f"    {prompt[:30]}... -> {result['text'][:30]} (ppl={result['perplexity']:.1f})")
    baseline_ppl = np.mean(baseline_ppls)
    print(f"  Baseline mean PPL: {baseline_ppl:.2f}")

    # === Thermodynamic analysis ===
    print("\n  Analyzing thermodynamic profiles...")
    all_dUdTs = []
    for prompt in EVAL_PROMPTS[:3]:
        _, _, dUdTs = measure_per_layer_thermodynamics(model, tok, device, prompt)
        all_dUdTs.append(dUdTs)

    mean_dUdT = np.mean(all_dUdTs, axis=0) if all_dUdTs else np.zeros(n_layers)

    # === Threshold sweep with generation quality ===
    thresholds = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    sweep_results = []

    for thresh in thresholds:
        # Determine skip layers based on |dU/dT| > threshold in later layers
        skip_layers = set()
        for i, cv in enumerate(mean_dUdT):
            if abs(cv) > thresh and i > 3:  # never skip first 4 layers
                skip_layers.add(i + 1)  # +1 because dUdT is between layers

        n_skipped = len(skip_layers)
        skip_frac = n_skipped / n_layers

        if n_skipped == 0:
            sweep_results.append({
                'threshold': float(thresh),
                'n_skipped': 0,
                'skip_fraction': 0.0,
                'mean_ppl': float(baseline_ppl),
                'ppl_ratio': 1.0,
            })
            continue

        ppls = []
        for prompt in EVAL_PROMPTS:
            result = run_with_ffn_skip(model, tok, device, prompt, skip_layers)
            ppls.append(result['perplexity'])

        mean_ppl = np.mean(ppls)
        ppl_ratio = mean_ppl / (baseline_ppl + 1e-10)

        sweep_results.append({
            'threshold': float(thresh),
            'n_skipped': n_skipped,
            'skip_fraction': float(skip_frac),
            'mean_ppl': float(mean_ppl),
            'ppl_ratio': float(ppl_ratio),
            'skip_layers': sorted(list(skip_layers)),
        })
        print(f"  thresh={thresh:.0f}: skip={n_skipped}/{n_layers} ({skip_frac:.0%}), "
              f"PPL={mean_ppl:.1f} ({ppl_ratio:.2f}x baseline)")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    threshs_plot = [r['threshold'] for r in sweep_results]
    skip_fracs = [r['skip_fraction'] for r in sweep_results]
    ppl_ratios = [r['ppl_ratio'] for r in sweep_results]

    # (a) PPL ratio vs skip fraction (Pareto)
    axes[0].scatter(skip_fracs, ppl_ratios, s=100, c=threshs_plot, cmap='viridis',
                    edgecolors='black', zorder=5)
    axes[0].plot(skip_fracs, ppl_ratios, '--', color='gray', alpha=0.5)
    axes[0].axhline(y=1.0, color='#27ae60', linestyle='--', label='Baseline PPL')
    axes[0].axhline(y=1.1, color='#f39c12', linestyle=':', label='10% degradation')
    axes[0].set_xlabel('FFN Skip Fraction')
    axes[0].set_ylabel('PPL Ratio (vs Baseline)')
    axes[0].set_title('(a) Quality vs Savings')
    axes[0].legend(fontsize=8)
    cb = plt.colorbar(axes[0].collections[0], ax=axes[0], shrink=0.7)
    cb.set_label('Threshold')

    # (b) dU/dT profile
    layers = np.arange(len(mean_dUdT))
    colors_cv = ['#c0392b' if cv < 0 else '#2980b9' for cv in mean_dUdT]
    axes[1].bar(layers, mean_dUdT, color=colors_cv, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layer Transition')
    axes[1].set_ylabel('$dU/dT$ (Specific Heat)')
    axes[1].set_title('(b) Thermodynamic Profile')
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    # (c) Threshold sweep
    axes[2].plot(threshs_plot, ppl_ratios, 'o-', color='#c0392b', linewidth=2, label='PPL ratio')
    ax2b = axes[2].twinx()
    ax2b.plot(threshs_plot, skip_fracs, 's-', color='#2980b9', linewidth=2, label='Skip fraction')
    axes[2].set_xlabel('$|dU/dT|$ Threshold')
    axes[2].set_ylabel('PPL Ratio', color='#c0392b')
    ax2b.set_ylabel('Skip Fraction', color='#2980b9')
    axes[2].set_title('(c) Threshold Sweep')
    axes[2].set_xscale('log')

    # Find optimal (ppl_ratio < 1.1 with max skip)
    valid_results = [r for r in sweep_results if r['ppl_ratio'] < 1.1 and r['skip_fraction'] > 0]
    if valid_results:
        optimal = max(valid_results, key=lambda r: r['skip_fraction'])
    else:
        optimal = sweep_results[0]

    fig.suptitle(f'Phase 88b: Thermodynamic Token Routing '
                 f'(Best: {optimal["skip_fraction"]:.0%} saved, '
                 f'PPL ratio={optimal["ppl_ratio"]:.2f}x)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase88b_token_routing')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Best <10% degradation: thresh={optimal['threshold']}, "
          f"skip={optimal['skip_fraction']:.0%}, PPL ratio={optimal['ppl_ratio']:.2f}x")
    print(f"{'='*70}")

    save_results('phase88b_token_routing', {
        'experiment': 'Thermodynamic Token Routing (Fixed)',
        'baseline_ppl': float(baseline_ppl),
        'sweep': sweep_results,
        'optimal': optimal,
        'dUdT_profile': [float(v) for v in mean_dUdT],
    })


if __name__ == '__main__':
    main()
