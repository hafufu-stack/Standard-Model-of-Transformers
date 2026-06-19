# -*- coding: utf-8 -*-
"""
Phase 368: Thermodynamic Layer Pruning Guide
=============================================
Test whether the phase transition point identifies layers
that can be safely pruned (removed) with minimal performance loss.

Method:
1. Identify the phase transition layer L0 from thermodynamic profiles.
2. Prune layers in different regions (pre-transition, transition, post-transition).
3. Measure output quality degradation (KL divergence from full model).
4. Hypothesis: pre-transition and post-steady-state layers are more prunable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of relativity states that",
    "In quantum mechanics, the uncertainty principle",
    "Machine learning algorithms can be categorized",
    "The human genome contains approximately",
    "Water molecules consist of two hydrogen",
    "The speed of light in vacuum is",
]


def get_output_distribution(model, tok, prompt, device):
    """Get the output probability distribution."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    logits = out.logits[0, -1, :].float()
    return torch.softmax(logits, dim=-1)


def kl_divergence(p, q):
    """KL(p || q)"""
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    return (p * (p.log() - q.log())).sum().item()


def main():
    print("=" * 70)
    print("Phase 368: Thermodynamic Layer Pruning")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)
        n_layers = len(model.model.layers)

        # Step 1: Find phase transition from thermodynamic profile
        thermo, _ = measure_full_thermodynamics(model, tok, PROMPTS[0], device)
        Ts = [t['T'] for t in thermo]
        dT = np.diff(Ts)
        # Phase transition = layer with maximum |dT|
        L0 = int(np.argmax(np.abs(dT)))
        print(f"  Phase transition at layer {L0}")

        # Step 2: Get baseline output distributions
        baseline_probs = {}
        for prompt in PROMPTS:
            baseline_probs[prompt] = get_output_distribution(model, tok, prompt, device)

        # Step 3: Prune each layer and measure KL divergence
        layer_kl = np.zeros(n_layers)

        for layer_idx in range(n_layers):
            # Skip layer by making it an identity function
            original_forward = model.model.layers[layer_idx].forward

            def skip_forward(*args, **kwargs):
                # Return input unchanged (identity)
                if args:
                    hidden = args[0]
                else:
                    hidden = kwargs.get('hidden_states', None)
                # Return in expected format
                return (hidden,)

            model.model.layers[layer_idx].forward = skip_forward

            kl_values = []
            for prompt in PROMPTS:
                try:
                    pruned_probs = get_output_distribution(model, tok, prompt, device)
                    kl = kl_divergence(baseline_probs[prompt], pruned_probs)
                    kl_values.append(kl)
                except Exception:
                    kl_values.append(float('inf'))

            model.model.layers[layer_idx].forward = original_forward

            mean_kl = np.mean([k for k in kl_values if np.isfinite(k)])
            layer_kl[layer_idx] = mean_kl

        # Classify layers by region
        pre_transition = layer_kl[:L0]
        transition_region = layer_kl[max(0, L0-1):min(n_layers, L0+2)]
        post_transition = layer_kl[min(n_layers-1, L0+2):]

        results[size] = {
            'L0': L0,
            'n_layers': n_layers,
            'layer_kl': layer_kl.tolist(),
            'pre_transition_kl': float(np.mean(pre_transition)) if len(pre_transition) > 0 else 0,
            'transition_kl': float(np.mean(transition_region)) if len(transition_region) > 0 else 0,
            'post_transition_kl': float(np.mean(post_transition)) if len(post_transition) > 0 else 0,
            'most_prunable_layers': [int(i) for i in np.argsort(layer_kl)[:3]],
            'least_prunable_layers': [int(i) for i in np.argsort(layer_kl)[-3:]],
        }

        print(f"  Pre-transition KL: {results[size]['pre_transition_kl']:.4f}")
        print(f"  Transition KL: {results[size]['transition_kl']:.4f}")
        print(f"  Post-transition KL: {results[size]['post_transition_kl']:.4f}")
        print(f"  Most prunable: layers {results[size]['most_prunable_layers']}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 368: Layer Pruning via Thermodynamics", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        r = results[size]
        kl = r['layer_kl']
        L0 = r['L0']
        ax.bar(range(len(kl)), kl, color='#3498db', alpha=0.7)
        ax.axvline(L0, color='red', ls='--', lw=2, label=f'Phase transition (L={L0})')
        ax.set_xlabel('Layer')
        ax.set_ylabel('KL Divergence (pruning cost)')
        ax.set_title(f'Qwen2.5-{size}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase368_pruning')
    plt.close()

    save_results('phase368_pruning', {
        'experiment': 'Thermodynamic Layer Pruning',
        'results': results,
    })


if __name__ == '__main__':
    main()
