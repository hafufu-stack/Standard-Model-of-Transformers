# -*- coding: utf-8 -*-
"""
Phase 121: SiLU as the Higgs Potential
The SiLU function x*sigmoid(x) has a "Mexican hat" shape near origin.
Track what fraction of neurons sit in the "valley" (negative dip)
vs the "linear regime" at each layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]


def main():
    print("=" * 70)
    print("Phase 121: SiLU as the Higgs Potential")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 21.7

    # SiLU has minimum at x ~ -1.28 with value ~ -0.278
    # Valley: x in [-3, 0] (negative dip region)
    # Linear: x > 1 (approximately linear region)
    # Dead: x < -5 (effectively zero)

    valley_fracs = []
    linear_fracs = []
    dead_fracs = []
    mean_silu_out = []
    valley_depth = []  # mean SiLU value in valley

    for li in range(n_layers):
        gate_data = []

        def make_hook(storage):
            def hook(module, input, output):
                storage.append(output[0, -1, :].detach().float().cpu())
            return hook

        h = model.model.layers[li].mlp.gate_proj.register_forward_hook(make_hook(gate_data))

        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                model(**inp)

        h.remove()

        if gate_data:
            all_gates = torch.stack(gate_data)  # (N, d_ff)

            # Apply SiLU to see output
            silu_out = torch.nn.functional.silu(all_gates)

            # Classify neurons by gate input value
            valley = ((all_gates > -3) & (all_gates < 0)).float().mean().item()
            linear = (all_gates > 1).float().mean().item()
            dead = (all_gates < -5).float().mean().item()

            # Mean SiLU output
            mean_s = silu_out.mean().item()

            # "Higgs valley depth" = mean SiLU for neurons in valley
            in_valley = (all_gates > -3) & (all_gates < 0)
            if in_valley.sum() > 0:
                vd = silu_out[in_valley].mean().item()
            else:
                vd = 0

            valley_fracs.append(float(valley))
            linear_fracs.append(float(linear))
            dead_fracs.append(float(dead))
            mean_silu_out.append(float(mean_s))
            valley_depth.append(float(vd))
        else:
            valley_fracs.append(0)
            linear_fracs.append(0)
            dead_fracs.append(0)
            mean_silu_out.append(0)
            valley_depth.append(0)

        if li % 7 == 0 or li == n_layers - 1:
            print(f"  L{li:2d}: valley={valley_fracs[-1]:.3f}, "
                  f"linear={linear_fracs[-1]:.3f}, dead={dead_fracs[-1]:.3f}")

    layers = np.arange(n_layers)

    # Phase transition signature
    pre_valley = np.mean(valley_fracs[:int(L0)])
    post_valley = np.mean(valley_fracs[int(L0):])
    pre_linear = np.mean(linear_fracs[:int(L0)])
    post_linear = np.mean(linear_fracs[int(L0):])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Valley fraction
    axes[0, 0].plot(layers, valley_fracs, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Valley Fraction ($-3 < x < 0$)')
    axes[0, 0].set_title('(a) Neurons in Higgs Valley')
    axes[0, 0].legend()

    # (b) Linear fraction
    axes[0, 1].plot(layers, linear_fracs, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Linear Fraction ($x > 1$)')
    axes[0, 1].set_title('(b) Neurons in Linear Regime')

    # (c) Dead fraction
    axes[0, 2].plot(layers, dead_fracs, 'o-', color='#7f8c8d', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Dead Fraction ($x < -5$)')
    axes[0, 2].set_title('(c) Dead Neurons')

    # (d) Stacked area chart
    axes[1, 0].stackplot(layers, dead_fracs, valley_fracs, linear_fracs,
                          [1 - d - v - l for d, v, l in zip(dead_fracs, valley_fracs, linear_fracs)],
                          colors=['#7f8c8d', '#c0392b', '#2980b9', '#27ae60'],
                          labels=['Dead', 'Valley', 'Linear', 'Other'],
                          alpha=0.7)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Fraction')
    axes[1, 0].set_title('(d) Neuron Population')
    axes[1, 0].legend(fontsize=7, loc='upper left')

    # (e) Valley depth (mean SiLU in valley)
    axes[1, 1].plot(layers, valley_depth, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Mean SiLU in Valley')
    axes[1, 1].set_title('(e) Higgs Valley Depth')

    # (f) SiLU potential illustration + summary
    x = np.linspace(-5, 3, 200)
    silu = x / (1 + np.exp(-x))
    axes[1, 2].plot(x, silu, 'k-', linewidth=2, label='SiLU($x$)')
    axes[1, 2].axhline(y=0, color='gray', linewidth=0.5)
    axes[1, 2].axvspan(-3, 0, alpha=0.2, color='#c0392b', label='Valley')
    axes[1, 2].axvspan(1, 3, alpha=0.2, color='#2980b9', label='Linear')
    axes[1, 2].set_xlabel('$x$ (gate input)')
    axes[1, 2].set_ylabel('SiLU($x$)')
    axes[1, 2].set_title(f'(f) SiLU = Higgs Potential')
    axes[1, 2].legend(fontsize=7)

    fig.suptitle(f'Phase 121: SiLU as Higgs Potential '
                 f'(valley: {pre_valley:.3f}->{post_valley:.3f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase121_higgs')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Valley: pre={pre_valley:.3f}, post={post_valley:.3f}")
    print(f"Linear: pre={pre_linear:.3f}, post={post_linear:.3f}")
    print(f"{'='*70}")

    save_results('phase121_higgs', {
        'experiment': 'SiLU as Higgs Potential',
        'valley_fracs': valley_fracs,
        'linear_fracs': linear_fracs,
        'dead_fracs': dead_fracs,
        'valley_depth': valley_depth,
        'summary': {
            'pre_valley': float(pre_valley),
            'post_valley': float(post_valley),
            'pre_linear': float(pre_linear),
            'post_linear': float(post_linear),
        }
    })


if __name__ == '__main__':
    main()
