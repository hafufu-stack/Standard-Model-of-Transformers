# -*- coding: utf-8 -*-
"""
Phase 165: Hidden State Norm Dynamics
Measure how hidden state norms evolve through layers.
This is the "energy" of the system. Does it show critical behavior?
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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def main():
    print("=" * 70)
    print("Phase 165: Hidden State Norm Dynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    all_norms = np.zeros((len(PROMPTS), n_layers))
    all_norm_growth = np.zeros((len(PROMPTS), n_layers - 1))
    all_cosine_prev = np.zeros((len(PROMPTS), n_layers - 1))
    all_dim = np.zeros((len(PROMPTS), n_layers))  # Participation ratio (effective dim)

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        prev_h = None
        for li in range(n_layers):
            h = out.hidden_states[li][0, -1, :].float()
            norm = h.norm().item()
            all_norms[pi, li] = norm

            # Participation ratio: (sum |h_i|)^2 / sum(|h_i|^2) normalized
            h_abs = h.abs()
            pr = (h_abs.sum()**2) / (h_abs.pow(2).sum() + 1e-10)
            all_dim[pi, li] = pr.item() / len(h)  # Normalize to [0, 1]

            if prev_h is not None:
                # Norm growth rate
                all_norm_growth[pi, li-1] = norm / (all_norms[pi, li-1] + 1e-10)

                # Cosine with previous
                cos = torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h.unsqueeze(0)).item()
                all_cosine_prev[pi, li-1] = cos

            prev_h = h

    mean_norms = np.mean(all_norms, axis=0)
    std_norms = np.std(all_norms, axis=0)
    mean_growth = np.mean(all_norm_growth, axis=0)
    mean_cosine = np.mean(all_cosine_prev, axis=0)
    mean_dim = np.mean(all_dim, axis=0)
    var_norms = np.var(all_norms, axis=0)

    layers = np.arange(n_layers)
    trans = np.arange(n_layers - 1) + 0.5

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Norm profile
    axes[0,0].plot(layers, mean_norms, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,0].fill_between(layers, mean_norms - std_norms, mean_norms + std_norms,
                           alpha=0.2, color='#2980b9')
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$||h||$')
    axes[0,0].set_title('(a) Hidden State Norm')
    axes[0,0].legend()

    # (b) Growth rate
    axes[0,1].plot(trans, mean_growth, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].axhline(y=1, color='gray', linewidth=0.5)
    axes[0,1].set_xlabel('Layer Transition')
    axes[0,1].set_ylabel('$||h_{l+1}|| / ||h_l||$')
    axes[0,1].set_title('(b) Norm Growth Rate')

    # (c) Cosine with previous layer
    axes[0,2].plot(trans, mean_cosine, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer Transition')
    axes[0,2].set_ylabel('cos($h_l$, $h_{l+1}$)')
    axes[0,2].set_title('(c) Representation Continuity')

    # (d) Effective dimension (participation ratio)
    axes[1,0].plot(layers, mean_dim, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Participation Ratio')
    axes[1,0].set_title('(d) Effective Dimensionality')

    # (e) Norm variance (fluctuations)
    axes[1,1].plot(layers, var_norms, 'o-', color='#e74c3c', markersize=4, linewidth=2)
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    peak_var = np.argmax(var_norms[4:]) + 4
    axes[1,1].axvline(x=peak_var, color='#27ae60', linewidth=1.5, linestyle=':',
                      label=f'Peak L{peak_var}')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Var($||h||$)')
    axes[1,1].set_title('(e) Norm Fluctuations')
    axes[1,1].legend()

    # (f) Summary
    pre_norm = np.mean(mean_norms[:20])
    post_norm = np.mean(mean_norms[20:])
    pre_dim = np.mean(mean_dim[:20])
    post_dim = np.mean(mean_dim[20:])
    summary = (
        f"Hidden State Norm Dynamics\n\n"
        f"Pre-L0 norm: {pre_norm:.1f}\n"
        f"Post-L0 norm: {post_norm:.1f}\n"
        f"Change: {(post_norm-pre_norm)/pre_norm*100:+.1f}%\n\n"
        f"Pre-L0 dim: {pre_dim:.4f}\n"
        f"Post-L0 dim: {post_dim:.4f}\n\n"
        f"Norm fluctuation peak: L{peak_var}\n"
        f"Distance from L0: {abs(peak_var - 21.7):.1f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 165: Hidden State Norm Dynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase165_norms')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-L0 norm: {pre_norm:.1f}, Post-L0 norm: {post_norm:.1f}")
    print(f"Norm fluctuation peak: L{peak_var}")
    print(f"{'='*70}")

    save_results('phase165_norms', {
        'experiment': 'Hidden State Norm Dynamics',
        'summary': {
            'pre_norm': float(pre_norm),
            'post_norm': float(post_norm),
            'peak_var_layer': int(peak_var),
        }
    })


if __name__ == '__main__':
    main()
