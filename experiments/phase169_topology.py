# -*- coding: utf-8 -*-
"""
Phase 169: Topological Order Detection
Measure the "winding number" of the eta trajectory.
How many times does eta oscillate around the critical point?
This is related to topological phase transitions.
Also measure the Berry phase (geometric phase) of the hidden state.
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
    print("Phase 169: Topological Order Detection")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    all_S = np.zeros((len(PROMPTS), n_layers))
    all_berry = np.zeros((len(PROMPTS), n_layers - 1))

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hidden_states = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            hidden_states.append(h)

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            all_S[pi, li] = S if not np.isnan(S) else 0

        # Berry phase: angle between consecutive hidden state directions
        for li in range(n_layers - 1):
            h1 = hidden_states[li]
            h2 = hidden_states[li + 1]
            # Normalize
            h1_n = h1 / (h1.norm() + 1e-10)
            h2_n = h2 / (h2.norm() + 1e-10)
            # Berry connection: phase angle
            dot = torch.dot(h1_n, h2_n).item()
            dot = max(-1, min(1, dot))
            angle = np.arccos(dot)
            all_berry[pi, li] = angle

    mean_S = np.mean(all_S, axis=0)
    mean_berry = np.mean(all_berry, axis=0)
    cumulative_berry = np.cumsum(mean_berry)

    # Winding analysis: count zero-crossings of dS/dL around the mean
    dS = np.gradient(mean_S)
    zero_crossings = 0
    for i in range(len(dS) - 1):
        if dS[i] * dS[i+1] < 0:
            zero_crossings += 1

    # Topological invariant: total accumulated Berry phase
    total_berry = cumulative_berry[-1]
    berry_winding = total_berry / (2 * np.pi)

    layers = np.arange(n_layers)
    trans = np.arange(n_layers - 1) + 0.5

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Berry connection (angle between layers)
    axes[0,0].plot(trans, mean_berry, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_xlabel('Layer Transition')
    axes[0,0].set_ylabel('Berry Angle (rad)')
    axes[0,0].set_title('(a) Berry Connection')
    axes[0,0].legend()

    # (b) Cumulative Berry phase
    axes[0,1].plot(trans, cumulative_berry, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    for n in range(1, int(berry_winding) + 2):
        axes[0,1].axhline(y=n * 2 * np.pi, color='gray', linewidth=0.5, linestyle=':')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Cumulative Phase (rad)')
    axes[0,1].set_title(f'(b) Berry Phase (winding={berry_winding:.2f})')

    # (c) dS/dL zero crossings
    dS_colors = ['#c0392b' if d > 0 else '#2980b9' for d in dS]
    axes[0,2].bar(layers, dS, color=dS_colors, alpha=0.7, edgecolor='black')
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=0, color='black', linewidth=1)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$dS/dL$')
    axes[0,2].set_title(f'(c) S Gradient ({zero_crossings} crossings)')

    # (d) Pre vs post L0 Berry angle
    pre_berry = np.mean(mean_berry[:20])
    post_berry = np.mean(mean_berry[20:])
    axes[1,0].bar(['Pre-$L_0$', 'Post-$L_0$'], [pre_berry, post_berry],
                  color=['#2980b9', '#c0392b'], alpha=0.8, edgecolor='black')
    axes[1,0].set_ylabel('Mean Berry Angle')
    axes[1,0].set_title('(d) Berry Angle by Phase')

    # (e) S trajectory colored by Berry angle
    for i in range(len(trans)):
        color_val = mean_berry[i] / (max(mean_berry) + 1e-10)
        c = plt.cm.hot(color_val)
        axes[1,1].plot([layers[i], layers[i+1]], [mean_S[i], mean_S[i+1]],
                      '-', color=c, linewidth=3)
    sm = plt.cm.ScalarMappable(cmap='hot',
                                norm=plt.Normalize(min(mean_berry), max(mean_berry)))
    plt.colorbar(sm, ax=axes[1,1], label='Berry Angle')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) S Colored by Berry Angle')

    # (f) Summary
    summary = (
        f"Topological Order Detection\n\n"
        f"Total Berry phase: {total_berry:.2f} rad\n"
        f"Berry winding number: {berry_winding:.2f}\n\n"
        f"dS/dL zero crossings: {zero_crossings}\n\n"
        f"Pre-L0 Berry: {pre_berry:.4f}\n"
        f"Post-L0 Berry: {post_berry:.4f}\n"
        f"Ratio: {post_berry/(pre_berry+1e-10):.2f}x\n\n"
        f"Geometric phase\n"
        f"{'ACCUMULATES' if berry_winding > 0.5 else 'is negligible'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 169: Topological Order Detection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase169_topology')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Berry winding: {berry_winding:.2f}")
    print(f"Zero crossings: {zero_crossings}")
    print(f"Pre/Post Berry ratio: {post_berry/(pre_berry+1e-10):.2f}x")
    print(f"{'='*70}")

    save_results('phase169_topology', {
        'experiment': 'Topological Order Detection',
        'summary': {
            'berry_winding': float(berry_winding),
            'zero_crossings': int(zero_crossings),
            'pre_berry': float(pre_berry),
            'post_berry': float(post_berry),
        }
    })


if __name__ == '__main__':
    main()
