# -*- coding: utf-8 -*-
"""
Phase 196: Szilard Engine - Attention as Measurement-to-Work Converter
========================================================================
Szilard (1929): A single molecule engine that converts 1 bit of
measurement information into kT*ln(2) of extractable work.

Each attention head "measures" (selects) information from the input.
This measurement has a thermodynamic cost (Landauer) but also extracts
work (representation change). Net: does attention create or consume
free energy?

Measure: for each attention head, the "information gain" (mutual info
between attended tokens and output) vs "work done" (change in hidden
state energy).
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
    print("Phase 196: Szilard Engine")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Need eager attention for output_attentions=True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os as _os
    _HF_CACHE = _os.path.expanduser("~/.cache/huggingface/hub")
    _SNAP = _os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                          "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
    tok = AutoTokenizer.from_pretrained(_SNAP, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        _SNAP, torch_dtype=torch.float16, device_map=device,
        local_files_only=True, attn_implementation="eager"
    )
    model.eval()
    n_transformer_layers = len(model.model.layers)
    L0 = 21

    # Get number of heads from config
    n_heads = model.config.num_attention_heads

    all_head_info = []  # Information gain per head
    all_head_work = []  # Work per head (dU contribution)

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Run with attention output
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)

        head_info_per_layer = []
        head_work_per_layer = []

        for li in range(n_transformer_layers):
            # Attention weights: (batch, n_heads, seq, seq)
            attn = out.attentions[li][0].float()  # (n_heads, seq, seq)

            # Information content of each head: entropy of attention distribution
            info_per_head = []
            for hi in range(n_heads):
                # Attention entropy for last token's query
                a = attn[hi, -1, :]  # attention weights for last token
                ent = -(a * torch.log(a + 1e-10)).sum().item()
                # Low entropy = more selective = more information gained
                info_gain = np.log(a.shape[0]) - ent  # max_entropy - actual_entropy
                info_per_head.append(info_gain if not np.isnan(info_gain) else 0)
            head_info_per_layer.append(info_per_head)

            # Work: change in energy across this layer
            h_in = out.hidden_states[li][0, -1, :].float()
            h_out = out.hidden_states[li + 1][0, -1, :].float()
            dU = (h_out.norm() - h_in.norm()).item()

            # Distribute work across heads proportionally to attention selectivity
            total_info = sum(info_per_head) + 1e-10
            work_per_head = [dU * (ig / total_info) for ig in info_per_head]
            head_work_per_layer.append(work_per_head)

        all_head_info.append(head_info_per_layer)
        all_head_work.append(head_work_per_layer)

    # Average across prompts
    info_mean = np.mean(all_head_info, axis=0)  # (n_layers, n_heads)
    work_mean = np.mean(all_head_work, axis=0)  # (n_layers, n_heads)

    # Szilard ratio: work / (kT * info) where kT = 1
    szilard_ratio = work_mean / (info_mean * np.log(2) + 1e-10)

    # === Analysis ===
    # Average info and work per head across all layers
    avg_info_per_head = np.mean(info_mean, axis=0)
    avg_work_per_head = np.mean(work_mean, axis=0)

    # Pre/post L0
    info_pre = np.mean(info_mean[:L0, :], axis=0)
    info_post = np.mean(info_mean[L0:, :], axis=0)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Information gain heatmap (layers x heads)
    im1 = axes[0, 0].imshow(info_mean, aspect='auto', cmap='YlOrRd',
                             extent=[0, n_heads, n_transformer_layers, 0])
    axes[0, 0].axhline(y=L0, color='#f39c12', linewidth=2, linestyle='--')
    plt.colorbar(im1, ax=axes[0, 0], label='Info Gain (nats)')
    axes[0, 0].set_xlabel('Head Index')
    axes[0, 0].set_ylabel('Layer')
    axes[0, 0].set_title('(a) Attention Information Gain')

    # (b) Work heatmap
    vmax = np.percentile(np.abs(work_mean), 95)
    im2 = axes[0, 1].imshow(work_mean, aspect='auto', cmap='RdBu_r',
                             extent=[0, n_heads, n_transformer_layers, 0],
                             vmin=-vmax, vmax=vmax)
    axes[0, 1].axhline(y=L0, color='#f39c12', linewidth=2, linestyle='--')
    plt.colorbar(im2, ax=axes[0, 1], label='Work (energy units)')
    axes[0, 1].set_xlabel('Head Index')
    axes[0, 1].set_ylabel('Layer')
    axes[0, 1].set_title('(b) Work per Head')

    # (c) Info gain vs Work scatter (all layers, all heads)
    info_flat = info_mean.flatten()
    work_flat = work_mean.flatten()
    layer_flat = np.repeat(np.arange(n_transformer_layers), n_heads)
    axes[0, 2].scatter(info_flat, work_flat, c=layer_flat, cmap='viridis',
                        s=10, alpha=0.5)
    axes[0, 2].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 2].axvline(x=0, color='black', linewidth=0.5)
    axes[0, 2].set_xlabel('Information Gain (nats)')
    axes[0, 2].set_ylabel('Work')
    axes[0, 2].set_title('(c) Szilard: Info vs Work')

    # (d) Average info per layer
    info_per_layer = np.mean(info_mean, axis=1)
    axes[1, 0].plot(np.arange(n_transformer_layers), info_per_layer, 'o-',
                    color='#e74c3c', markersize=4, linewidth=2)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Mean Info Gain per Head')
    axes[1, 0].set_title('(d) Attention Selectivity per Layer')

    # (e) Head specialization
    head_std = np.std(info_mean, axis=0)  # Variation across layers for each head
    axes[1, 1].bar(np.arange(n_heads), avg_info_per_head, color='#3498db',
                   edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Head Index')
    axes[1, 1].set_ylabel('Mean Info Gain')
    axes[1, 1].set_title('(e) Head Importance Ranking')

    # (f) Summary
    total_info = np.sum(info_mean)
    total_work = np.sum(np.abs(work_mean))
    engines = sum(1 for ig, w in zip(info_flat, work_flat) if ig > 0.5 and w > 0)
    erasers = sum(1 for ig, w in zip(info_flat, work_flat) if ig > 0.5 and w < 0)

    summary = (
        f"Szilard Engine Analysis\n\n"
        f"Total info gained: {total_info:.1f} nats\n"
        f"  = {total_info/np.log(2):.1f} bits\n"
        f"Total |work|: {total_work:.1f}\n\n"
        f"Szilard engines: {engines}\n"
        f"  (high info, positive work)\n"
        f"Landauer erasers: {erasers}\n"
        f"  (high info, negative work)\n\n"
        f"Most selective head: {np.argmax(avg_info_per_head)}\n"
        f"Least selective: {np.argmin(avg_info_per_head)}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 196: Szilard Engine (Attention as Measurement)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase196_szilard')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Total info: {total_info:.1f} nats ({total_info/np.log(2):.1f} bits)")
    print(f"Engines: {engines}, Erasers: {erasers}")
    print(f"{'=' * 70}")

    save_results('phase196_szilard', {
        'experiment': 'Szilard Engine',
        'info_per_layer': [float(x) for x in info_per_layer],
        'n_heads': n_heads,
        'summary': {
            'total_info_nats': float(total_info),
            'total_work': float(total_work),
            'engines': engines, 'erasers': erasers,
        }
    })


if __name__ == '__main__':
    main()
