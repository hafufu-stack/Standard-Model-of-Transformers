# -*- coding: utf-8 -*-
"""
Phase 108: FFN Nonlinearity as Heat Engine
Phase 106 showed FFN is the primary driver of eta.
Investigate the FFN activation statistics (pre/post nonlinearity)
to understand HOW the FFN acts as a thermodynamic heat engine.
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
    "The cosmic microwave background reveals the early universe",
    "General relativity describes gravity as spacetime curvature",
]


def main():
    print("=" * 70)
    print("Phase 108: FFN Nonlinearity as Heat Engine")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 21.7

    # Hook into FFN gate/up/down projections to capture activation stats
    layer_stats = []

    for li in range(n_layers):
        gate_acts = []
        up_acts = []
        down_acts = []

        # Hook the gate and up projections
        gate_data = []
        up_data = []

        def make_gate_hook(storage):
            def hook(module, input, output):
                storage.append(output[0, -1, :].detach().float().cpu())
                return output
            return hook

        # Qwen2.5 uses SiLU gating: output = down(gate_act * up_proj(x))
        # where gate_act = SiLU(gate_proj(x))
        mlp = model.model.layers[li].mlp
        h_gate = mlp.gate_proj.register_forward_hook(make_gate_hook(gate_data))
        h_up = mlp.up_proj.register_forward_hook(make_gate_hook(up_data))

        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                model(**inp)

        h_gate.remove()
        h_up.remove()

        if gate_data and up_data:
            gate_all = torch.stack(gate_data)
            up_all = torch.stack(up_data)

            # Activation statistics
            gate_mean = gate_all.mean().item()
            gate_std = gate_all.std().item()
            gate_sparsity = (gate_all.abs() < 0.1).float().mean().item()

            # SiLU applied to gate
            silu_gate = torch.nn.functional.silu(gate_all)
            silu_mean = silu_gate.mean().item()
            silu_std = silu_gate.std().item()
            silu_sparsity = (silu_gate.abs() < 0.01).float().mean().item()

            # Information compression ratio
            # H(gate) vs H(SiLU(gate))
            def est_entropy(x):
                x_flat = x.flatten()
                hist, _ = np.histogram(x_flat.numpy(), bins=50, density=True)
                hist = hist[hist > 0]
                return -np.sum(hist * np.log(hist + 1e-10)) * (x_flat.max().item() - x_flat.min().item()) / 50

            H_pre = est_entropy(gate_all)
            H_post = est_entropy(silu_gate)
            compression = 1.0 - H_post / (H_pre + 1e-10) if H_pre > 0 else 0

            # "Work" = how much does FFN change the representation?
            up_mean = up_all.mean().item()
            up_std = up_all.std().item()

            stats = {
                'layer': li,
                'gate_mean': float(gate_mean),
                'gate_std': float(gate_std),
                'gate_sparsity': float(gate_sparsity),
                'silu_mean': float(silu_mean),
                'silu_std': float(silu_std),
                'silu_sparsity': float(silu_sparsity),
                'H_pre': float(H_pre),
                'H_post': float(H_post),
                'compression': float(compression),
                'up_mean': float(up_mean),
                'up_std': float(up_std),
            }
        else:
            stats = {'layer': li, 'gate_mean': 0, 'gate_std': 0, 'gate_sparsity': 0,
                     'silu_mean': 0, 'silu_std': 0, 'silu_sparsity': 0,
                     'H_pre': 0, 'H_post': 0, 'compression': 0,
                     'up_mean': 0, 'up_std': 0}

        layer_stats.append(stats)
        if li % 7 == 0 or li == n_layers - 1:
            print(f"  L{li:2d}: sparsity={stats['silu_sparsity']:.3f}, "
                  f"compress={stats['compression']:.3f}, "
                  f"H_pre={stats['H_pre']:.3f}, H_post={stats['H_post']:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers = [s['layer'] for s in layer_stats]

    # (a) Sparsity profile
    silu_sparsities = [s['silu_sparsity'] for s in layer_stats]
    gate_sparsities = [s['gate_sparsity'] for s in layer_stats]
    axes[0,0].plot(layers, silu_sparsities, 'o-', color='#c0392b', markersize=4,
                   label='SiLU output')
    axes[0,0].plot(layers, gate_sparsities, 's-', color='#2980b9', markersize=4,
                   label='Gate linear')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Sparsity')
    axes[0,0].set_title('(a) FFN Sparsity')
    axes[0,0].legend(fontsize=8)

    # (b) Information compression
    compressions = [s['compression'] for s in layer_stats]
    colors_c = ['#27ae60' if c > 0 else '#c0392b' for c in compressions]
    axes[0,1].bar(layers, compressions, color=colors_c, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Compression $(1 - H_{post}/H_{pre})$')
    axes[0,1].set_title('(b) Information Compression')

    # (c) Entropy before and after SiLU
    H_pres = [s['H_pre'] for s in layer_stats]
    H_posts = [s['H_post'] for s in layer_stats]
    axes[0,2].plot(layers, H_pres, 'o-', color='#8e44ad', markersize=4, label='$H_{pre}$')
    axes[0,2].plot(layers, H_posts, 's-', color='#e67e22', markersize=4, label='$H_{post}$')
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Entropy')
    axes[0,2].set_title('(c) Entropy Before/After Nonlinearity')
    axes[0,2].legend(fontsize=8)

    # (d) Gate activation std (a proxy for "energy flow")
    gate_stds = [s['gate_std'] for s in layer_stats]
    silu_stds = [s['silu_std'] for s in layer_stats]
    axes[1,0].plot(layers, gate_stds, 'o-', color='#c0392b', markersize=4, label='Gate')
    axes[1,0].plot(layers, silu_stds, 's-', color='#2980b9', markersize=4, label='SiLU')
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Activation Std')
    axes[1,0].set_title('(d) Activation Spread')
    axes[1,0].legend(fontsize=8)

    # (e) Pre vs Post transition comparison
    pre_stats = [s for s in layer_stats if s['layer'] < L0]
    post_stats = [s for s in layer_stats if s['layer'] >= L0]
    pre_comp = np.mean([s['compression'] for s in pre_stats])
    post_comp = np.mean([s['compression'] for s in post_stats])
    pre_sparse = np.mean([s['silu_sparsity'] for s in pre_stats])
    post_sparse = np.mean([s['silu_sparsity'] for s in post_stats])

    categories = ['Compression\n(pre)', 'Compression\n(post)',
                   'Sparsity\n(pre)', 'Sparsity\n(post)']
    vals = [pre_comp, post_comp, pre_sparse, post_sparse]
    bar_colors = ['#3498db', '#c0392b', '#3498db', '#c0392b']
    axes[1,1].bar(range(4), vals, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(4))
    axes[1,1].set_xticklabels(categories, fontsize=8)
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('(e) Pre vs Post Transition')

    # (f) Summary
    summary = (
        f"FFN Heat Engine Analysis\n\n"
        f"Pre-transition (L<{L0:.0f}):\n"
        f"  Compression: {pre_comp:.3f}\n"
        f"  SiLU sparsity: {pre_sparse:.3f}\n\n"
        f"Post-transition (L>={L0:.0f}):\n"
        f"  Compression: {post_comp:.3f}\n"
        f"  SiLU sparsity: {post_sparse:.3f}\n\n"
        f"Compression change: {post_comp-pre_comp:+.3f}\n"
        f"Sparsity change: {post_sparse-pre_sparse:+.3f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 108: FFN Nonlinearity as Heat Engine',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase108_ffn_engine')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-transition: compress={pre_comp:.3f}, sparsity={pre_sparse:.3f}")
    print(f"Post-transition: compress={post_comp:.3f}, sparsity={post_sparse:.3f}")
    print(f"{'='*70}")

    save_results('phase108_ffn_engine', {
        'experiment': 'FFN Nonlinearity as Heat Engine',
        'layer_stats': layer_stats,
        'summary': {
            'pre_compression': float(pre_comp),
            'post_compression': float(post_comp),
            'pre_sparsity': float(pre_sparse),
            'post_sparsity': float(post_sparse),
        }
    })


if __name__ == '__main__':
    main()
