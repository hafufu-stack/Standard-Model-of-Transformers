# -*- coding: utf-8 -*-
"""
Phase 106: Attention vs FFN Contribution to Phase Transition
Is the eta phase transition driven by attention patterns or FFN processing?
Separately zero out attention and FFN at each layer and measure eta impact.
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
]


def measure_eta_full(model, tok, device):
    """Full model eta (baseline)."""
    etas = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_vals = []
        for li, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(T):
                T_vals.append(T)
        if len(T_vals) >= 4:
            T_hot = max(T_vals)
            T_cold = min(T_vals[len(T_vals)//2:])
            if T_hot > 0.01:
                etas.append(1.0 - T_cold / T_hot)
    return float(np.mean(etas)) if etas else 0.0


def measure_eta_with_ablation(model, tok, device, layer_idx, component='ffn'):
    """Measure eta with one component ablated at layer_idx."""
    hook = None

    if component == 'ffn':
        def hook_fn(module, input, output):
            return output * 0.0
        hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    elif component == 'attn':
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (output[0] * 0.0,) + output[1:]
            return output * 0.0
        hook = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)

    eta = measure_eta_full(model, tok, device)

    if hook:
        hook.remove()

    return eta


def main():
    print("=" * 70)
    print("Phase 106: Attention vs FFN Contribution")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 21.7

    baseline_eta = measure_eta_full(model, tok, device)
    print(f"  Baseline eta: {baseline_eta:.4f}")

    ffn_impacts = []
    attn_impacts = []

    for li in range(n_layers):
        # FFN ablation
        eta_no_ffn = measure_eta_with_ablation(model, tok, device, li, 'ffn')
        ffn_impact = baseline_eta - eta_no_ffn  # positive = FFN helps

        # Attention ablation
        eta_no_attn = measure_eta_with_ablation(model, tok, device, li, 'attn')
        attn_impact = baseline_eta - eta_no_attn  # positive = attn helps

        ffn_impacts.append(float(ffn_impact))
        attn_impacts.append(float(attn_impact))

        if li % 7 == 0 or li == n_layers - 1:
            print(f"  L{li:2d}: FFN impact={ffn_impact:+.4f}, Attn impact={attn_impact:+.4f}")

    ffn_arr = np.array(ffn_impacts)
    attn_arr = np.array(attn_impacts)

    # Pre/post transition comparison
    pre_mask = np.arange(n_layers) < L0
    post_mask = ~pre_mask

    ffn_pre = np.mean(ffn_arr[pre_mask])
    ffn_post = np.mean(ffn_arr[post_mask])
    attn_pre = np.mean(attn_arr[pre_mask])
    attn_post = np.mean(attn_arr[post_mask])

    # Most critical layer for each component
    ffn_critical = np.argmax(np.abs(ffn_arr))
    attn_critical = np.argmax(np.abs(attn_arr))

    # Dominance ratio at each layer
    dominance = ffn_arr / (np.abs(attn_arr) + np.abs(ffn_arr) + 1e-10)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) FFN vs Attention impact profile
    x = np.arange(n_layers)
    w = 0.35
    axes[0,0].bar(x - w/2, ffn_arr, w, color='#c0392b', alpha=0.7, label='FFN')
    axes[0,0].bar(x + w/2, attn_arr, w, color='#2980b9', alpha=0.7, label='Attention')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,0].axhline(y=0, color='black', linewidth=0.5)
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$\\Delta\\eta$ (impact on efficiency)')
    axes[0,0].set_title('(a) Component Impact Profile')
    axes[0,0].legend(fontsize=8)

    # (b) Cumulative contribution
    ffn_cum = np.cumsum(ffn_arr)
    attn_cum = np.cumsum(attn_arr)
    axes[0,1].plot(x, ffn_cum, 'o-', color='#c0392b', markersize=3, label='FFN cumulative')
    axes[0,1].plot(x, attn_cum, 's-', color='#2980b9', markersize=3, label='Attn cumulative')
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Cumulative $\\Delta\\eta$')
    axes[0,1].set_title('(b) Cumulative Contribution')
    axes[0,1].legend(fontsize=8)

    # (c) Dominance ratio
    axes[0,2].plot(x, dominance, 'o-', color='#27ae60', markersize=4)
    axes[0,2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal')
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].fill_between(x, 0, dominance, where=dominance > 0.5,
                            alpha=0.2, color='#c0392b', label='FFN dominant')
    axes[0,2].fill_between(x, 0, dominance, where=dominance <= 0.5,
                            alpha=0.2, color='#2980b9', label='Attn dominant')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('FFN/(FFN+Attn)')
    axes[0,2].set_title('(c) Dominance Ratio')
    axes[0,2].legend(fontsize=7)

    # (d) Pre vs Post transition
    categories = ['FFN\n(pre)', 'FFN\n(post)', 'Attn\n(pre)', 'Attn\n(post)']
    vals = [ffn_pre, ffn_post, attn_pre, attn_post]
    bar_colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9']
    axes[1,0].bar(range(4), vals, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(4))
    axes[1,0].set_xticklabels(categories, fontsize=9)
    axes[1,0].set_ylabel('Mean $\\Delta\\eta$')
    axes[1,0].set_title('(d) Pre vs Post Transition')
    axes[1,0].axhline(y=0, color='black', linewidth=0.5)

    # (e) Scatter: FFN impact vs Attn impact
    axes[1,1].scatter(attn_arr, ffn_arr, c=x, cmap='coolwarm', s=60,
                      edgecolors='black', zorder=5)
    axes[1,1].axhline(y=0, color='gray', linewidth=0.5)
    axes[1,1].axvline(x=0, color='gray', linewidth=0.5)
    axes[1,1].plot([-0.1, 0.1], [-0.1, 0.1], 'k--', alpha=0.3)
    cb = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1], shrink=0.7)
    cb.set_label('Layer')
    axes[1,1].set_xlabel('Attention Impact')
    axes[1,1].set_ylabel('FFN Impact')
    axes[1,1].set_title('(e) Component Correlation')

    # (f) Summary
    ffn_total = np.sum(np.abs(ffn_arr))
    attn_total = np.sum(np.abs(attn_arr))
    driver = 'FFN' if ffn_total > attn_total else 'Attention'
    summary = (
        f"Component Analysis\n\n"
        f"FFN total impact: {ffn_total:.3f}\n"
        f"Attn total impact: {attn_total:.3f}\n"
        f"Primary driver: {driver}\n\n"
        f"FFN critical layer: L{ffn_critical}\n"
        f"Attn critical layer: L{attn_critical}\n\n"
        f"FFN pre/post: {ffn_pre:.4f}/{ffn_post:.4f}\n"
        f"Attn pre/post: {attn_pre:.4f}/{attn_post:.4f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 106: Attention vs FFN ({driver} dominant)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase106_attn_vs_ffn')
    plt.close()

    print(f"\n{'='*70}")
    print(f"FFN total: {ffn_total:.3f}, Attn total: {attn_total:.3f}")
    print(f"Driver: {driver}")
    print(f"FFN critical: L{ffn_critical}, Attn critical: L{attn_critical}")
    print(f"{'='*70}")

    save_results('phase106_attn_vs_ffn', {
        'experiment': 'Attention vs FFN Contribution',
        'baseline_eta': float(baseline_eta),
        'ffn_impacts': ffn_impacts,
        'attn_impacts': attn_impacts,
        'summary': {
            'ffn_total': float(ffn_total),
            'attn_total': float(attn_total),
            'driver': driver,
            'ffn_critical': int(ffn_critical),
            'attn_critical': int(attn_critical),
            'ffn_pre': float(ffn_pre),
            'ffn_post': float(ffn_post),
            'attn_pre': float(attn_pre),
            'attn_post': float(attn_post),
        }
    })


if __name__ == '__main__':
    main()
