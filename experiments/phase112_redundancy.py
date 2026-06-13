# -*- coding: utf-8 -*-
"""
Phase 112: Cooling Valley Redundancy Analysis
Phase 109 showed L12-18 are safest to prune. Phase 100 showed kT minimum there.
WHY are these layers redundant? Measure:
1. Residual stream cosine similarity (how much does each layer change the repr?)
2. Attention entropy (are attention patterns diffuse or focused?)
3. FFN contribution norm
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
    print("Phase 112: Cooling Valley Redundancy Analysis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 21.7

    # Collect per-layer metrics
    cos_sims = [[] for _ in range(n_layers)]  # cosine sim between input/output
    attn_entropies = [[] for _ in range(n_layers)]  # attention entropy
    ffn_norms = [[] for _ in range(n_layers)]  # FFN contribution norm
    residual_ratios = [[] for _ in range(n_layers)]  # ||delta|| / ||input||

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Hook to capture layer input/output
        layer_inputs = {}
        layer_outputs = {}
        attn_weights = {}

        hooks = []
        for li in range(n_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    layer_inputs[idx] = input[0][0, -1, :].detach().float().cpu()
                    if isinstance(output, tuple):
                        layer_outputs[idx] = output[0][0, -1, :].detach().float().cpu()
                    else:
                        layer_outputs[idx] = output[0, -1, :].detach().float().cpu()
                return hook_fn
            hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

        # Also hook attention to get weights
        attn_hooks = []
        for li in range(n_layers):
            def make_attn_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attn_weights[idx] = output[1].detach().float().cpu()
                return hook_fn
            attn_hooks.append(model.model.layers[li].self_attn.register_forward_hook(make_attn_hook(li)))

        with torch.no_grad():
            model(**inp, output_attentions=True)

        for h in hooks:
            h.remove()
        for h in attn_hooks:
            h.remove()

        for li in range(n_layers):
            if li in layer_inputs and li in layer_outputs:
                inp_vec = layer_inputs[li]
                out_vec = layer_outputs[li]
                delta = out_vec - inp_vec

                # Cosine similarity
                cos = torch.nn.functional.cosine_similarity(
                    inp_vec.unsqueeze(0), out_vec.unsqueeze(0)).item()
                cos_sims[li].append(cos)

                # Residual ratio
                ratio = delta.norm().item() / (inp_vec.norm().item() + 1e-10)
                residual_ratios[li].append(ratio)

                # FFN norm (approximated as ||output - input||)
                ffn_norms[li].append(delta.norm().item())

            if li in attn_weights:
                aw = attn_weights[li]  # (batch, heads, seq, seq)
                # Average attention entropy across heads
                aw_last = aw[0, :, -1, :]  # (heads, seq) - attention from last token
                aw_last = aw_last + 1e-10
                entropy = -(aw_last * torch.log(aw_last)).sum(dim=-1).mean().item()
                attn_entropies[li].append(entropy)

    # Average across prompts
    avg_cos = [np.mean(v) if v else 0 for v in cos_sims]
    avg_attn_ent = [np.mean(v) if v else 0 for v in attn_entropies]
    avg_ffn_norm = [np.mean(v) if v else 0 for v in ffn_norms]
    avg_res_ratio = [np.mean(v) if v else 0 for v in residual_ratios]

    layers = np.arange(n_layers)

    # Identify cooling valley
    valley_start = 10
    valley_end = 18
    valley_mask = (layers >= valley_start) & (layers <= valley_end)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cosine similarity (higher = less change = more redundant)
    colors_a = ['#f39c12' if valley_mask[i] else '#2980b9' for i in range(n_layers)]
    axes[0,0].bar(layers, avg_cos, color=colors_a, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(x=L0, color='#c0392b', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Cosine Similarity (in, out)')
    axes[0,0].set_title('(a) Representation Change')
    axes[0,0].legend(fontsize=8)

    # (b) Residual ratio
    axes[0,1].plot(layers, avg_res_ratio, 'o-', color='#8e44ad', markersize=4, linewidth=1.5)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].axvspan(valley_start, valley_end, alpha=0.15, color='#f39c12', label='Cooling Valley')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$||\\Delta h|| / ||h_{in}||$')
    axes[0,1].set_title('(b) Relative Update Magnitude')
    axes[0,1].legend(fontsize=8)

    # (c) Attention entropy
    axes[0,2].plot(layers, avg_attn_ent, 'o-', color='#27ae60', markersize=4, linewidth=1.5)
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axvspan(valley_start, valley_end, alpha=0.15, color='#f39c12')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Attention Entropy')
    axes[0,2].set_title('(c) Attention Pattern Entropy')

    # (d) FFN contribution norm
    axes[1,0].bar(layers, avg_ffn_norm, color=colors_a, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(x=L0, color='#c0392b', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$||\\Delta h||$ (FFN contribution)')
    axes[1,0].set_title('(d) FFN Contribution Norm')

    # (e) Combined "redundancy score"
    # High cos_sim + low residual_ratio + low ffn_norm = redundant
    cos_arr = np.array(avg_cos)
    res_arr = np.array(avg_res_ratio)
    ffn_arr = np.array(avg_ffn_norm)

    # Normalize each metric to [0,1]
    def norm01(x):
        mn, mx = np.min(x), np.max(x)
        return (x - mn) / (mx - mn + 1e-10)

    redundancy = norm01(cos_arr) + (1 - norm01(res_arr)) + (1 - norm01(ffn_arr))
    redundancy /= 3.0  # average

    colors_r = ['#c0392b' if r > 0.6 else '#f39c12' if r > 0.4 else '#27ae60'
                for r in redundancy]
    axes[1,1].bar(layers, redundancy, color=colors_r, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=L0, color='#2980b9', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Redundancy Score')
    axes[1,1].set_title('(e) Composite Redundancy')

    # Find most redundant layers
    sorted_red = sorted(enumerate(redundancy), key=lambda x: x[1], reverse=True)
    top_redundant = [l for l, _ in sorted_red[:5]]

    # (f) Summary
    valley_cos = np.mean(cos_arr[valley_mask])
    other_cos = np.mean(cos_arr[~valley_mask])
    valley_res = np.mean(res_arr[valley_mask])
    other_res = np.mean(res_arr[~valley_mask])

    summary = (
        f"Cooling Valley Analysis\n\n"
        f"Cosine sim (valley): {valley_cos:.4f}\n"
        f"Cosine sim (other): {other_cos:.4f}\n\n"
        f"Residual ratio (valley): {valley_res:.4f}\n"
        f"Residual ratio (other): {other_res:.4f}\n\n"
        f"Most redundant: {top_redundant}\n\n"
        f"Valley layers change repr\n"
        f"{'LESS' if valley_cos > other_cos else 'MORE'} than others"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 112: Cooling Valley Redundancy (most redundant: {top_redundant[:3]})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase112_redundancy')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Valley cos sim: {valley_cos:.4f}, Other: {other_cos:.4f}")
    print(f"Valley residual: {valley_res:.4f}, Other: {other_res:.4f}")
    print(f"Most redundant: {top_redundant}")
    print(f"{'='*70}")

    save_results('phase112_redundancy', {
        'experiment': 'Cooling Valley Redundancy',
        'cos_sim': [float(v) for v in avg_cos],
        'attn_entropy': [float(v) for v in avg_attn_ent],
        'ffn_norm': [float(v) for v in avg_ffn_norm],
        'residual_ratio': [float(v) for v in avg_res_ratio],
        'redundancy': [float(v) for v in redundancy],
        'summary': {
            'valley_cos': float(valley_cos),
            'other_cos': float(other_cos),
            'valley_res': float(valley_res),
            'other_res': float(other_res),
            'top_redundant': top_redundant,
        }
    })


if __name__ == '__main__':
    main()
