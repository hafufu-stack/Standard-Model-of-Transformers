# -*- coding: utf-8 -*-
"""
Phase 158: Attention Entropy Phase Transition
Measure attention entropy at each layer to see if attention heads
undergo their own phase transition. Does attention become more
"focused" (low entropy) after L0?
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


def main():
    print("=" * 70)
    print("Phase 158: Attention Entropy Phase Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16,
        device_map=device, local_files_only=True,
        attn_implementation='eager')
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
    n_real_layers = len(model.model.layers)

    all_attn_entropy = [[] for _ in range(n_real_layers)]
    all_attn_gini = [[] for _ in range(n_real_layers)]
    all_head_entropy = [[[] for _ in range(model.config.num_attention_heads)]
                        for _ in range(n_real_layers)]

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)

        attn_weights = out.attentions  # tuple of (batch, n_heads, seq, seq)

        for li in range(min(n_real_layers, len(attn_weights))):
            attn = attn_weights[li]  # (1, n_heads, seq, seq)
            if attn is None:
                all_attn_entropy[li].append(0)
                all_attn_gini[li].append(0)
                continue

            # Attention entropy: average over heads and queries
            # For the last token's attention distribution
            last_attn = attn[0, :, -1, :]  # (n_heads, seq)
            head_entropies = []
            for hi in range(last_attn.shape[0]):
                a = last_attn[hi].float()
                a = a / (a.sum() + 1e-10)  # normalize
                H = -(a * torch.log(a + 1e-10)).sum().item()
                head_entropies.append(H if not np.isnan(H) else 0)
                if hi < model.config.num_attention_heads:
                    all_head_entropy[li][hi].append(head_entropies[-1])

            mean_entropy = np.mean(head_entropies)
            all_attn_entropy[li].append(mean_entropy)

            # Gini coefficient (attention concentration)
            sorted_a = torch.sort(last_attn.flatten().float())[0]
            n = sorted_a.shape[0]
            if n > 0:
                index = torch.arange(1, n + 1, device=device).float()
                gini = 1.0 - 2.0 * (torch.sum((n + 1 - index) * sorted_a) /
                                      (n * sorted_a.sum() + 1e-10)).item()
                gini = max(0, min(gini, 1))
            else:
                gini = 0
            all_attn_gini[li].append(gini)

    avg_entropy = [np.mean(v) if v else 0 for v in all_attn_entropy]
    avg_gini = [np.mean(v) if v else 0 for v in all_attn_gini]
    layers = np.arange(n_real_layers)

    # Per-head entropy averaged
    n_heads = model.config.num_attention_heads
    head_entropy_matrix = np.zeros((n_real_layers, n_heads))
    for li in range(n_real_layers):
        for hi in range(n_heads):
            if all_head_entropy[li][hi]:
                head_entropy_matrix[li, hi] = np.mean(all_head_entropy[li][hi])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Attention entropy vs layer
    axes[0,0].plot(layers, avg_entropy, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Attention Entropy')
    axes[0,0].set_title('(a) Attention Entropy')
    axes[0,0].legend()

    # (b) Gini coefficient
    axes[0,1].plot(layers, avg_gini, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Gini (attention concentration)')
    axes[0,1].set_title('(b) Attention Concentration')

    # (c) Per-head entropy heatmap
    im = axes[0,2].imshow(head_entropy_matrix.T, aspect='auto', cmap='hot')
    axes[0,2].axvline(x=21.7, color='cyan', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Head')
    axes[0,2].set_title('(c) Per-Head Entropy')
    plt.colorbar(im, ax=axes[0,2], label='$H$')

    # (d) Entropy gradient (dH/dL)
    dH = np.gradient(avg_entropy)
    dH_colors = ['#c0392b' if d < 0 else '#2980b9' for d in dH]
    axes[1,0].bar(layers, dH, color=dH_colors, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].axhline(y=0, color='black', linewidth=1)
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$dH/dL$')
    axes[1,0].set_title('(d) Attention Entropy Gradient')

    # (e) Pre vs Post transition comparison
    pre = np.mean(avg_entropy[:20])
    post = np.mean(avg_entropy[20:])
    axes[1,1].bar(['Pre-$L_0$', 'Post-$L_0$'], [pre, post],
                  color=['#2980b9', '#c0392b'], alpha=0.8, edgecolor='black')
    axes[1,1].set_ylabel('Mean Attention Entropy')
    drop = (post - pre) / (pre + 1e-10) * 100
    axes[1,1].set_title(f'(e) Attn Entropy: {drop:+.1f}%')

    # (f) Summary
    min_entropy_layer = np.argmin(avg_entropy)
    summary = (
        f"Attention Entropy Phase Transition\n\n"
        f"Mean pre-L0 entropy: {pre:.3f}\n"
        f"Mean post-L0 entropy: {post:.3f}\n"
        f"Change: {drop:+.1f}%\n\n"
        f"Min entropy layer: L{min_entropy_layer}\n"
        f"Max Gini layer: L{np.argmax(avg_gini)}\n\n"
        f"Attention {'FOCUSES' if drop < -5 else 'does NOT change'}\n"
        f"after phase transition"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 158: Attention Entropy Phase Transition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase158_attention')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-L0 attention entropy: {pre:.3f}")
    print(f"Post-L0 attention entropy: {post:.3f}")
    print(f"Change: {drop:+.1f}%")
    print(f"{'='*70}")

    save_results('phase158_attention', {
        'experiment': 'Attention Entropy Phase Transition',
        'summary': {
            'pre_entropy': float(pre),
            'post_entropy': float(post),
            'change_pct': float(drop),
            'min_entropy_layer': int(min_entropy_layer),
        }
    })


if __name__ == '__main__':
    main()
