# -*- coding: utf-8 -*-
"""
Phase 162: Mutual Information Flow
Measure mutual information between adjacent layers to quantify
where information is created, destroyed, and transformed.
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
    print("Phase 162: Mutual Information Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # For each pair of adjacent layers, measure:
    # 1. Cosine similarity of hidden states (geometric overlap)
    # 2. KL divergence of output distributions (information change)
    # 3. Representation similarity (CKA-like metric)

    all_cos_sim = [[] for _ in range(n_layers - 1)]
    all_kl_div = [[] for _ in range(n_layers - 1)]
    all_norm_ratio = [[] for _ in range(n_layers - 1)]
    all_rank_corr = [[] for _ in range(n_layers - 1)]

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get output distributions at each layer
        layer_probs = []
        layer_hidden = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            layer_hidden.append(h)

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            layer_probs.append(probs)

        for li in range(n_layers - 1):
            h1 = layer_hidden[li]
            h2 = layer_hidden[li + 1]
            p1 = layer_probs[li]
            p2 = layer_probs[li + 1]

            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                h1.unsqueeze(0), h2.unsqueeze(0)).item()
            all_cos_sim[li].append(cos)

            # KL divergence
            kl = torch.sum(p1 * torch.log((p1 + 1e-10) / (p2 + 1e-10))).item()
            kl = max(0, min(kl, 100))
            all_kl_div[li].append(kl)

            # Norm ratio
            n1 = h1.norm().item()
            n2 = h2.norm().item()
            all_norm_ratio[li].append(n2 / (n1 + 1e-10))

            # Rank correlation of top-100 tokens
            top100_1 = torch.topk(p1, 100).indices
            top100_2 = torch.topk(p2, 100).indices
            # Jaccard similarity
            set1 = set(top100_1.cpu().numpy().tolist())
            set2 = set(top100_2.cpu().numpy().tolist())
            jaccard = len(set1 & set2) / (len(set1 | set2) + 1e-10)
            all_rank_corr[li].append(jaccard)

    # Averages
    avg_cos = [np.mean(v) if v else 0 for v in all_cos_sim]
    avg_kl = [np.mean(v) if v else 0 for v in all_kl_div]
    avg_norm = [np.mean(v) if v else 0 for v in all_norm_ratio]
    avg_rank = [np.mean(v) if v else 0 for v in all_rank_corr]

    layers = np.arange(n_layers - 1) + 0.5  # Midpoint between layers

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Cosine similarity
    axes[0,0].plot(layers, avg_cos, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_xlabel('Layer Transition')
    axes[0,0].set_ylabel('Cosine Similarity')
    axes[0,0].set_title('(a) Hidden State Continuity')
    axes[0,0].legend()

    # (b) KL divergence
    axes[0,1].plot(layers, avg_kl, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer Transition')
    axes[0,1].set_ylabel('KL Divergence')
    axes[0,1].set_title('(b) Information Change')

    # (c) Norm ratio
    axes[0,2].plot(layers, avg_norm, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=1, color='gray', linewidth=0.5)
    axes[0,2].set_xlabel('Layer Transition')
    axes[0,2].set_ylabel('$||h_{l+1}|| / ||h_l||$')
    axes[0,2].set_title('(c) Norm Growth')

    # (d) Rank correlation (Jaccard of top-100)
    axes[1,0].plot(layers, avg_rank, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer Transition')
    axes[1,0].set_ylabel('Jaccard (top-100)')
    axes[1,0].set_title('(d) Token Rank Stability')

    # (e) Information flow rate (1 - cosine as "novelty")
    novelty = [1 - c for c in avg_cos]
    nov_colors = ['#c0392b' if n > 0.05 else '#27ae60' for n in novelty]
    axes[1,1].bar(layers, novelty, color=nov_colors, alpha=0.7, width=0.8, edgecolor='black')
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Novelty (1 - cos)')
    axes[1,1].set_title('(e) Information Novelty')

    # (f) Summary
    max_kl_layer = np.argmax(avg_kl)
    max_novelty_layer = np.argmax(novelty)
    pre_kl = np.mean(avg_kl[:20])
    post_kl = np.mean(avg_kl[20:])
    summary = (
        f"Mutual Information Flow\n\n"
        f"Max KL divergence: L{max_kl_layer} ({avg_kl[max_kl_layer]:.3f})\n"
        f"Max novelty: L{max_novelty_layer} ({novelty[max_novelty_layer]:.4f})\n\n"
        f"Pre-L0 avg KL: {pre_kl:.3f}\n"
        f"Post-L0 avg KL: {post_kl:.3f}\n"
        f"Ratio: {post_kl/(pre_kl+1e-10):.2f}x\n\n"
        f"Information change is\n"
        f"{'LARGER' if post_kl > pre_kl else 'SMALLER'}\n"
        f"post-transition"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 162: Mutual Information Flow',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase162_mutual_info')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Max KL: L{max_kl_layer} ({avg_kl[max_kl_layer]:.3f})")
    print(f"Pre-L0 KL: {pre_kl:.3f}, Post-L0 KL: {post_kl:.3f}")
    print(f"{'='*70}")

    save_results('phase162_mutual_info', {
        'experiment': 'Mutual Information Flow',
        'summary': {
            'max_kl_layer': int(max_kl_layer),
            'pre_kl': float(pre_kl),
            'post_kl': float(post_kl),
        }
    })


if __name__ == '__main__':
    main()
