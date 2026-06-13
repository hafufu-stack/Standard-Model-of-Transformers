# -*- coding: utf-8 -*-
"""
Phase 117: Early Exit at Phase Transition
If eta converges at L0, can we exit the model early (at layer L0)
and still get acceptable predictions? This would enable 25% speedup.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

TEST_TEXTS = [
    "The theory of general relativity predicts that massive objects warp the fabric of spacetime",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy",
    "The human brain contains approximately eighty six billion neurons connected by synapses",
    "Machine learning algorithms can identify complex patterns in large datasets",
    "The periodic table organizes all known chemical elements according to atomic number",
    "Quantum entanglement allows particles to be correlated regardless of distance",
    "Climate models predict significant changes in global temperature patterns",
    "The discovery of antibiotics revolutionized medicine for bacterial infections",
    "Evolution by natural selection is the primary mechanism driving biological diversity",
    "Cryptographic algorithms protect sensitive information transmitted across networks",
]


def main():
    print("=" * 70)
    print("Phase 117: Early Exit at Phase Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)
    L0 = 22  # integer version

    # For each exit layer, measure:
    # 1. Top-1 agreement with full model
    # 2. Top-5 agreement
    # 3. KL divergence from full model output
    # 4. Effective PPL

    results = []

    for prompt in TEST_TEXTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Full model prediction
        full_logits = out.logits[0, -1, :].float()
        full_probs = torch.softmax(full_logits, dim=-1)
        full_top1 = torch.argmax(full_probs).item()
        full_top5 = set(torch.topk(full_probs, 5).indices.tolist())

        # Early exit at each layer
        per_prompt = []
        for li in range(n_layers + 1):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()

            probs = torch.softmax(logits, dim=-1)
            top1 = torch.argmax(probs).item()
            top5 = set(torch.topk(probs, 5).indices.tolist())

            # Top-1 match
            top1_match = int(top1 == full_top1)

            # Top-5 overlap
            top5_overlap = len(top5 & full_top5) / 5.0

            # KL divergence
            kl = torch.nn.functional.kl_div(
                torch.log(probs + 1e-10),
                full_probs,
                reduction='sum'
            ).item()
            if np.isnan(kl) or kl > 100:
                kl = 100.0

            per_prompt.append({
                'layer': li,
                'top1_match': top1_match,
                'top5_overlap': top5_overlap,
                'kl': kl,
            })

        results.append(per_prompt)

    # Average across prompts
    avg_top1 = []
    avg_top5 = []
    avg_kl = []

    for li in range(n_layers + 1):
        avg_top1.append(np.mean([r[li]['top1_match'] for r in results]))
        avg_top5.append(np.mean([r[li]['top5_overlap'] for r in results]))
        avg_kl.append(np.mean([r[li]['kl'] for r in results]))

    layers = np.arange(n_layers + 1)

    # Find earliest layer with >90% top-1 match
    early_90 = next((li for li in range(n_layers + 1) if avg_top1[li] >= 0.9), n_layers)
    # Find earliest with >80%
    early_80 = next((li for li in range(n_layers + 1) if avg_top1[li] >= 0.8), n_layers)

    # Speedup at L0
    speedup_L0 = (n_layers - L0) / n_layers * 100

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Top-1 accuracy
    axes[0, 0].plot(layers, avg_top1, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0={L0}$')
    axes[0, 0].axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Exit Layer')
    axes[0, 0].set_ylabel('Top-1 Match Rate')
    axes[0, 0].set_title(f'(a) Top-1 Agreement (90% at L{early_90})')
    axes[0, 0].legend()

    # (b) Top-5 overlap
    axes[0, 1].plot(layers, avg_top5, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Exit Layer')
    axes[0, 1].set_ylabel('Top-5 Overlap')
    axes[0, 1].set_title('(b) Top-5 Agreement')

    # (c) KL divergence
    axes[0, 2].plot(layers, avg_kl, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Exit Layer')
    axes[0, 2].set_ylabel('KL Divergence')
    axes[0, 2].set_title('(c) KL from Full Model')
    axes[0, 2].set_yscale('log')

    # (d) Speedup vs quality trade-off
    speedups = [(n_layers - li) / n_layers * 100 for li in range(n_layers + 1)]
    axes[1, 0].scatter(speedups, avg_top1, c=layers, cmap='coolwarm', s=60,
                       edgecolors='black', zorder=5)
    axes[1, 0].scatter([speedup_L0], [avg_top1[L0]], s=200, marker='*',
                       color='#f39c12', zorder=10, label=f'L0 ({speedup_L0:.0f}%)')
    axes[1, 0].set_xlabel('Speedup (%)')
    axes[1, 0].set_ylabel('Top-1 Accuracy')
    axes[1, 0].set_title('(d) Speed-Quality Tradeoff')
    axes[1, 0].legend()

    # (e) Quality at L0 vs other checkpoints
    checkpoints = {'L10': 10, 'L15': 15, f'L{L0}': L0, 'L25': 25, f'L{n_layers}': n_layers}
    cp_names = list(checkpoints.keys())
    cp_top1 = [avg_top1[v] if v <= n_layers else 1.0 for v in checkpoints.values()]
    cp_kl = [avg_kl[v] if v <= n_layers else 0.0 for v in checkpoints.values()]
    cp_speeds = [(n_layers - v) / n_layers * 100 for v in checkpoints.values()]

    bar_colors = ['#27ae60' if t > 0.8 else '#f39c12' if t > 0.5 else '#c0392b' for t in cp_top1]
    axes[1, 1].bar(range(len(cp_names)), cp_top1, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_xticks(range(len(cp_names)))
    axes[1, 1].set_xticklabels([f'{n}\n({s:.0f}%)' for n, s in zip(cp_names, cp_speeds)], fontsize=8)
    axes[1, 1].set_ylabel('Top-1 Accuracy')
    axes[1, 1].set_title('(e) Checkpoint Comparison')

    # (f) Summary
    summary = (
        f"Early Exit Analysis\n\n"
        f"At L0 (L{L0}):\n"
        f"  Top-1: {avg_top1[L0]:.1%}\n"
        f"  Top-5: {avg_top5[L0]:.1%}\n"
        f"  KL: {avg_kl[L0]:.3f}\n"
        f"  Speedup: {speedup_L0:.0f}%\n\n"
        f"90% top-1 at: L{early_90} ({(n_layers-early_90)/n_layers*100:.0f}% speedup)\n"
        f"80% top-1 at: L{early_80} ({(n_layers-early_80)/n_layers*100:.0f}% speedup)\n\n"
        f"Phase transition predicts\n"
        f"early exit: {'YES' if avg_top1[L0] >= 0.7 else 'NO'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 117: Early Exit (L{L0}: {avg_top1[L0]:.0%} top-1, '
                 f'{speedup_L0:.0f}% speedup)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase117_early_exit')
    plt.close()

    print(f"\n{'='*70}")
    print(f"At L{L0}: top-1={avg_top1[L0]:.1%}, top-5={avg_top5[L0]:.1%}, "
          f"KL={avg_kl[L0]:.3f}, speedup={speedup_L0:.0f}%")
    print(f"90% top-1 at L{early_90}, 80% at L{early_80}")
    print(f"{'='*70}")

    save_results('phase117_early_exit', {
        'experiment': 'Early Exit at Phase Transition',
        'avg_top1': [float(v) for v in avg_top1],
        'avg_top5': [float(v) for v in avg_top5],
        'avg_kl': [float(v) for v in avg_kl],
        'summary': {
            'L0_top1': float(avg_top1[L0]),
            'L0_top5': float(avg_top5[L0]),
            'L0_kl': float(avg_kl[L0]),
            'L0_speedup': float(speedup_L0),
            'early_90_layer': int(early_90),
            'early_80_layer': int(early_80),
        }
    })


if __name__ == '__main__':
    main()
