# -*- coding: utf-8 -*-
"""
Phase 297: Drag Coefficient Near Sonic Barrier
=================================================
In aerodynamics, drag spikes at Mach=1 (wave drag).
Measure the analogous "drag" in transformer information flow:
- How much information is lost/dissipated at each layer?
- Does dissipation spike at the Mach=1 transition?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_layer_drag(model, tok, prompt, device):
    """Measure information dissipation (drag) at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Drag = how much the hidden state changes per layer
    # High drag = large change = resistance to smooth flow
    layer_drag = []
    layer_entropy_change = []
    layer_rank_change = []

    for li in range(n_layers):
        h_before = out.hidden_states[li][0].float()
        h_after = out.hidden_states[li + 1][0].float()

        # 1. Cosine distance = 1 - cos(h_before, h_after) for last token
        cos = torch.nn.functional.cosine_similarity(
            h_before[-1:], h_after[-1:]).item()
        drag = 1.0 - cos

        # 2. Change in SVD entropy
        _, s_b, _ = torch.linalg.svd(h_before, full_matrices=False)
        _, s_a, _ = torch.linalg.svd(h_after, full_matrices=False)

        s_b_norm = s_b / (s_b.sum() + 1e-10)
        s_a_norm = s_a / (s_a.sum() + 1e-10)

        ent_b = -(s_b_norm * torch.log(s_b_norm + 1e-15)).sum().item()
        ent_a = -(s_a_norm * torch.log(s_a_norm + 1e-15)).sum().item()
        d_entropy = ent_a - ent_b

        # 3. Effective rank change
        rank_b = torch.exp(torch.tensor(ent_b)).item()
        rank_a = torch.exp(torch.tensor(ent_a)).item()
        d_rank = rank_a - rank_b

        layer_drag.append(float(drag))
        layer_entropy_change.append(float(d_entropy))
        layer_rank_change.append(float(d_rank))

    return {
        'layer_drag': layer_drag,
        'layer_entropy_change': layer_entropy_change,
        'layer_rank_change': layer_rank_change,
    }


def main():
    print("=" * 70)
    print("Phase 297: Drag Coefficient Near Sonic Barrier")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B', '7B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        drags = []
        for prompt in PROMPTS:
            d = measure_layer_drag(model, tok, prompt, device)
            drags.append(d)

        n = len(drags[0]['layer_drag'])
        avg_drag = [float(np.mean([d['layer_drag'][i] for d in drags])) for i in range(n)]
        avg_dent = [float(np.mean([d['layer_entropy_change'][i] for d in drags])) for i in range(n)]
        avg_drank = [float(np.mean([d['layer_rank_change'][i] for d in drags])) for i in range(n)]

        # Find drag peak (wave drag)
        max_drag_layer = int(np.argmax(avg_drag))
        max_drag_value = float(max(avg_drag))

        # Total drag
        total_drag = float(np.sum(avg_drag))

        # Drag coefficient = total drag / n_layers
        Cd = total_drag / n

        all_results[size] = {
            'n_layers': n,
            'avg_drag': [round(d, 6) for d in avg_drag],
            'avg_entropy_change': [round(d, 4) for d in avg_dent],
            'avg_rank_change': [round(d, 4) for d in avg_drank],
            'max_drag_layer': max_drag_layer,
            'max_drag_value': round(max_drag_value, 6),
            'total_drag': round(total_drag, 6),
            'Cd': round(Cd, 6),
            'peak_at_relative_depth': round(max_drag_layer / n, 4),
        }
        print(f"  Peak drag: L{max_drag_layer} ({max_drag_value:.6f})")
        print(f"  Total drag: {total_drag:.4f}")
        print(f"  Cd = {Cd:.6f}")
        print(f"  Peak at depth: {max_drag_layer/n:.2f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) Drag profiles
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_drag'], '-', color=colors[size], lw=2, label=size)
        axes[0, 0].axvline(data['max_drag_layer'], color=colors[size], ls='--', alpha=0.3)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Drag (1 - cos similarity)')
    axes[0, 0].set_title('(a) Information Drag Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Normalized depth
    for size, data in all_results.items():
        n = len(data['avg_drag'])
        x = np.linspace(0, 1, n)
        axes[0, 1].plot(x, data['avg_drag'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('Drag')
    axes[0, 1].set_title('(b) Drag vs Normalized Depth', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Drag coefficient vs model size
    sizes = list(all_results.keys())
    cds = [all_results[s]['Cd'] for s in sizes]
    axes[0, 2].bar(sizes, cds, color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Drag Coefficient Cd')
    axes[0, 2].set_title('(c) Cd vs Scale', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # (d) Entropy change profile
    for size, data in all_results.items():
        axes[1, 0].plot(data['avg_entropy_change'], '-', color=colors[size], lw=2, label=size)
    axes[1, 0].axhline(0, color='black', ls='-', lw=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Entropy Change')
    axes[1, 0].set_title('(d) SVD Entropy Change per Layer', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Peak drag depth vs model
    peak_depths = [all_results[s]['peak_at_relative_depth'] for s in sizes]
    axes[1, 1].bar(sizes, peak_depths, color=[colors[s] for s in sizes])
    axes[1, 1].set_ylabel('Relative Depth of Peak Drag')
    axes[1, 1].set_title('(e) Where Does Drag Peak?', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "DRAG COEFFICIENT\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Cd = {d['Cd']:.6f}\n"
        txt += f"  Peak: L{d['max_drag_layer']} (depth={d['peak_at_relative_depth']:.2f})\n\n"
    txt += "Analogy: wave drag\n"
    txt += "peaks at Mach = 1"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 297: Drag Coefficient Near Sonic Barrier",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase297_drag_coefficient')
    plt.close()

    save_results('phase297_drag_coefficient', {
        'experiment': 'Drag Coefficient Near Sonic Barrier',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
