# -*- coding: utf-8 -*-
"""
Phase 327: Emergent Gauge Symmetry -- Local vs Global Invariance
=================================================================
Does the transformer develop emergent gauge (local) symmetry?
Compare local transformations (per-layer rotation invariance)
with global transformations (same rotation across all layers).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_gauge_symmetry(model, tok, prompt, device):
    """Test for emergent gauge symmetry."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    D = model.config.hidden_size

    # 1. Global rotation invariance
    # Rotate all hidden states by the same angle
    # and check if key quantities change
    np.random.seed(42)
    # Random rotation matrix (2D subspace)
    theta = 0.1  # small angle
    R = np.eye(D)
    R[0, 0] = np.cos(theta); R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta); R[1, 1] = np.cos(theta)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    # Compute quantities before and after global rotation
    cos_sims_global = []
    for li in range(n_layers):
        h1 = out.hidden_states[li][0, -1, :].float()
        h2 = out.hidden_states[li + 1][0, -1, :].float()
        # Original cosine
        cos_orig = torch.nn.functional.cosine_similarity(
            h1.unsqueeze(0), h2.unsqueeze(0)).item()
        # Rotated cosine
        h1_r = h1 @ R_t
        h2_r = h2 @ R_t
        cos_rot = torch.nn.functional.cosine_similarity(
            h1_r.unsqueeze(0), h2_r.unsqueeze(0)).item()
        cos_sims_global.append(abs(cos_orig - cos_rot))

    global_invariance = 1.0 - float(np.mean(cos_sims_global))

    # 2. Local gauge invariance
    # Apply DIFFERENT rotations at each layer
    cos_sims_local = []
    for li in range(n_layers):
        h1 = out.hidden_states[li][0, -1, :].float()
        h2 = out.hidden_states[li + 1][0, -1, :].float()
        cos_orig = torch.nn.functional.cosine_similarity(
            h1.unsqueeze(0), h2.unsqueeze(0)).item()

        # Different rotation per layer
        theta_l = 0.1 * (li + 1)
        R_l = torch.eye(D, device=device, dtype=torch.float32)
        R_l[0, 0] = np.cos(theta_l); R_l[0, 1] = -np.sin(theta_l)
        R_l[1, 0] = np.sin(theta_l); R_l[1, 1] = np.cos(theta_l)

        theta_l2 = 0.1 * (li + 2)
        R_l2 = torch.eye(D, device=device, dtype=torch.float32)
        R_l2[0, 0] = np.cos(theta_l2); R_l2[0, 1] = -np.sin(theta_l2)
        R_l2[1, 0] = np.sin(theta_l2); R_l2[1, 1] = np.cos(theta_l2)

        h1_r = h1 @ R_l
        h2_r = h2 @ R_l2
        cos_rot = torch.nn.functional.cosine_similarity(
            h1_r.unsqueeze(0), h2_r.unsqueeze(0)).item()
        cos_sims_local.append(abs(cos_orig - cos_rot))

    local_invariance = 1.0 - float(np.mean(cos_sims_local))

    # Gauge symmetry = local invariance close to global invariance
    gauge_ratio = local_invariance / (global_invariance + 1e-10)

    return {
        'global_invariance': round(global_invariance, 6),
        'local_invariance': round(local_invariance, 6),
        'gauge_ratio': round(gauge_ratio, 4),
        'has_gauge': gauge_ratio > 0.9,
    }


def main():
    print("=" * 70)
    print("Phase 327: Emergent Gauge Symmetry")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        gauge_data = []
        for prompt in PROMPTS:
            g = measure_gauge_symmetry(model, tok, prompt, device)
            gauge_data.append(g)

        all_results[size] = {
            'global_invariance': round(float(np.mean([g['global_invariance'] for g in gauge_data])), 6),
            'local_invariance': round(float(np.mean([g['local_invariance'] for g in gauge_data])), 6),
            'gauge_ratio': round(float(np.mean([g['gauge_ratio'] for g in gauge_data])), 4),
            'has_gauge': sum(1 for g in gauge_data if g['has_gauge']) >= 3,
        }
        gs = 'YES' if all_results[size]['has_gauge'] else 'NO'
        print(f"  Global inv: {all_results[size]['global_invariance']:.6f}")
        print(f"  Local inv: {all_results[size]['local_invariance']:.6f}")
        print(f"  Gauge: {gs} (ratio={all_results[size]['gauge_ratio']:.4f})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    sizes = list(all_results.keys())

    x = np.arange(len(sizes))
    w = 0.35
    axes[0].bar(x - w/2, [all_results[s]['global_invariance'] for s in sizes], w,
               label='Global', color='#3498db')
    axes[0].bar(x + w/2, [all_results[s]['local_invariance'] for s in sizes], w,
               label='Local', color='#e74c3c')
    axes[0].set_xticks(x); axes[0].set_xticklabels(sizes)
    axes[0].set_ylabel('Invariance'); axes[0].set_title('(a) Global vs Local', fontweight='bold')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(sizes, [all_results[s]['gauge_ratio'] for s in sizes],
               color=[colors[s] for s in sizes])
    axes[1].axhline(1.0, color='gold', ls='--', lw=2, label='Exact gauge')
    axes[1].set_ylabel('Local/Global'); axes[1].set_title('(b) Gauge Ratio', fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    txt = "GAUGE SYMMETRY\n\n"
    for s in sizes:
        d = all_results[s]
        gs = 'YES' if d['has_gauge'] else 'NO'
        txt += f"{s}: {gs}\n  ratio={d['gauge_ratio']:.3f}\n\n"
    axes[2].text(0.5, 0.5, txt, ha='center', va='center',
                transform=axes[2].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                family='monospace')
    axes[2].axis('off'); axes[2].set_title('(c) Summary')

    fig.suptitle("Phase 327: Emergent Gauge Symmetry", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase327_gauge')
    plt.close()
    save_results('phase327_gauge', {'experiment': 'Gauge Symmetry', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
