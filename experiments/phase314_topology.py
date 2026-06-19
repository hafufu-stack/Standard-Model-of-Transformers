# -*- coding: utf-8 -*-
"""
Phase 314: Topological Invariants -- Winding Numbers & Berry Phase
===================================================================
Topological invariants are quantities that don't change under smooth
deformations. For transformers:
- Berry phase: geometric phase acquired by hidden states
- Winding number: how many times the state "wraps around"
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


def compute_topology(model, tok, prompt, device):
    """Compute topological invariants."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Project hidden states to 2D for winding number
    h_list = [out.hidden_states[li][0, -1, :].float().cpu().numpy() for li in range(n_layers + 1)]

    # PCA to 2D
    from sklearn.decomposition import PCA
    h_stack = np.stack(h_list)  # (n_layers+1, D)
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(h_stack)  # (n_layers+1, 2)

    # Winding number: integral of d(theta) / (2*pi)
    # theta = atan2(y, x) around centroid
    centroid = h_2d.mean(axis=0)
    h_centered = h_2d - centroid
    angles = np.arctan2(h_centered[:, 1], h_centered[:, 0])

    # Unwrap and compute winding number
    d_theta = np.diff(np.unwrap(angles))
    winding_number = float(np.sum(d_theta) / (2 * np.pi))

    # Berry phase: sum of phases between consecutive states
    # phi_Berry = -Im(sum log <psi_i | psi_{i+1}>)
    berry_phases = []
    for i in range(n_layers):
        h1 = torch.tensor(h_list[i], dtype=torch.float32)
        h2 = torch.tensor(h_list[i + 1], dtype=torch.float32)
        # Normalize
        h1_n = h1 / (h1.norm() + 1e-10)
        h2_n = h2 / (h2.norm() + 1e-10)
        overlap = float((h1_n * h2_n).sum())
        phase = np.arccos(np.clip(overlap, -1, 1))
        berry_phases.append(phase)

    total_berry = float(np.sum(berry_phases))

    # Chern number (2D approximation)
    # Using curvature of the 2D projection
    if len(h_2d) > 3:
        areas = []
        for i in range(len(h_2d) - 2):
            v1 = h_2d[i + 1] - h_2d[i]
            v2 = h_2d[i + 2] - h_2d[i]
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            areas.append(area)
        chern = float(np.sum(areas)) / (2 * np.pi)
    else:
        chern = 0

    return {
        'winding_number': round(winding_number, 4),
        'total_berry_phase': round(total_berry, 4),
        'berry_per_layer': [round(b, 4) for b in berry_phases],
        'chern_number': round(chern, 4),
        'pca_variance': [round(float(v), 4) for v in pca.explained_variance_ratio_],
        'h_2d': h_2d.tolist(),
    }


def main():
    print("=" * 70)
    print("Phase 314: Topological Invariants")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        topo_data = []
        for prompt in PROMPTS:
            t = compute_topology(model, tok, prompt, device)
            topo_data.append(t)

        avg_winding = float(np.mean([t['winding_number'] for t in topo_data]))
        avg_berry = float(np.mean([t['total_berry_phase'] for t in topo_data]))
        avg_chern = float(np.mean([t['chern_number'] for t in topo_data]))

        n = len(topo_data[0]['berry_per_layer'])
        avg_berry_profile = [float(np.mean([t['berry_per_layer'][i] for t in topo_data])) for i in range(n)]

        all_results[size] = {
            'avg_winding': round(avg_winding, 4),
            'avg_berry': round(avg_berry, 4),
            'avg_chern': round(avg_chern, 4),
            'berry_profile': [round(b, 4) for b in avg_berry_profile],
            'example_trajectory': topo_data[0]['h_2d'],
        }
        print(f"  Winding number: {avg_winding:.4f}")
        print(f"  Berry phase: {avg_berry:.4f}")
        print(f"  Chern number: {avg_chern:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Berry phase profile
    for size, data in all_results.items():
        axes[0, 0].plot(data['berry_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Berry Phase')
    axes[0, 0].set_title('(a) Berry Phase per Layer', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) 2D trajectory
    for size, data in all_results.items():
        traj = np.array(data['example_trajectory'])
        axes[0, 1].plot(traj[:, 0], traj[:, 1], 'o-', color=colors[size], lw=1.5,
                       markersize=4, label=size)
        axes[0, 1].plot(traj[0, 0], traj[0, 1], 's', color=colors[size], markersize=10)
        axes[0, 1].plot(traj[-1, 0], traj[-1, 1], '*', color=colors[size], markersize=14)
    axes[0, 1].set_xlabel('PC1'); axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('(b) Layer Trajectory (PCA)', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Topological invariants
    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.25
    axes[0, 2].bar(x - w, [all_results[s]['avg_winding'] for s in sizes], w,
                  label='Winding #', color='#3498db')
    axes[0, 2].bar(x, [all_results[s]['avg_chern'] for s in sizes], w,
                  label='Chern #', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_title('(c) Topological Numbers', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d-e) empty
    axes[1, 0].axis('off'); axes[1, 1].axis('off')

    txt = "TOPOLOGY\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Wind = {d['avg_winding']:.3f}\n"
        txt += f"  Berry = {d['avg_berry']:.3f}\n"
        txt += f"  Chern = {d['avg_chern']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 314: Topological Invariants -- Berry Phase & Winding Number",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase314_topology')
    plt.close()
    save_results('phase314_topology', {'experiment': 'Topological Invariants', 'results': {k: {kk: vv for kk, vv in v.items() if kk != 'example_trajectory'} for k, v in all_results.items()}})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
