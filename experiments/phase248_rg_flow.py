# -*- coding: utf-8 -*-
"""
Phase 248: Renormalization Group Flow (Depth-Space)
=====================================================
Compute RG flow by coarse-graining over layers.
Block-spin analogy: group 2, 4, 8 layers together and check
if the effective thermodynamic description remains self-similar.
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
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "Chemical reactions follow conservation of mass",
    "The brain contains billions of neurons",
    "Entropy always increases in closed systems",
]


def rg_flow(model, tok, device, model_name):
    """Compute RG flow by coarse-graining layers."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    # Get fine-grained T, P1 at every layer
    all_T, all_P1, all_U = [], [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, P1_l, U_l = [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)
        all_T.append(T_l); all_P1.append(P1_l); all_U.append(U_l)

    n = min(len(t) for t in all_T)
    avg = lambda d: np.array([np.mean([d[p][i] for p in range(len(d))]) for i in range(n)])
    mean_T = avg(all_T)
    mean_P1 = avg(all_P1)
    mean_U = avg(all_U)

    # Coarse-grain at multiple block sizes
    block_sizes = [1, 2, 4]
    if n >= 16:
        block_sizes.append(8)
    
    rg_levels = {}
    for bs in block_sizes:
        n_blocks = n // bs
        if n_blocks < 3:
            continue
        cg_T = [float(np.mean(mean_T[i*bs:(i+1)*bs])) for i in range(n_blocks)]
        cg_P1 = [float(np.mean(mean_P1[i*bs:(i+1)*bs])) for i in range(n_blocks)]
        cg_U = [float(np.mean(mean_U[i*bs:(i+1)*bs])) for i in range(n_blocks)]
        
        # Compute RG-invariant quantities
        # Spearman correlation (arrow strength)
        rho_S, _ = stats.spearmanr(range(n_blocks), cg_T)
        rho_P1, _ = stats.spearmanr(range(n_blocks), cg_P1)
        
        # Slope (dT/dl)
        slope_T = float(np.polyfit(range(n_blocks), cg_T, 1)[0])
        slope_P1 = float(np.polyfit(range(n_blocks), cg_P1, 1)[0])
        
        rg_levels[bs] = {
            'block_size': bs,
            'n_blocks': n_blocks,
            'cg_T': cg_T, 'cg_P1': cg_P1, 'cg_U': cg_U,
            'rho_S': float(rho_S), 'rho_P1': float(rho_P1),
            'slope_T': slope_T, 'slope_P1': slope_P1,
            'T_range': [float(min(cg_T)), float(max(cg_T))],
        }

    # Self-similarity: do rho_S, slope_T change under RG?
    rho_by_bs = [(bs, rg_levels[bs]['rho_S']) for bs in sorted(rg_levels.keys())]
    slope_by_bs = [(bs, rg_levels[bs]['slope_T']) for bs in sorted(rg_levels.keys())]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'n_points': n,
        'rg_levels': rg_levels,
        'rho_by_bs': rho_by_bs,
        'slope_by_bs': slope_by_bs,
        'fine_T': mean_T.tolist(),
        'fine_P1': mean_P1.tolist(),
    }


def main():
    print("=" * 70)
    print("Phase 248: RG Flow (Depth-Space)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = rg_flow(model, tok, device, size)
        results[size] = r
        print(f"  RG levels: {list(r['rg_levels'].keys())}")
        for bs, level in r['rg_levels'].items():
            print(f"    bs={bs}: rho_S={level['rho_S']:.3f}, slope_T={level['slope_T']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    bs_colors = {1: '#2c3e50', 2: '#e74c3c', 4: '#2ecc71', 8: '#f39c12'}

    # (a) T at different coarse-graining levels (1.5B)
    r15 = results[list(results.keys())[-1]]
    for bs, level in r15['rg_levels'].items():
        x = np.linspace(0, 1, len(level['cg_T']))
        axes[0, 0].plot(x, level['cg_T'], '-o', color=bs_colors.get(bs, 'gray'),
                       lw=2, markersize=4, label=f'bs={bs}')
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) T at Different RG Scales')
    axes[0, 0].legend(fontsize=7)

    # (b) P1 at different scales
    for bs, level in r15['rg_levels'].items():
        x = np.linspace(0, 1, len(level['cg_P1']))
        axes[0, 1].plot(x, level['cg_P1'], '-o', color=bs_colors.get(bs, 'gray'),
                       lw=2, markersize=4, label=f'bs={bs}')
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('P1')
    axes[0, 1].set_title('(b) P1 at Different RG Scales')
    axes[0, 1].legend(fontsize=7)

    # (c) RG flow: rho_S vs block_size
    for size, r in results.items():
        bs_list = [x[0] for x in r['rho_by_bs']]
        rho_list = [x[1] for x in r['rho_by_bs']]
        axes[0, 2].plot(bs_list, rho_list, '-o', color=colors[size], lw=2, 
                       markersize=6, label=size)
    axes[0, 2].set_xlabel('Block Size')
    axes[0, 2].set_ylabel('rho_S (Arrow Strength)')
    axes[0, 2].set_title('(c) Arrow Under RG')
    axes[0, 2].legend(fontsize=8)

    # (d) Slope stability
    for size, r in results.items():
        bs_list = [x[0] for x in r['slope_by_bs']]
        slope_list = [x[1] for x in r['slope_by_bs']]
        axes[1, 0].plot(bs_list, slope_list, '-o', color=colors[size], lw=2,
                       markersize=6, label=size)
    axes[1, 0].set_xlabel('Block Size')
    axes[1, 0].set_ylabel('Slope dT/dl')
    axes[1, 0].set_title('(d) Slope Under RG')
    axes[1, 0].legend(fontsize=8)

    # (e) Rescaled comparison: collapse?
    for bs, level in r15['rg_levels'].items():
        T_rescaled = (np.array(level['cg_T']) - np.min(level['cg_T'])) / (np.max(level['cg_T']) - np.min(level['cg_T']) + 1e-10)
        x = np.linspace(0, 1, len(T_rescaled))
        axes[1, 1].plot(x, T_rescaled, '-', color=bs_colors.get(bs, 'gray'),
                       lw=2, label=f'bs={bs}')
    axes[1, 1].set_xlabel('Rescaled Depth')
    axes[1, 1].set_ylabel('Rescaled T')
    axes[1, 1].set_title('(e) Rescaled T (Collapse Test)')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    summary = "RG FLOW\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        for bs, level in r['rg_levels'].items():
            summary += f"  bs={bs}: rho={level['rho_S']:.3f}\n"
        summary += "\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 248: Renormalization Group Flow",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase248_rg_flow')
    plt.close()
    save_results('phase248_rg_flow', {'experiment': 'RG Flow', 'results': results})


if __name__ == '__main__':
    main()
