# -*- coding: utf-8 -*-
"""
Phase 233: Scaling Laws of Thermodynamic Variables
=====================================================
How do key thermodynamic quantities scale with model size?
Test across Qwen2.5 family: 0.5B, 1.5B, 3B, 7B
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_any_model, get_model_internals, save_results, save_figure

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

QWEN_SIZES = [
    ('Qwen/Qwen2.5-0.5B',  '0.5B',  0.5e9),
    ('Qwen/Qwen2.5-1.5B',  '1.5B',  1.5e9),
    ('Qwen/Qwen2.5-3B',    '3B',    3.0e9),
    ('Qwen/Qwen2.5-7B',    '7B',    7.0e9),
]


def profile_size(model, tok, device, name):
    """Measure key thermodynamic variables."""
    internals = get_model_internals(model)
    norm_layer = internals['norm']
    lm_head = internals['lm_head']
    n_layers = internals['n_layers']

    all_T, all_U, all_P1 = [], [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, U_l, P1_l = [], [], []
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
        all_T.append(T_l); all_U.append(U_l); all_P1.append(P1_l)

    n = min(len(t) for t in all_T)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_T, mean_U, mean_P1 = avg(all_T), avg(all_U), avg(all_P1)
    dT = [mean_T[i+1] - mean_T[i] for i in range(n-1)]

    # Summary statistics
    rho_S, _ = stats.spearmanr(range(n), mean_T)
    rho_P1, _ = stats.spearmanr(range(n), mean_P1)
    max_abs_dT = max(abs(x) for x in dT) if dT else 0

    # Fisher information (top-K)
    K = 200
    total_fisher = 0
    total_geo = 0
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        prev_pk = None
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            topk = probs.topk(K)
            pk = topk.values.cpu().numpy()
            pk = pk / (pk.sum() + 1e-10)
            total_fisher += float(np.sum(1.0 / (pk + 1e-10)))
            if prev_pk is not None:
                overlap = np.sum(np.sqrt(pk * prev_pk + 1e-20))
                fid = min(float(overlap), 1.0)
                total_geo += 2 * np.arccos(max(fid, -1.0))
            prev_pk = pk.copy()
    total_fisher /= len(PROMPTS)
    total_geo /= len(PROMPTS)

    return {
        'model': name,
        'n_layers': n_layers,
        'n_states': n,
        'T_final': mean_T[-1],
        'T_initial': mean_T[0],
        'T_range': max(mean_T) - min(mean_T),
        'P1_final': mean_P1[-1],
        'U_final': mean_U[-1],
        'rho_S': float(rho_S),
        'rho_P1': float(rho_P1),
        'max_abs_dT': max_abs_dT,
        'total_fisher': total_fisher,
        'total_geodesic': total_geo,
        'mean_T': mean_T,
        'mean_U': mean_U,
        'mean_P1': mean_P1,
    }


def main():
    print("=" * 70)
    print("Phase 233: Scaling Laws of Thermodynamic Variables")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    param_counts = {}

    for model_id, short, n_params in QWEN_SIZES:
        print(f"\n--- {short} ({model_id}) ---")
        try:
            model, tok = load_any_model(model_id, device=device)
            r = profile_size(model, tok, device, short)
            results[short] = r
            param_counts[short] = n_params
            print(f"  {r['n_layers']}L, T_final={r['T_final']:.3f}")
            print(f"  Geodesic={r['total_geodesic']:.3f}, Fisher={r['total_fisher']:.1f}")
            print(f"  Arrow: rho_S={r['rho_S']:.4f}, rho_P1={r['rho_P1']:.4f}")
            del model, tok
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    sizes = list(results.keys())
    N_list = [param_counts[s] for s in sizes]
    cmap = plt.cm.viridis
    size_colors = {s: cmap(i / (len(sizes)-1)) for i, s in enumerate(sizes)}

    # (a) T profiles (normalized depth)
    for s in sizes:
        r = results[s]
        x = np.linspace(0, 1, len(r['mean_T']))
        axes[0, 0].plot(x, r['mean_T'], '-', color=size_colors[s], lw=2,
                       label=f"{s} ({r['n_layers']}L)")
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) T vs Depth')
    axes[0, 0].legend(fontsize=7)

    # (b) P1 profiles
    for s in sizes:
        r = results[s]
        x = np.linspace(0, 1, len(r['mean_P1']))
        axes[0, 1].plot(x, r['mean_P1'], '-', color=size_colors[s], lw=2, label=s)
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('P1')
    axes[0, 1].set_title('(b) P1 vs Depth')
    axes[0, 1].legend(fontsize=7)

    # (c) T_final vs N
    T_finals = [results[s]['T_final'] for s in sizes]
    axes[0, 2].plot(N_list, T_finals, '-o', color='steelblue', lw=2, markersize=8)
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_xlabel('Parameters')
    axes[0, 2].set_ylabel('T_final')
    axes[0, 2].set_title('(c) Final Temperature vs Size')
    for s, n, t in zip(sizes, N_list, T_finals):
        axes[0, 2].annotate(s, (n, t), fontsize=8, ha='left')

    # (d) Geodesic vs N
    geo_vals = [results[s]['total_geodesic'] for s in sizes]
    axes[1, 0].plot(N_list, geo_vals, '-o', color='coral', lw=2, markersize=8)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Parameters')
    axes[1, 0].set_ylabel('Total Geodesic')
    axes[1, 0].set_title('(d) Geodesic Length vs Size')
    # Fit power law
    if len(N_list) >= 3:
        log_N = np.log(N_list)
        log_G = np.log(geo_vals)
        slope, intercept, r_val, _, _ = stats.linregress(log_N, log_G)
        axes[1, 0].set_title(f'(d) Geodesic vs Size (alpha={slope:.3f})')

    # (e) Arrow strength vs N
    rho_S_vals = [results[s]['rho_S'] for s in sizes]
    rho_P1_vals = [results[s]['rho_P1'] for s in sizes]
    axes[1, 1].plot(N_list, rho_S_vals, '-o', color='steelblue', lw=2, label='rho(S)')
    axes[1, 1].plot(N_list, rho_P1_vals, '-o', color='coral', lw=2, label='rho(P1)')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Parameters')
    axes[1, 1].set_ylabel('Spearman rho')
    axes[1, 1].set_title('(e) Arrow Strength vs Size')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)

    # (f) Summary
    summary = "SCALING LAWS\n\n"
    summary += f"{'Size':>6} {'T_f':>6} {'P1_f':>6} {'Geo':>7} {'rho_S':>7}\n"
    summary += "-" * 35 + "\n"
    for s in sizes:
        r = results[s]
        summary += f"{s:>6} {r['T_final']:>6.2f} {r['P1_final']:>6.3f} "
        summary += f"{r['total_geodesic']:>7.2f} {r['rho_S']:>7.3f}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 233: Scaling Laws of Thermodynamic Variables",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase233_scaling')
    plt.close()
    save_results('phase233_scaling', {'experiment': 'Scaling Laws', 'results': results,
                                      'param_counts': param_counts})


if __name__ == '__main__':
    main()
