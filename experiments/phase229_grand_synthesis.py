# -*- coding: utf-8 -*-
"""
Phase 229: Grand Synthesis - The Thermodynamic Identity of Transformers
=========================================================================
Integrate ALL findings from Seasons 15-17+ into a unified framework.
Compute all key metrics in one pass for maximum consistency.
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
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
    "Superconductors have zero electrical resistance",
    "The brain contains billions of neurons",
    "Algorithms determine computational complexity",
    "Chemical reactions follow conservation of mass",
    "Stars form from collapsing molecular clouds",
]


def grand_synthesis(model, tok, device, model_name):
    """Comprehensive single-pass measurement of all thermodynamic quantities."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    K = 200

    # Collect everything per prompt
    all_data = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layers = []
        prev_probs_topk = None
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = h.norm().item()
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / ((h_prob ** 2).sum().item() + 1e-10)
            S_hidden = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()
            E_var = float(h_sq.var().item())

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            P1 = probs.max().item()
            T = S if not np.isnan(S) else 0

            # Fisher & geodesic
            topk = probs.topk(K)
            p_topk = topk.values.cpu().numpy()
            p_topk = p_topk / (p_topk.sum() + 1e-10)
            fisher_trace = float(np.sum(1.0 / (p_topk + 1e-10)))

            geo_dist = 0
            fidelity = 1
            if prev_probs_topk is not None:
                overlap = np.sum(np.sqrt(p_topk * prev_probs_topk + 1e-20))
                fidelity = min(float(overlap), 1.0)
                geo_dist = 2 * np.arccos(fidelity)
            prev_probs_topk = p_topk.copy()

            layers.append({
                'U': U, 'T': T, 'S': float(S) if not np.isnan(S) else 0,
                'P1': float(P1), 'PR': PR, 'S_hidden': S_hidden,
                'E_var': E_var, 'fisher': fisher_trace,
                'geo_dist': geo_dist, 'fidelity': fidelity,
            })
        all_data.append(layers)

    n = min(len(d) for d in all_data)
    var_keys = ['U', 'T', 'S', 'P1', 'PR', 'S_hidden', 'E_var', 'fisher', 'geo_dist', 'fidelity']

    # Compute means
    means = {k: [float(np.mean([all_data[p][l][k] for p in range(len(PROMPTS))]))
                 for l in range(n)] for k in var_keys}

    # Compute stds
    stds = {k: [float(np.std([all_data[p][l][k] for p in range(len(PROMPTS))]))
                for l in range(n)] for k in var_keys}

    # Derivatives
    dT = [means['T'][i+1] - means['T'][i] for i in range(n-1)]
    dU = [means['U'][i+1] - means['U'][i] for i in range(n-1)]

    # Key transition points
    abs_dT = [abs(x) for x in dT]
    L_ignition = int(np.argmax(abs_dT))

    # Crystallization point (where P1 crosses 0.5 or T drops below median)
    T_med = np.median(means['T'][1:])
    L_crystal = 0
    for i in range(n-1, 0, -1):
        if means['T'][i] > T_med:
            L_crystal = i
            break

    # Correlation matrix of all observables
    corr_matrix = np.corrcoef([means[k] for k in var_keys])
    corr_labels = var_keys

    # Arrow of depth
    rho_S, p_S = stats.spearmanr(range(n), means['S'])
    rho_P1, p_P1 = stats.spearmanr(range(n), means['P1'])

    # Total geodesic
    total_geo = sum(means['geo_dist'])

    # Efficiency: eta = -dS/dU at each layer
    eta = [-dT[i] / (abs(dU[i]) + 1e-10) for i in range(len(dT))]

    # Summary statistics
    summary_stats = {
        'L_ignition': L_ignition,
        'L_crystal': L_crystal,
        'rho_S': float(rho_S),
        'p_S': float(p_S),
        'rho_P1': float(rho_P1),
        'total_geodesic': total_geo,
        'mean_eta': float(np.mean(eta)),
        'T_initial': means['T'][0],
        'T_final': means['T'][-1],
        'P1_initial': means['P1'][0],
        'P1_final': means['P1'][-1],
        'U_initial': means['U'][0],
        'U_final': means['U'][-1],
    }

    return {
        'model': model_name,
        'n_layers': n_layers,
        'means': means,
        'stds': stds,
        'dT': [float(x) for x in dT],
        'dU': [float(x) for x in dU],
        'eta': [float(x) for x in eta],
        'corr_matrix': corr_matrix.tolist(),
        'corr_labels': corr_labels,
        'summary': summary_stats,
    }


def main():
    print("=" * 70)
    print("Phase 229: Grand Synthesis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = grand_synthesis(model, tok, device, size)
        results[size] = r
        s = r['summary']
        print(f"  Ignition: L{s['L_ignition']}, Crystal: L{s['L_crystal']}")
        print(f"  Arrow: rho(S)={s['rho_S']:.4f}, rho(P1)={s['rho_P1']:.4f}")
        print(f"  Geodesic: {s['total_geodesic']:.3f}")
        print(f"  T: {s['T_initial']:.2f} -> {s['T_final']:.2f}")
        print(f"  P1: {s['P1_initial']:.4f} -> {s['P1_final']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization: The Grand Synthesis Figure ===
    fig = plt.figure(figsize=(20, 16))

    # Row 1: Core profiles
    ax1 = fig.add_subplot(3, 4, 1)
    ax2 = fig.add_subplot(3, 4, 2)
    ax3 = fig.add_subplot(3, 4, 3)
    ax4 = fig.add_subplot(3, 4, 4)

    # Row 2: Derived quantities
    ax5 = fig.add_subplot(3, 4, 5)
    ax6 = fig.add_subplot(3, 4, 6)
    ax7 = fig.add_subplot(3, 4, 7)
    ax8 = fig.add_subplot(3, 4, 8)

    # Row 3: Correlation + Phase diagram + Summary
    ax9 = fig.add_subplot(3, 4, 9)
    ax10 = fig.add_subplot(3, 4, 10)
    ax11 = fig.add_subplot(3, 4, 11)
    ax12 = fig.add_subplot(3, 4, 12)

    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (1) Temperature
    for size, r in results.items():
        ax1.plot(range(len(r['means']['T'])), r['means']['T'],
                '-', color=colors[size], lw=2, label=size)
        ax1.fill_between(range(len(r['means']['T'])),
                        [m-s for m,s in zip(r['means']['T'], r['stds']['T'])],
                        [m+s for m,s in zip(r['means']['T'], r['stds']['T'])],
                        color=colors[size], alpha=0.1)
    ax1.set_xlabel('Layer'); ax1.set_ylabel('T')
    ax1.set_title('Temperature'); ax1.legend(fontsize=7)

    # (2) Internal Energy
    for size, r in results.items():
        ax2.plot(range(len(r['means']['U'])), r['means']['U'],
                '-', color=colors[size], lw=2, label=size)
    ax2.set_xlabel('Layer'); ax2.set_ylabel('U')
    ax2.set_title('Internal Energy'); ax2.legend(fontsize=7)

    # (3) Order Parameter P1
    for size, r in results.items():
        ax3.plot(range(len(r['means']['P1'])), r['means']['P1'],
                '-', color=colors[size], lw=2, label=size)
    ax3.set_xlabel('Layer'); ax3.set_ylabel('P1')
    ax3.set_title('Order Parameter'); ax3.legend(fontsize=7)

    # (4) Participation Ratio
    for size, r in results.items():
        ax4.plot(range(len(r['means']['PR'])), r['means']['PR'],
                '-', color=colors[size], lw=2, label=size)
    ax4.set_xlabel('Layer'); ax4.set_ylabel('PR')
    ax4.set_title('Participation Ratio'); ax4.legend(fontsize=7)

    # (5) dT/dl
    for size, r in results.items():
        ax5.plot(range(len(r['dT'])), r['dT'],
                '-', color=colors[size], lw=2, label=size)
    ax5.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax5.set_xlabel('Layer'); ax5.set_ylabel('dT/dl')
    ax5.set_title('Temperature Gradient'); ax5.legend(fontsize=7)

    # (6) Geodesic distance
    for size, r in results.items():
        ax6.plot(range(len(r['means']['geo_dist'])), r['means']['geo_dist'],
                '-', color=colors[size], lw=2, label=size)
    ax6.set_xlabel('Layer'); ax6.set_ylabel('ds')
    ax6.set_title('Geodesic Step'); ax6.legend(fontsize=7)

    # (7) Fisher information
    for size, r in results.items():
        ax7.plot(range(len(r['means']['fisher'])), r['means']['fisher'],
                '-', color=colors[size], lw=2, label=size)
    ax7.set_xlabel('Layer'); ax7.set_ylabel('Tr(F)')
    ax7.set_title('Fisher Information'); ax7.legend(fontsize=7)

    # (8) Efficiency eta
    for size, r in results.items():
        ax8.plot(range(len(r['eta'])), r['eta'],
                '-', color=colors[size], lw=2, label=size)
    ax8.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax8.set_xlabel('Layer'); ax8.set_ylabel('eta')
    ax8.set_title('Efficiency -dS/dU'); ax8.legend(fontsize=7)

    # (9) Correlation matrix (1.5B)
    r15 = results['1.5B']
    corr = np.array(r15['corr_matrix'])
    short_labels = ['U', 'T', 'S', 'P1', 'PR', 'S_h', 'E_v', 'F', 'geo', 'fid']
    im = ax9.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax9.set_xticks(range(len(short_labels))); ax9.set_xticklabels(short_labels, fontsize=6, rotation=45)
    ax9.set_yticks(range(len(short_labels))); ax9.set_yticklabels(short_labels, fontsize=6)
    ax9.set_title('Correlation (1.5B)')
    fig.colorbar(im, ax=ax9, shrink=0.7)

    # (10) Phase diagram T vs U
    for size, r in results.items():
        ax10.scatter(r['means']['U'], r['means']['T'], c=range(len(r['means']['T'])),
                    cmap='coolwarm', s=25, alpha=0.7)
        ax10.plot(r['means']['U'], r['means']['T'], '-', color=colors[size], alpha=0.3)
    ax10.set_xlabel('U'); ax10.set_ylabel('T')
    ax10.set_title('Phase Diagram')

    # (11) S vs P1
    for size, r in results.items():
        ax11.scatter(r['means']['S'], r['means']['P1'], c=range(len(r['means']['S'])),
                    cmap='viridis', s=25, alpha=0.7)
        ax11.plot(r['means']['S'], r['means']['P1'], '-', color=colors[size], alpha=0.3)
    ax11.set_xlabel('Entropy S'); ax11.set_ylabel('P1')
    ax11.set_title('S vs P1')

    # (12) Summary
    summary = "GRAND SYNTHESIS\n"
    for size, r in results.items():
        s = r['summary']
        summary += f"\n{size} ({r['n_layers']}L):\n"
        summary += f"  Ignite L{s['L_ignition']}, "
        summary += f"Crystal L{s['L_crystal']}\n"
        summary += f"  T: {s['T_initial']:.1f}->{s['T_final']:.1f}\n"
        summary += f"  P1: {s['P1_initial']:.3f}->{s['P1_final']:.3f}\n"
        summary += f"  Arrow rho={s['rho_S']:.3f}\n"
        summary += f"  Geodesic={s['total_geodesic']:.2f}\n"
    ax12.text(0.5, 0.5, summary, ha='center', va='center',
              transform=ax12.transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
              family='monospace')
    ax12.axis('off')
    ax12.set_title('Summary')

    fig.suptitle("Phase 229: Grand Synthesis - The Thermodynamic Identity of Transformers",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase229_grand_synthesis')
    plt.close()
    save_results('phase229_grand_synthesis', {'experiment': 'Grand Synthesis', 'results': results})


if __name__ == '__main__':
    main()
