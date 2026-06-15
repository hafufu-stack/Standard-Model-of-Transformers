# -*- coding: utf-8 -*-
"""
Phase 244: Thermodynamic Maxwell Relations
=============================================
Test if Maxwell relations hold in transformer thermodynamics.
(dT/dV)_S = -(dP/dS)_V  and other cross-derivatives.
Here V = hidden dimension participation, P = P1 (order parameter / "pressure").
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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "Chemical reactions follow conservation of mass",
    "Photosynthesis converts sunlight to chemical energy",
    "The brain contains billions of interconnected neurons",
]


def maxwell_relations(model, tok, device, model_name):
    """Test Maxwell relations in transformer thermodynamics."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_T, all_S, all_U, all_P, all_V = [], [], [], [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, S_l, U_l, P_l, V_l = [], [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            # Volume = participation ratio of hidden state
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            V_l.append(1.0 / (h_prob ** 2).sum().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)
            S_l.append(float(S) if not np.isnan(S) else 0)
        all_T.append(T_l); all_S.append(S_l); all_U.append(U_l)
        all_P.append(P_l); all_V.append(V_l)

    n = min(len(t) for t in all_T)
    avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]
    mean_T, mean_S, mean_U = avg(all_T), avg(all_S), avg(all_U)
    mean_P, mean_V = avg(all_P), avg(all_V)

    # Compute numerical derivatives
    dT = np.diff(mean_T)
    dS = np.diff(mean_S)
    dU = np.diff(mean_U)
    dP = np.diff(mean_P)
    dV = np.diff(mean_V)

    # Maxwell relation 1: (dT/dV)_S vs -(dP/dS)_V
    dT_dV = dT / (dV + 1e-10)
    dP_dS = dP / (dS + 1e-10)
    neg_dP_dS = -dP_dS

    r_maxwell1, p_maxwell1 = stats.pearsonr(dT_dV, neg_dP_dS)

    # Maxwell relation 2: (dS/dV)_T vs (dP/dT)_V
    dS_dV = dS / (dV + 1e-10)
    dP_dT = dP / (dT + 1e-10)
    r_maxwell2, p_maxwell2 = stats.pearsonr(dS_dV, dP_dT)

    # Test dU = TdS - PdV (first law)
    TdS = np.array(mean_T[:-1]) * dS
    PdV = np.array(mean_P[:-1]) * dV
    dU_predicted = TdS - PdV
    r_firstlaw, _ = stats.pearsonr(dU, dU_predicted)

    # Helmholtz free energy F = U - TS
    F = [mean_U[i] - mean_T[i] * mean_S[i] for i in range(n)]
    # Gibbs free energy G = U - TS + PV
    G = [mean_U[i] - mean_T[i] * mean_S[i] + mean_P[i] * mean_V[i] for i in range(n)]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'maxwell1': {'r': float(r_maxwell1), 'p': float(p_maxwell1)},
        'maxwell2': {'r': float(r_maxwell2), 'p': float(p_maxwell2)},
        'first_law': {'r': float(r_firstlaw)},
        'mean_T': mean_T, 'mean_S': mean_S, 'mean_U': mean_U,
        'mean_P': mean_P, 'mean_V': mean_V,
        'F': F, 'G': G,
        'dT_dV': dT_dV.tolist(), 'neg_dP_dS': neg_dP_dS.tolist(),
        'dS_dV': dS_dV.tolist(), 'dP_dT': dP_dT.tolist(),
    }


def main():
    print("=" * 70)
    print("Phase 244: Maxwell Relations")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = maxwell_relations(model, tok, device, size)
        results[size] = r
        print(f"  Maxwell 1: r={r['maxwell1']['r']:.4f} (p={r['maxwell1']['p']:.4f})")
        print(f"  Maxwell 2: r={r['maxwell2']['r']:.4f} (p={r['maxwell2']['p']:.4f})")
        print(f"  First law: r={r['first_law']['r']:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Maxwell 1: dT/dV vs -dP/dS
    for size, r in results.items():
        c = colors[size]
        axes[0, 0].scatter(r['dT_dV'], r['neg_dP_dS'], color=c, s=20, alpha=0.6, label=size)
    # Diagonal
    lims = axes[0, 0].get_xlim()
    axes[0, 0].plot(lims, lims, 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('dT/dV'); axes[0, 0].set_ylabel('-dP/dS')
    r_last = results[list(results.keys())[-1]]
    axes[0, 0].set_title(f"(a) Maxwell 1 (r={r_last['maxwell1']['r']:.3f})")
    axes[0, 0].legend(fontsize=8)

    # (b) Maxwell 2: dS/dV vs dP/dT
    for size, r in results.items():
        c = colors[size]
        axes[0, 1].scatter(r['dS_dV'], r['dP_dT'], color=c, s=20, alpha=0.6, label=size)
    axes[0, 1].set_xlabel('dS/dV'); axes[0, 1].set_ylabel('dP/dT')
    axes[0, 1].set_title(f"(b) Maxwell 2 (r={r_last['maxwell2']['r']:.3f})")
    axes[0, 1].legend(fontsize=8)

    # (c) Free energies
    for size, r in results.items():
        c = colors[size]
        x = np.linspace(0, 1, len(r['F']))
        axes[0, 2].plot(x, r['F'], '-', color=c, lw=2, label=f'F ({size})')
        axes[0, 2].plot(x, r['G'], '--', color=c, lw=1.5, label=f'G ({size})')
    axes[0, 2].set_xlabel('Normalized Depth')
    axes[0, 2].set_ylabel('Free Energy')
    axes[0, 2].set_title('(c) Helmholtz (F) and Gibbs (G)')
    axes[0, 2].legend(fontsize=7)

    # (d) T, S, P, V profiles
    for size, r in results.items():
        c = colors[size]
        x = range(len(r['mean_T']))
        axes[1, 0].plot(x, r['mean_T'], '-', color=c, lw=2, label=f'T ({size})')
    ax2 = axes[1, 0].twinx()
    for size, r in results.items():
        c = colors[size]
        ax2.plot(range(len(r['mean_V'])), r['mean_V'], '--', color=c, lw=1, alpha=0.5)
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('T')
    ax2.set_ylabel('V (dashed)')
    axes[1, 0].set_title('(d) T and V Profiles')
    axes[1, 0].legend(fontsize=7)

    # (e) P vs V
    for size, r in results.items():
        c = colors[size]
        axes[1, 1].scatter(r['mean_V'], r['mean_P'], c=range(len(r['mean_V'])),
                          cmap='viridis', s=30, alpha=0.7)
        axes[1, 1].plot(r['mean_V'], r['mean_P'], '-', color=c, alpha=0.3, label=size)
    axes[1, 1].set_xlabel('V (Volume)'); axes[1, 1].set_ylabel('P (Pressure)')
    axes[1, 1].set_title('(e) P-V Diagram')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "MAXWELL RELATIONS\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Maxwell 1: r={r['maxwell1']['r']:.3f}\n"
        summary += f"  Maxwell 2: r={r['maxwell2']['r']:.3f}\n"
        summary += f"  First law: r={r['first_law']['r']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 244: Maxwell Relations in Transformer Thermodynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase244_maxwell')
    plt.close()
    save_results('phase244_maxwell', {'experiment': 'Maxwell Relations', 'results': results})


if __name__ == '__main__':
    main()
