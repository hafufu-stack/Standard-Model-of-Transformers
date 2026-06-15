# -*- coding: utf-8 -*-
"""
Phase 256: Semantic-Qubit Integration - Dual Temperature Unification
=====================================================================
The Standard Model defines T = Shannon entropy of softmax output.
Semantic-Qubit defines T_H = Boltzmann fit of |h|^2 (Hawking temperature).

This phase measures BOTH temperatures simultaneously across all layers
and tests whether they are related by a universal scaling law:
    T_output = f(T_Hawking)

If yes, the two independent research programs have discovered the
SAME underlying thermodynamic structure from different angles.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
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
    "The brain contains billions of neurons",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
    "Yesterday tomorrow forgot to remember the",
    "Seven abstract thoughts collided creating new",
    "The moon decided to become a professional",
    "Random words create unpredictable sequences when",
    "Colorless green ideas sleep furiously in",
]


def fit_boltzmann_temperature(h_np):
    """SQ-style Boltzmann temperature from hidden state."""
    energies = np.sort(h_np ** 2)[::-1]
    probs = energies / (np.sum(energies) + 1e-10)
    ranks = np.arange(1, min(len(probs), 200) + 1).astype(float)
    log_probs = np.log(probs[:len(ranks)] + 1e-15)
    valid = np.isfinite(log_probs)
    if np.sum(valid) < 5:
        return 0.0
    try:
        slope, _ = np.polyfit(ranks[valid], log_probs[valid], 1)
        T_H = -1.0 / (slope + 1e-15)
        return float(T_H)
    except Exception:
        return 0.0


def compute_participation_ratio(h_np):
    """SQ-style participation ratio from hidden state."""
    h_sq = h_np ** 2
    h_prob = h_sq / (np.sum(h_sq) + 1e-10)
    pr = 1.0 / (np.sum(h_prob ** 2) + 1e-10)
    return float(pr)


def dual_temperature(model, tok, device, model_name):
    """Measure both SM-T and SQ-T_H at every layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)

    all_T_sm = []   # SM output entropy temperature
    all_T_sq = []   # SQ Hawking temperature
    all_PR = []     # SQ participation ratio
    all_U = []      # Energy (norm)

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_sm_l, T_sq_l, PR_l, U_l = [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            h_np = h.cpu().numpy()

            # SM Temperature: Shannon entropy of output distribution
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_sm = float(S) if not np.isnan(S) else 0.0

            # SQ Hawking Temperature
            T_H = fit_boltzmann_temperature(h_np)

            # SQ Participation Ratio
            pr = compute_participation_ratio(h_np)

            # Energy
            U = float(h.norm().item())

            T_sm_l.append(T_sm)
            T_sq_l.append(T_H)
            PR_l.append(pr)
            U_l.append(U)

        all_T_sm.append(T_sm_l)
        all_T_sq.append(T_sq_l)
        all_PR.append(PR_l)
        all_U.append(U_l)

    n = min(len(t) for t in all_T_sm)
    avg = lambda d: np.array([float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)])
    mean_T_sm = avg(all_T_sm)
    mean_T_sq = avg(all_T_sq)
    mean_PR = avg(all_PR)
    mean_U = avg(all_U)

    # === Key Test 1: Correlation T_sm vs T_sq ===
    r_dual, p_dual = stats.pearsonr(mean_T_sm, mean_T_sq)
    rho_dual, _ = stats.spearmanr(mean_T_sm, mean_T_sq)

    # === Key Test 2: Linear fit T_sm = a * T_sq + b ===
    slope, intercept, r_lin, _, _ = stats.linregress(mean_T_sq, mean_T_sm)

    # === Key Test 3: Power-law fit T_sm = A * T_sq^alpha ===
    try:
        mask = (mean_T_sq > 0) & (mean_T_sm > 0)
        log_fit = np.polyfit(np.log(mean_T_sq[mask] + 1e-10),
                            np.log(mean_T_sm[mask] + 1e-10), 1)
        alpha = log_fit[0]
        A_power = np.exp(log_fit[1])
    except Exception:
        alpha, A_power = 0, 0

    # === Key Test 4: Noether charge PR * T_sm ===
    PRT_sm = mean_PR * mean_T_sm
    PRT_sm_cv = float(np.std(PRT_sm[1:]) / (np.mean(PRT_sm[1:]) + 1e-10))
    PRT_sm_mean = float(np.mean(PRT_sm[1:]))

    # === Key Test 5: Cooling law comparison ===
    # SM: does T_sm decrease with depth? (confirmed)
    rho_sm, _ = stats.spearmanr(range(n), mean_T_sm)
    # SQ: does T_sq decrease with depth?
    rho_sq, _ = stats.spearmanr(range(n), mean_T_sq)

    return {
        'model': model_name,
        'n_layers': n_layers,
        'mean_T_sm': mean_T_sm.tolist(),
        'mean_T_sq': mean_T_sq.tolist(),
        'mean_PR': mean_PR.tolist(),
        'mean_U': mean_U.tolist(),
        'PRT_sm': PRT_sm.tolist(),
        'dual_correlation': {'r': float(r_dual), 'p': float(p_dual), 'rho': float(rho_dual)},
        'linear_fit': {'slope': float(slope), 'intercept': float(intercept), 'r': float(r_lin)},
        'power_law': {'alpha': float(alpha), 'A': float(A_power)},
        'noether': {'PRT_mean': PRT_sm_mean, 'PRT_cv': PRT_sm_cv},
        'cooling_arrow': {'rho_sm': float(rho_sm), 'rho_sq': float(rho_sq)},
    }


def main():
    print("=" * 70)
    print("Phase 256: Dual Temperature Unification (SM x SQ)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = dual_temperature(model, tok, device, size)
        results[size] = r
        print(f"  Dual correlation: r={r['dual_correlation']['r']:.4f} (p={r['dual_correlation']['p']:.4f})")
        print(f"  Linear fit: T_sm = {r['linear_fit']['slope']:.4f} * T_sq + {r['linear_fit']['intercept']:.4f} (r={r['linear_fit']['r']:.4f})")
        print(f"  Power law: T_sm ~ T_sq^{r['power_law']['alpha']:.3f}")
        print(f"  Noether PRT: mean={r['noether']['PRT_mean']:.1f}, CV={r['noether']['PRT_cv']:.3f}")
        print(f"  Cooling arrow: SM rho={r['cooling_arrow']['rho_sm']:.3f}, SQ rho={r['cooling_arrow']['rho_sq']:.3f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) T_sm vs T_sq scatter
    for size, r in results.items():
        axes[0, 0].scatter(r['mean_T_sq'], r['mean_T_sm'], c=colors[size],
                          s=30, alpha=0.7, label=f"{size} (r={r['dual_correlation']['r']:.3f})")
    axes[0, 0].set_xlabel('T_Hawking (SQ)'); axes[0, 0].set_ylabel('T_entropy (SM)')
    axes[0, 0].set_title('(a) Dual Temperature Correlation')
    axes[0, 0].legend(fontsize=8)

    # (b) Both temperatures vs depth
    for size, r in results.items():
        x = np.linspace(0, 1, len(r['mean_T_sm']))
        axes[0, 1].plot(x, r['mean_T_sm'], '-', color=colors[size], lw=2, label=f'T_sm ({size})')
        ax2 = axes[0, 1].twinx()
        ax2.plot(x, r['mean_T_sq'], '--', color=colors[size], lw=1.5, alpha=0.6)
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('T_sm (solid)')
    ax2.set_ylabel('T_sq (dashed)')
    axes[0, 1].set_title('(b) Dual Temperature Profiles')
    axes[0, 1].legend(fontsize=7, loc='upper right')

    # (c) Noether PRT profile
    for size, r in results.items():
        axes[0, 2].plot(range(len(r['PRT_sm'])), r['PRT_sm'], '-o',
                       color=colors[size], lw=2, markersize=3,
                       label=f"{size} (CV={r['noether']['PRT_cv']:.3f})")
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('PR x T_sm')
    axes[0, 2].set_title('(c) Noether Charge (PR x T)')
    axes[0, 2].legend(fontsize=7)

    # (d) PR profile
    for size, r in results.items():
        axes[1, 0].plot(range(len(r['mean_PR'])), r['mean_PR'], '-',
                       color=colors[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('PR')
    axes[1, 0].set_title('(d) Participation Ratio')
    axes[1, 0].legend(fontsize=8)

    # (e) U profile (energy)
    for size, r in results.items():
        axes[1, 1].plot(range(len(r['mean_U'])), r['mean_U'], '-',
                       color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Layer'); axes[1, 1].set_ylabel('||h|| (Energy)')
    axes[1, 1].set_title('(e) Internal Energy')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = "DUAL TEMPERATURE UNIFICATION\n\n"
    for size, r in results.items():
        dc = r['dual_correlation']
        lf = r['linear_fit']
        ca = r['cooling_arrow']
        summary += f"{size}:\n"
        summary += f"  r(T_sm, T_sq) = {dc['r']:.4f}\n"
        summary += f"  T_sm ~ T_sq^{r['power_law']['alpha']:.2f}\n"
        summary += f"  PRT CV = {r['noether']['PRT_cv']:.3f}\n"
        summary += f"  Arrow: SM={ca['rho_sm']:.3f}, SQ={ca['rho_sq']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 256: Dual Temperature Unification (SM x SQ)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase256_dual_temperature')
    plt.close()
    save_results('phase256_dual_temperature', {
        'experiment': 'Dual Temperature Unification',
        'results': results,
    })


if __name__ == '__main__':
    main()
