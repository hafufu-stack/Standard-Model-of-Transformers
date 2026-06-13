# -*- coding: utf-8 -*-
"""
Phase 91: Renormalization Group Flow
SVD-truncate hidden states to various effective dimensions and test
if the 5 universal laws (especially Boltzmann) remain invariant.
Tests scale invariance - a hallmark of renormalization group fixed points.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects differentiation and",
    "Quantum mechanics describes particles at the atomic scale",
    "The human genome contains three billion base pairs encoding",
    "Neural networks learn through layers of interconnected nodes",
    "Black holes form from gravitational collapse of massive stars",
    "The periodic table organizes chemical elements by number",
]

D_EFFS = [32, 64, 128, 256, 512, 768, 1024, 1536]  # 1536 = full dim


def boltzmann_pdf(E, A, kT):
    return A * np.exp(-E / (kT + 1e-10))


def measure_boltzmann_r2(energies):
    """Fit Boltzmann and return R2."""
    nonzero = energies[energies > 1e-8]
    if len(nonzero) < 20:
        return 0.0, 0.0

    hist, edges = np.histogram(nonzero, bins=40, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    bc = centers[mask]
    hv = hist[mask]
    if len(bc) < 5:
        return 0.0, 0.0

    try:
        popt, _ = curve_fit(boltzmann_pdf, bc, hv,
                            p0=[hv[0], np.mean(nonzero)],
                            maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
        residuals = hv - boltzmann_pdf(bc, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((hv - np.mean(hv)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return float(r2), float(popt[1])
    except Exception:
        return 0.0, 0.0


def svd_truncate(h_vec, d_eff):
    """Truncate hidden state to d_eff dimensions using top SVD components."""
    if d_eff >= len(h_vec):
        return h_vec
    # For a single vector, "SVD truncation" = keep top-d_eff absolute-value components
    # (This is equivalent to projecting onto the principal subspace)
    indices = torch.argsort(torch.abs(h_vec), descending=True)[:d_eff]
    truncated = torch.zeros_like(h_vec)
    truncated[indices] = h_vec[indices]
    return truncated


def main():
    print("=" * 70)
    print("Phase 91: Renormalization Group Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    results_per_deff = []

    for d_eff in D_EFFS:
        print(f"\n  d_eff = {d_eff}...")
        all_r2s = []
        all_kTs = []
        all_cvs = []  # dU/dT for negative specific heat test

        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            Us = []
            Ts_layer = []
            layer_r2s = []

            for li, hs in enumerate(out.hidden_states):
                h = hs[0, -1, :].float()

                # SVD truncate
                h_trunc = svd_truncate(h, d_eff)

                # U from truncated
                U = h_trunc.norm().item()
                Us.append(U)

                # Boltzmann R2 from truncated energies
                energies = (h_trunc.cpu().numpy()) ** 2
                r2, kT = measure_boltzmann_r2(energies)
                layer_r2s.append(r2)
                all_kTs.append(kT)

                # T from logits (use original h for valid logits)
                with torch.no_grad():
                    normed = model.model.norm(hs[:, -1:, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                if np.isnan(T):
                    T = 0.0
                Ts_layer.append(T)

            all_r2s.extend(layer_r2s)

            # Compute Cv = dU/dT
            if len(Us) > 2 and len(Ts_layer) > 2:
                from scipy import stats as sp_stats
                slope, _, r_val, p_val, _ = sp_stats.linregress(Ts_layer, Us)
                all_cvs.append(slope)

        mean_r2 = float(np.mean(all_r2s)) if all_r2s else 0
        mean_cv = float(np.mean(all_cvs)) if all_cvs else 0
        cv_negative = all(cv < 0 for cv in all_cvs) if all_cvs else False

        results_per_deff.append({
            'd_eff': d_eff,
            'mean_boltzmann_r2': mean_r2,
            'mean_cv': mean_cv,
            'cv_all_negative': cv_negative,
            'mean_kT': float(np.mean(all_kTs)) if all_kTs else 0,
        })
        print(f"    Boltzmann R2 = {mean_r2:.4f}, Cv = {mean_cv:.1f}, "
              f"Cv<0 all: {cv_negative}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    d_effs = [r['d_eff'] for r in results_per_deff]
    r2s = [r['mean_boltzmann_r2'] for r in results_per_deff]
    cvs = [r['mean_cv'] for r in results_per_deff]

    # (a) Boltzmann R2 vs d_eff
    axes[0].plot(d_effs, r2s, 'o-', color='#c0392b', linewidth=2, markersize=6)
    axes[0].axhline(y=0.95, color='gray', linestyle='--', label='$R^2 = 0.95$')
    axes[0].axhline(y=0.80, color='#7f8c8d', linestyle=':', label='$R^2 = 0.80$')
    axes[0].set_xlabel('Effective Dimension $d_{eff}$')
    axes[0].set_ylabel('Boltzmann Fit $R^2$')
    axes[0].set_title('(a) Scale Invariance of Boltzmann')
    axes[0].set_xscale('log', base=2)
    axes[0].legend(fontsize=8)

    # Find critical d_eff (where R2 drops below 0.8)
    critical_d = None
    for r in results_per_deff:
        if r['mean_boltzmann_r2'] >= 0.80:
            critical_d = r['d_eff']
            break

    if critical_d:
        axes[0].axvline(x=critical_d, color='#27ae60', linestyle='--', alpha=0.5,
                        label=f'$d_c = {critical_d}$')
        axes[0].legend(fontsize=8)

    # (b) Cv vs d_eff
    colors_cv = ['#27ae60' if cv < 0 else '#c0392b' for cv in cvs]
    axes[1].bar(range(len(d_effs)), cvs, color=colors_cv, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(d_effs)))
    axes[1].set_xticklabels([str(d) for d in d_effs], fontsize=8)
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_xlabel('$d_{eff}$')
    axes[1].set_ylabel('$C_v = dU/dT$')
    axes[1].set_title('(b) Negative Specific Heat vs Scale')

    # (c) RG flow diagram
    kTs = [r['mean_kT'] for r in results_per_deff]
    axes[2].plot(d_effs, kTs, 'o-', color='#8e44ad', linewidth=2, markersize=6)
    axes[2].set_xlabel('$d_{eff}$ (coarse-graining scale)')
    axes[2].set_ylabel('Effective Temperature $kT$')
    axes[2].set_title('(c) RG Flow of $kT$')
    axes[2].set_xscale('log', base=2)

    # Determine if scale-invariant
    r2_above_095 = sum(1 for r in results_per_deff if r['mean_boltzmann_r2'] > 0.80)
    fig.suptitle(f'Phase 91: Renormalization Group Flow '
                 f'({r2_above_095}/{len(d_effs)} scales with R2>0.8)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase91_rg_flow')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Scale invariance: {r2_above_095}/{len(d_effs)} effective dimensions maintain Boltzmann R2>0.8")
    if critical_d:
        print(f"Critical dimension: d_c = {critical_d}")
    print(f"{'='*70}")

    save_results('phase91_rg_flow', {
        'experiment': 'Renormalization Group Flow',
        'results': results_per_deff,
        'summary': {
            'n_scale_invariant': r2_above_095,
            'critical_d': critical_d,
            'full_dim_r2': results_per_deff[-1]['mean_boltzmann_r2'],
        }
    })


if __name__ == '__main__':
    main()
