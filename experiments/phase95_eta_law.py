# -*- coding: utf-8 -*-
"""
Phase 95: eta = 1 - 1/sqrt(L) Verification
Phase 86b discovered eta = 1-1/sqrt(L) with MAE=0.020.
Verify by testing layer-truncated models (use only first N layers).
If eta changes as predicted when L changes, the law is confirmed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "The human genome encodes three billion base pairs",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Climate change affects global ecosystems",
]


def measure_eta_truncated(model, tok, device, max_layer):
    """Measure eta using only first max_layer hidden states."""
    etas = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        # Only use layers 0..max_layer
        for li in range(min(max_layer + 1, len(out.hidden_states))):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(T):
                T_vals.append(T)

        if len(T_vals) >= 4:
            T_hot = max(T_vals)
            T_cold = min(T_vals[len(T_vals)//2:])
            if T_hot > 0.01:
                eta = 1.0 - T_cold / T_hot
                etas.append(eta)

    return float(np.mean(etas)) if etas else 0.0, float(np.std(etas)) if etas else 0.0


def main():
    print("=" * 70)
    print("Phase 95: eta = 1 - 1/sqrt(L) Verification")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    total_layers = len(model.model.layers) + 1  # 29 for Qwen 1.5B

    # Test with different effective layer counts
    layer_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    results = []

    for L in layer_counts:
        if L > total_layers:
            continue
        eta_mean, eta_std = measure_eta_truncated(model, tok, device, L)
        predicted = 1.0 - 1.0 / np.sqrt(L)
        error = abs(eta_mean - predicted)
        results.append({
            'L': L,
            'eta_measured': float(eta_mean),
            'eta_std': float(eta_std),
            'eta_predicted': float(predicted),
            'error': float(error),
        })
        print(f"  L={L:2d}: measured={eta_mean:.4f}+/-{eta_std:.4f}, "
              f"predicted={predicted:.4f}, error={error:.4f}")

    # === Fit: eta = 1 - a/L^b ===
    from scipy.optimize import curve_fit

    Ls = np.array([r['L'] for r in results])
    etas_m = np.array([r['eta_measured'] for r in results])

    def power_law(L, a, b):
        return 1.0 - a / (L ** b)

    try:
        popt, pcov = curve_fit(power_law, Ls, etas_m, p0=[1.0, 0.5], maxfev=5000)
        fit_a, fit_b = popt
        fit_pred = power_law(Ls, *popt)
        residuals = etas_m - fit_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((etas_m - np.mean(etas_m))**2)
        fit_r2 = 1 - ss_res / (ss_tot + 1e-10)
        print(f"\n  Fitted: eta = 1 - {fit_a:.4f}/L^{fit_b:.4f} (R2={fit_r2:.4f})")
    except Exception:
        fit_a, fit_b, fit_r2 = 1.0, 0.5, 0.0
        fit_pred = 1.0 - 1.0 / np.sqrt(Ls)

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Measured vs predicted
    axes[0].errorbar(Ls, etas_m, yerr=[r['eta_std'] for r in results],
                     fmt='o', color='#c0392b', markersize=6, capsize=3, label='Measured')
    L_smooth = np.linspace(3, 30, 100)
    axes[0].plot(L_smooth, 1.0 - 1.0/np.sqrt(L_smooth), '--', color='#2980b9',
                 linewidth=2, label='$\\eta = 1 - 1/\\sqrt{L}$')
    axes[0].plot(L_smooth, power_law(L_smooth, fit_a, fit_b), ':', color='#27ae60',
                 linewidth=2, label=f'Fit: $1 - {fit_a:.2f}/L^{{{fit_b:.2f}}}$')
    axes[0].set_xlabel('Effective Layer Count $L$')
    axes[0].set_ylabel('Carnot Efficiency $\\eta$')
    axes[0].set_title('(a) $\\eta$ vs Layer Count')
    axes[0].legend(fontsize=8)

    # (b) Residuals
    preds_sqrt = [1.0 - 1.0/np.sqrt(r['L']) for r in results]
    residuals_sqrt = [r['eta_measured'] - p for r, p in zip(results, preds_sqrt)]
    axes[1].bar(range(len(results)), residuals_sqrt, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels([str(r['L']) for r in results], fontsize=8)
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_xlabel('$L$')
    axes[1].set_ylabel('Residual ($\\eta_{measured} - \\eta_{predicted}$)')
    mae = np.mean([abs(r) for r in residuals_sqrt])
    axes[1].set_title(f'(b) Residuals (MAE={mae:.4f})')

    # (c) Predicted vs measured scatter
    pred_vals = [r['eta_predicted'] for r in results]
    meas_vals = [r['eta_measured'] for r in results]
    axes[2].scatter(pred_vals, meas_vals, s=100, c='#8e44ad', edgecolors='black', zorder=5)
    mn = min(min(pred_vals), min(meas_vals)) - 0.05
    mx = max(max(pred_vals), max(meas_vals)) + 0.05
    axes[2].plot([mn, mx], [mn, mx], 'k--', alpha=0.3, label='Perfect')
    from scipy import stats as sp_stats
    r_corr, p_corr = sp_stats.pearsonr(pred_vals, meas_vals)
    axes[2].set_xlabel('Predicted $\\eta = 1 - 1/\\sqrt{L}$')
    axes[2].set_ylabel('Measured $\\eta$')
    axes[2].set_title(f'(c) Pred vs Meas ($r={r_corr:.3f}$, $p={p_corr:.1e}$)')
    axes[2].legend()

    fig.suptitle(f'Phase 95: $\\eta = 1 - 1/\\sqrt{{L}}$ Verification '
                 f'(fit: $b={fit_b:.3f}$, $R^2={fit_r2:.4f}$)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase95_eta_law')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Hypothesis: eta = 1 - 1/sqrt(L)")
    print(f"Fitted exponent b = {fit_b:.4f} (theoretical = 0.5)")
    print(f"Fit R2 = {fit_r2:.4f}")
    print(f"Correlation: r = {r_corr:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"VERDICT: {'CONFIRMED' if fit_r2 > 0.8 and abs(fit_b - 0.5) < 0.2 else 'NEEDS MORE DATA'}")
    print(f"{'='*70}")

    save_results('phase95_eta_law', {
        'experiment': 'eta = 1 - 1/sqrt(L) Verification',
        'results': results,
        'fit': {
            'a': float(fit_a),
            'b': float(fit_b),
            'r2': float(fit_r2),
            'formula': f'eta = 1 - {fit_a:.4f}/L^{fit_b:.4f}',
        },
        'correlation': {'r': float(r_corr), 'p': float(p_corr)},
        'summary': {
            'mae': float(mae),
            'fit_b': float(fit_b),
            'fit_r2': float(fit_r2),
            'confirmed': bool(fit_r2 > 0.8 and abs(fit_b - 0.5) < 0.2),
        }
    })


if __name__ == '__main__':
    main()
