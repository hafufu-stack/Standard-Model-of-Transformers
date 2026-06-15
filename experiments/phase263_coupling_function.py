# -*- coding: utf-8 -*-
"""
Phase 263: Dual Temperature Coupling Function T_sm = f(T_H)
=============================================================
Phase 256 showed T_sm and T_H are anti-correlated (r ~ -0.6).
This phase identifies the EXACT functional form of the coupling:
  - Linear: T_sm = a - b*T_H
  - Power: T_sm = A * T_H^alpha
  - Logarithmic: T_sm = a - b*ln(T_H)
  - Exponential: T_sm = a * exp(-b*T_H)

Also investigates:
- Does the coupling function depend on model size?
- Layer-resolved analysis (early vs mid vs late layers)
- Residuals and goodness-of-fit
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
    "General relativity describes gravity as spacetime curvature",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "Machine learning discovers hidden patterns",
    "Stars form from collapsing molecular clouds",
    "The brain contains billions of neurons",
    "Purple elephants calculated the square root of",
    "Colorless green ideas sleep furiously in",
]


def fit_boltzmann_T(h_np):
    """SQ Hawking temperature from hidden state."""
    energies = np.sort(h_np ** 2)[::-1]
    probs = energies / (np.sum(energies) + 1e-10)
    ranks = np.arange(1, min(len(probs), 200) + 1).astype(float)
    log_probs = np.log(probs[:len(ranks)] + 1e-15)
    valid = np.isfinite(log_probs)
    if np.sum(valid) < 5:
        return 0.0
    try:
        slope, _ = np.polyfit(ranks[valid], log_probs[valid], 1)
        return float(-1.0 / (slope + 1e-15))
    except Exception:
        return 0.0


def measure_dual_temperatures(model, tok, device, model_name):
    """Measure both T_sm and T_H at every layer, across all prompts."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_T_sm, all_T_H = [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_sm_l, T_H_l = [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            h_np = h.cpu().numpy()

            # SM Temperature
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_sm = float(S) if not np.isnan(S) else 0.0

            # SQ Temperature
            T_H = fit_boltzmann_T(h_np)

            T_sm_l.append(T_sm)
            T_H_l.append(T_H)

        all_T_sm.append(T_sm_l)
        all_T_H.append(T_H_l)

    n = min(len(t) for t in all_T_sm)
    avg = lambda d: np.array([float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)])
    mean_T_sm = avg(all_T_sm)
    mean_T_H = avg(all_T_H)

    return mean_T_sm, mean_T_H, n


def fit_coupling(T_sm, T_H):
    """Fit multiple functional forms and select the best."""
    # Skip embedding (layer 0)
    x = T_H[1:]
    y = T_sm[1:]

    # Remove any zeros or negatives
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 5:
        return {'best': 'insufficient_data'}

    fits = {}

    # 1. Linear: y = a + b*x
    try:
        slope, intercept, r, _, _ = stats.linregress(x, y)
        ss_res = np.sum((y - (intercept + slope * x))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        fits['linear'] = {
            'params': {'a': round(float(intercept), 4), 'b': round(float(slope), 4)},
            'r2': round(float(r2), 4),
            'formula': f'T_sm = {intercept:.3f} + {slope:.3f}*T_H',
            'residuals': (y - (intercept + slope * x)).tolist(),
        }
    except Exception:
        pass

    # 2. Power: y = A * x^alpha (fit in log space)
    try:
        log_fit = np.polyfit(np.log(x), np.log(y), 1)
        alpha = log_fit[0]
        A = np.exp(log_fit[1])
        y_pred = A * x**alpha
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        fits['power'] = {
            'params': {'A': round(float(A), 4), 'alpha': round(float(alpha), 4)},
            'r2': round(float(r2), 4),
            'formula': f'T_sm = {A:.3f} * T_H^{alpha:.3f}',
            'residuals': (y - y_pred).tolist(),
        }
    except Exception:
        pass

    # 3. Logarithmic: y = a + b*ln(x)
    try:
        log_x = np.log(x)
        slope_l, intercept_l, r_l, _, _ = stats.linregress(log_x, y)
        y_pred = intercept_l + slope_l * log_x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        fits['logarithmic'] = {
            'params': {'a': round(float(intercept_l), 4), 'b': round(float(slope_l), 4)},
            'r2': round(float(r2), 4),
            'formula': f'T_sm = {intercept_l:.3f} + {slope_l:.3f}*ln(T_H)',
            'residuals': (y - y_pred).tolist(),
        }
    except Exception:
        pass

    # 4. Exponential: y = a * exp(b*x)
    try:
        log_y = np.log(y)
        slope_e, intercept_e, r_e, _, _ = stats.linregress(x, log_y)
        a_e = np.exp(intercept_e)
        b_e = slope_e
        y_pred = a_e * np.exp(b_e * x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        fits['exponential'] = {
            'params': {'a': round(float(a_e), 4), 'b': round(float(b_e), 4)},
            'r2': round(float(r2), 4),
            'formula': f'T_sm = {a_e:.3f} * exp({b_e:.3f}*T_H)',
            'residuals': (y - y_pred).tolist(),
        }
    except Exception:
        pass

    # 5. Inverse: y = a / x + b
    try:
        inv_x = 1.0 / x
        slope_i, intercept_i, r_i, _, _ = stats.linregress(inv_x, y)
        y_pred = intercept_i + slope_i / x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        fits['inverse'] = {
            'params': {'a': round(float(slope_i), 4), 'b': round(float(intercept_i), 4)},
            'r2': round(float(r2), 4),
            'formula': f'T_sm = {slope_i:.3f}/T_H + {intercept_i:.3f}',
            'residuals': (y - y_pred).tolist(),
        }
    except Exception:
        pass

    # Select best fit
    best_name = max(fits, key=lambda k: fits[k]['r2'])
    fits['best'] = best_name
    fits['best_r2'] = fits[best_name]['r2']
    fits['best_formula'] = fits[best_name]['formula']
    fits['x'] = x.tolist()
    fits['y'] = y.tolist()

    return fits


def main():
    print("=" * 70)
    print("Phase 263: Dual Temperature Coupling Function")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        T_sm, T_H, n = measure_dual_temperatures(model, tok, device, size)
        fits = fit_coupling(T_sm, T_H)

        all_results[size] = {
            'T_sm': T_sm.tolist(),
            'T_H': T_H.tolist(),
            'fits': {k: v for k, v in fits.items() if k not in ('x', 'y', 'residuals')},
            'best_fit': fits.get('best', 'N/A'),
            'best_r2': fits.get('best_r2', 0),
            'best_formula': fits.get('best_formula', 'N/A'),
        }

        print(f"  Fit comparison (R^2):")
        for name in ['linear', 'power', 'logarithmic', 'exponential', 'inverse']:
            if name in fits:
                marker = " ***" if name == fits.get('best') else ""
                print(f"    {name:15s}: R^2={fits[name]['r2']:.4f}  {fits[name]['formula']}{marker}")
        print(f"  Best: {fits.get('best_formula', 'N/A')}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        T_sm = np.array(data['T_sm'])
        T_H = np.array(data['T_H'])
        c = colors[size]

        # (a) T_sm vs T_H scatter with best fit
        axes[0, 0].scatter(T_H[1:], T_sm[1:], c=c, s=20, alpha=0.7, label=size)
        # Plot best fit line
        fit_name = data['best_fit']
        if fit_name in data['fits'] and 'params' in data['fits'][fit_name]:
            x_fit = np.linspace(T_H[1:].min(), T_H[1:].max(), 100)
            p = data['fits'][fit_name]['params']
            if fit_name == 'linear':
                y_fit = p['a'] + p['b'] * x_fit
            elif fit_name == 'power':
                y_fit = p['A'] * x_fit**p['alpha']
            elif fit_name == 'logarithmic':
                y_fit = p['a'] + p['b'] * np.log(x_fit + 1e-10)
            elif fit_name == 'exponential':
                y_fit = p['a'] * np.exp(p['b'] * x_fit)
            elif fit_name == 'inverse':
                y_fit = p['a'] / x_fit + p['b']
            else:
                y_fit = None
            if y_fit is not None:
                axes[0, 0].plot(x_fit, y_fit, '--', color=c, lw=1.5, alpha=0.8)

    axes[0, 0].set_xlabel('T_Hawking (SQ)')
    axes[0, 0].set_ylabel('T_entropy (SM)')
    axes[0, 0].set_title('(a) Dual Temperature Coupling', fontweight='bold')
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    # (b) R^2 comparison bars
    fit_names = ['linear', 'power', 'logarithmic', 'exponential', 'inverse']
    x_pos = np.arange(len(fit_names))
    width = 0.35
    for i, (size, data) in enumerate(all_results.items()):
        r2s = [data['fits'].get(fn, {}).get('r2', 0) for fn in fit_names]
        offset = -width/2 + i * width
        bars = axes[0, 1].bar(x_pos + offset, r2s, width, color=colors[size],
                             alpha=0.8, label=size, edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(fit_names, rotation=30, fontsize=8)
    axes[0, 1].set_ylabel('R^2')
    axes[0, 1].set_title('(b) Fit Quality Comparison', fontweight='bold')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3, axis='y')

    # (c) Both temperatures vs depth
    for size, data in all_results.items():
        x = np.linspace(0, 1, len(data['T_sm']))
        c = colors[size]
        axes[0, 2].plot(x, data['T_sm'], '-', color=c, lw=2, label=f'T_sm ({size})')
        ax2 = axes[0, 2].twinx()
        ax2.plot(x, data['T_H'], '--', color=c, lw=1.5, alpha=0.6)
    axes[0, 2].set_xlabel('Normalized Depth')
    axes[0, 2].set_ylabel('T_sm (solid)')
    ax2.set_ylabel('T_H (dashed)')
    axes[0, 2].set_title('(c) Temperature Profiles', fontweight='bold')
    axes[0, 2].legend(fontsize=7, loc='upper right')
    axes[0, 2].grid(alpha=0.3)

    # (d) T_sm vs T_H in log-log
    for size, data in all_results.items():
        T_sm = np.array(data['T_sm'])[1:]
        T_H = np.array(data['T_H'])[1:]
        valid = (T_sm > 0) & (T_H > 0)
        axes[1, 0].scatter(np.log(T_H[valid]), np.log(T_sm[valid]),
                          c=colors[size], s=20, alpha=0.7, label=size)
    axes[1, 0].set_xlabel('ln(T_H)')
    axes[1, 0].set_ylabel('ln(T_sm)')
    axes[1, 0].set_title('(d) Log-Log Space', fontweight='bold')
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    # (e) Layer-resolved: early/mid/late correlation
    for size, data in all_results.items():
        T_sm = np.array(data['T_sm'])[1:]
        T_H = np.array(data['T_H'])[1:]
        n = len(T_sm)
        thirds = [('Early', 0, n//3), ('Mid', n//3, 2*n//3), ('Late', 2*n//3, n)]
        rs = []
        for label, s, e in thirds:
            if e - s >= 3:
                r, _ = stats.pearsonr(T_H[s:e], T_sm[s:e])
                rs.append((label, float(r)))
            else:
                rs.append((label, 0))
        x_t = range(len(rs))
        axes[1, 1].bar([xi + (0 if size == '0.5B' else 0.35) for xi in x_t],
                      [r[1] for r in rs], 0.35, color=colors[size],
                      alpha=0.8, label=size, edgecolor='black')
    axes[1, 1].set_xticks([xi + 0.175 for xi in range(3)])
    axes[1, 1].set_xticklabels(['Early', 'Mid', 'Late'])
    axes[1, 1].set_ylabel('r(T_sm, T_H)')
    axes[1, 1].set_title('(e) Layer-Resolved Correlation', fontweight='bold')
    axes[1, 1].axhline(0, color='gray', ls='--', lw=1)
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3, axis='y')

    # (f) Summary
    summary = "DUAL TEMPERATURE COUPLING\n\n"
    for size, data in all_results.items():
        summary += f"{size}:\n"
        summary += f"  Best: {data['best_fit']}\n"
        summary += f"  R^2 = {data['best_r2']:.4f}\n"
        summary += f"  {data['best_formula']}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 263: Dual Temperature Coupling Function T_sm = f(T_H)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase263_coupling_function')
    plt.close()

    save_results('phase263_coupling_function', {
        'experiment': 'Dual Temperature Coupling Function',
        'results': all_results,
    })


if __name__ == '__main__':
    main()
