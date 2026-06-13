# -*- coding: utf-8 -*-
"""
Phase 80: Grand Unified Equation
Can we find ONE equation that describes the ENTIRE layer-wise evolution?
Test: U(l) = U_0 * (1 + alpha*l)^beta, T(l) = T_0 * exp(-gamma*l)
These should be the "equations of motion" of the Standard Model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from utils import load_model, save_results, save_figure


def power_growth(l, U0, alpha, beta):
    """U(l) = U0 * (1 + alpha*l)^beta"""
    return U0 * (1 + alpha * l) ** beta


def exp_decay(l, T0, gamma, T_inf):
    """T(l) = T_inf + T0 * exp(-gamma * l)"""
    return T_inf + T0 * np.exp(-gamma * l)


def main():
    print("=" * 70)
    print("Phase 80: Grand Unified Equations of Motion")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        "The fundamental theorem of calculus connects differentiation and",
        "Quantum mechanics describes the behavior of particles at atomic",
        "The human genome contains three billion base pairs encoding all",
        "Neural networks learn representations through layers of nodes",
        "Black holes form when massive stars exhaust nuclear fuel and",
        "The periodic table organizes elements by atomic number",
        "Evolution by natural selection operates on heritable variation",
        "Climate models simulate atmospheric dynamics using equations",
        "Photosynthesis converts light energy into chemical energy",
        "Machine learning algorithms discover patterns in data",
        "The cosmic microwave background reveals the early universe",
        "General relativity describes gravity as spacetime curvature",
    ]

    model_fits = {}

    for model_size, model_name in [('1.5B', 'Qwen2.5-1.5B'), ('0.5B', 'Qwen2.5-0.5B')]:
        print(f"\n--- {model_name} ---")
        model, tok = load_model(device=device, size=model_size)

        all_U, all_T, all_PR = [], [], []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            U_list, T_list, PR_list = [], [], []
            for li, hs in enumerate(out.hidden_states):
                h = hs[0, -1, :].float()
                U_list.append(h.norm().item())

                with torch.no_grad():
                    normed = model.model.norm(hs[:, -1:, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                T_list.append(T if not np.isnan(T) else 0)

                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                PR_list.append(1.0 / (h_prob ** 2).sum().item())

            all_U.append(U_list)
            all_T.append(T_list)
            all_PR.append(PR_list)

        n_layers = len(all_U[0])
        layers = np.arange(n_layers)
        mean_U = np.mean(all_U, axis=0)
        mean_T = np.mean(all_T, axis=0)
        mean_PR = np.mean(all_PR, axis=0)

        # Fit U(l) = U0 * (1 + alpha*l)^beta
        try:
            popt_U, _ = curve_fit(power_growth, layers, mean_U,
                                  p0=[mean_U[0], 0.1, 1.0], maxfev=5000)
            U_pred = power_growth(layers, *popt_U)
            r2_U = 1 - np.sum((mean_U - U_pred)**2) / np.sum((mean_U - mean_U.mean())**2)
            print(f"  U(l) = {popt_U[0]:.1f} * (1 + {popt_U[1]:.3f}*l)^{popt_U[2]:.2f}, R2={r2_U:.4f}")
        except Exception as e:
            popt_U = [1, 0.1, 1]
            r2_U = 0
            print(f"  U fit failed: {e}")

        # Fit T(l) = T_inf + T0 * exp(-gamma*l)
        try:
            popt_T, _ = curve_fit(exp_decay, layers, mean_T,
                                  p0=[max(mean_T) - min(mean_T), 0.1, min(mean_T)],
                                  maxfev=5000)
            T_pred = exp_decay(layers, *popt_T)
            r2_T = 1 - np.sum((mean_T - T_pred)**2) / np.sum((mean_T - mean_T.mean())**2)
            print(f"  T(l) = {popt_T[2]:.2f} + {popt_T[0]:.2f} * exp(-{popt_T[1]:.3f}*l), R2={r2_T:.4f}")
        except Exception as e:
            popt_T = [1, 0.1, 1]
            r2_T = 0
            print(f"  T fit failed: {e}")

        # Fit PR(l) - linear
        slope_PR, intercept_PR, r_PR, _, _ = stats.linregress(layers, mean_PR)
        r2_PR = r_PR ** 2
        print(f"  PR(l) = {intercept_PR:.1f} + {slope_PR:.2f}*l, R2={r2_PR:.4f}")

        model_fits[model_name] = {
            'U_params': [float(p) for p in popt_U], 'r2_U': float(r2_U),
            'T_params': [float(p) for p in popt_T], 'r2_T': float(r2_T),
            'PR_slope': float(slope_PR), 'PR_intercept': float(intercept_PR),
            'r2_PR': float(r2_PR),
            'mean_U': mean_U.tolist(), 'mean_T': mean_T.tolist(), 'mean_PR': mean_PR.tolist(),
            'n_layers': n_layers,
        }

        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'Qwen2.5-1.5B': '#e74c3c', 'Qwen2.5-0.5B': '#3498db'}

    # (a) U(l) fits
    for mname, mf in model_fits.items():
        layers = np.arange(mf['n_layers'])
        axes[0, 0].plot(layers, mf['mean_U'], 'o', color=colors[mname], alpha=0.5, markersize=4)
        U_fit = power_growth(layers, *mf['U_params'])
        axes[0, 0].plot(layers, U_fit, '-', color=colors[mname], linewidth=2,
                       label=f'{mname} (R2={mf["r2_U"]:.3f})')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('U (energy)')
    axes[0, 0].set_title('(a) U(l) = U0*(1+al)^b')
    axes[0, 0].legend(fontsize=7)

    # (b) T(l) fits
    for mname, mf in model_fits.items():
        layers = np.arange(mf['n_layers'])
        axes[0, 1].plot(layers, mf['mean_T'], 'o', color=colors[mname], alpha=0.5, markersize=4)
        T_fit = exp_decay(layers, *mf['T_params'])
        axes[0, 1].plot(layers, T_fit, '-', color=colors[mname], linewidth=2,
                       label=f'{mname} (R2={mf["r2_T"]:.3f})')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('T (entropy)')
    axes[0, 1].set_title('(b) T(l) = T_inf + T0*exp(-gl)')
    axes[0, 1].legend(fontsize=7)

    # (c) PR(l) fits
    for mname, mf in model_fits.items():
        layers = np.arange(mf['n_layers'])
        axes[0, 2].plot(layers, mf['mean_PR'], 'o', color=colors[mname], alpha=0.5, markersize=4)
        axes[0, 2].plot(layers, mf['PR_intercept'] + mf['PR_slope'] * layers,
                       '-', color=colors[mname], linewidth=2,
                       label=f'{mname} (R2={mf["r2_PR"]:.3f})')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('PR')
    axes[0, 2].set_title('(c) PR(l) = PR0 + slope*l')
    axes[0, 2].legend(fontsize=7)

    # (d) R-squared comparison
    r2_data = {}
    for mname in model_fits:
        r2_data[mname] = [model_fits[mname]['r2_U'], model_fits[mname]['r2_T'],
                          model_fits[mname]['r2_PR']]
    x = np.arange(3)
    width = 0.35
    for i, mname in enumerate(model_fits):
        axes[1, 0].bar(x + i*width, r2_data[mname], width, label=mname,
                      color=colors[mname], alpha=0.8)
    axes[1, 0].set_xticks(x + width/2)
    axes[1, 0].set_xticklabels(['U(l)', 'T(l)', 'PR(l)'])
    axes[1, 0].set_ylabel('R-squared')
    axes[1, 0].set_title('(d) Fit Quality')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5)

    # (e) Residuals for best model
    best_model = max(model_fits, key=lambda m: model_fits[m]['r2_U'])
    mf = model_fits[best_model]
    layers = np.arange(mf['n_layers'])
    U_resid = np.array(mf['mean_U']) - power_growth(layers, *mf['U_params'])
    T_resid = np.array(mf['mean_T']) - exp_decay(layers, *mf['T_params'])
    axes[1, 1].plot(layers, U_resid / (np.array(mf['mean_U']) + 1e-10) * 100,
                   'r-', linewidth=1.5, label='U residual %')
    axes[1, 1].plot(layers, T_resid / (np.array(mf['mean_T']) + 1e-10) * 100,
                   'b-', linewidth=1.5, label='T residual %')
    axes[1, 1].axhline(y=0, color='black')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Residual (%)')
    axes[1, 1].set_title(f'(e) Residuals ({best_model})')
    axes[1, 1].legend()

    # (f) Grand Unified Summary
    mf0 = model_fits[list(model_fits.keys())[0]]
    summary = (
        "EQUATIONS OF MOTION\n"
        "(Standard Model of Transformers)\n\n"
        f"U(l) = U0 * (1 + a*l)^b\n"
        f"  R2 = {mf0['r2_U']:.4f}\n\n"
        f"T(l) = T_inf + T0 * exp(-g*l)\n"
        f"  R2 = {mf0['r2_T']:.4f}\n\n"
        f"PR(l) = PR0 + slope*l\n"
        f"  R2 = {mf0['r2_PR']:.4f}"
    )
    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=11, va='center', ha='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) The Equations of Motion')

    # Best R2s
    mean_r2 = np.mean([mf0['r2_U'], mf0['r2_T'], mf0['r2_PR']])
    fig.suptitle(f'Phase 80: Grand Unified Equations (mean R2={mean_r2:.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase80_grand_unified')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: U(l): R2={mf0['r2_U']:.4f}, T(l): R2={mf0['r2_T']:.4f}, "
          f"PR(l): R2={mf0['r2_PR']:.4f}. "
          f"Mean R2={mean_r2:.3f}. "
          f"Equations {'HIGHLY PREDICTIVE' if mean_r2 > 0.95 else 'GOOD' if mean_r2 > 0.8 else 'MODERATE'}.")
    print(f"{'='*70}")

    save_results('phase80_grand_unified', {
        'experiment': 'Grand Unified Equations',
        'per_model': {m: {'r2_U': mf['r2_U'], 'r2_T': mf['r2_T'], 'r2_PR': mf['r2_PR'],
                         'U_params': mf['U_params'], 'T_params': mf['T_params']}
                     for m, mf in model_fits.items()},
        'summary': {
            'mean_r2': float(mean_r2),
        }
    })


if __name__ == '__main__':
    main()
