# -*- coding: utf-8 -*-
"""
Phase 82: Thermal Conductivity & Diffusion
Measure how "heat" (entropy) diffuses between adjacent layers.
Fourier's law: J = -kappa * dT/dx
If thermal conductivity kappa is constant, heat flow is predictable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 82: Thermal Conductivity (Fourier's Law)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and",
        "Quantum mechanics describes particles at the atomic scale",
        "The human genome contains three billion base pairs encoding",
        "Neural networks learn through layers of interconnected nodes",
        "Black holes form from gravitational collapse of massive stars",
        "The periodic table organizes chemical elements by number",
        "Evolution by natural selection operates on heritable variation",
        "Climate models simulate atmospheric dynamics using equations",
        "Photosynthesis converts light energy into chemical energy",
        "Machine learning algorithms discover patterns in data",
        "General relativity describes gravity as curvature of spacetime",
        "The cosmic microwave background reveals the early universe",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_profile = []
        U_profile = []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U_profile.append(h.norm().item())
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_profile.append(T if not np.isnan(T) else 0)

        # Compute heat flux J and temperature gradient dT/dl
        J_list = []  # heat flux = dU/dl (energy flow)
        dT_list = []  # temperature gradient
        kappa_list = []  # thermal conductivity
        for i in range(1, len(T_profile)):
            dT = T_profile[i] - T_profile[i-1]
            dU = U_profile[i] - U_profile[i-1]
            J = dU  # heat flux proxy
            J_list.append(J)
            dT_list.append(dT)
            if abs(dT) > 1e-6:
                kappa = -J / dT  # Fourier: J = -kappa * dT/dx
                kappa_list.append(kappa)

        all_results.append({
            'T_profile': T_profile, 'U_profile': U_profile,
            'J': J_list, 'dT': dT_list, 'kappa': kappa_list,
        })

    n_trans = len(all_results[0]['J'])

    # J vs dT correlation (Fourier's law test)
    all_J = [j for r in all_results for j in r['J']]
    all_dT = [d for r in all_results for d in r['dT']]
    r_fourier, p_fourier = stats.pearsonr(all_J, all_dT)

    all_kappa = [k for r in all_results for k in r['kappa']]
    mean_kappa = np.mean(all_kappa) if all_kappa else 0
    std_kappa = np.std(all_kappa) if all_kappa else 0
    cv_kappa = std_kappa / (abs(mean_kappa) + 1e-10)

    print(f"\n=== Fourier's Law ===")
    print(f"  J-dT correlation: r={r_fourier:.3f}")
    print(f"  kappa = {mean_kappa:.1f} +/- {std_kappa:.1f} (CV={cv_kappa:.2f})")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) J vs dT scatter (Fourier test)
    axes[0, 0].scatter(all_dT, all_J, s=10, alpha=0.3, color='#e74c3c')
    slope_f, int_f, _, _, _ = stats.linregress(all_dT, all_J)
    dt_fit = np.linspace(min(all_dT), max(all_dT), 50)
    axes[0, 0].plot(dt_fit, slope_f * dt_fit + int_f, 'b--', linewidth=2,
                    label=f'slope={slope_f:.1f} (=-kappa)')
    axes[0, 0].set_xlabel('dT/dl (temperature gradient)')
    axes[0, 0].set_ylabel('J (heat flux)')
    axes[0, 0].set_title(f'(a) Fourier Law (r={r_fourier:.3f})')
    axes[0, 0].legend()

    # (b) kappa per layer
    mean_kappa_by_layer = []
    for l in range(n_trans):
        kappas_l = []
        for r in all_results:
            if l < len(r['dT']) and abs(r['dT'][l]) > 1e-6:
                kappas_l.append(-r['J'][l] / r['dT'][l])
        mean_kappa_by_layer.append(np.mean(kappas_l) if kappas_l else 0)
    axes[0, 1].plot(range(n_trans), mean_kappa_by_layer, 'o-', color='#f39c12',
                    linewidth=2, markersize=3)
    axes[0, 1].axhline(y=mean_kappa, color='blue', linestyle='--',
                       label=f'Mean kappa={mean_kappa:.1f}')
    axes[0, 1].set_xlabel('Layer Transition')
    axes[0, 1].set_ylabel('kappa')
    axes[0, 1].set_title(f'(b) Thermal Conductivity (CV={cv_kappa:.2f})')
    axes[0, 1].legend()

    # (c) Temperature profile
    mean_T = np.mean([r['T_profile'] for r in all_results], axis=0)
    axes[0, 2].plot(mean_T, 'o-', color='#3498db', linewidth=2, markersize=3)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Temperature')
    axes[0, 2].set_title('(c) Temperature Profile')

    # (d) Heat flux profile
    mean_J = np.mean([r['J'] for r in all_results], axis=0)
    colors_j = ['#2ecc71' if j > 0 else '#e74c3c' for j in mean_J]
    axes[1, 0].bar(range(n_trans), mean_J, color=colors_j, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black')
    axes[1, 0].set_xlabel('Layer Transition')
    axes[1, 0].set_ylabel('Heat Flux J')
    axes[1, 0].set_title('(d) Heat Flux Profile')

    # (e) kappa distribution
    axes[1, 1].hist(all_kappa, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=mean_kappa, color='red', linewidth=2)
    axes[1, 1].set_xlabel('kappa')
    axes[1, 1].set_title('(e) kappa Distribution')

    # (f) Summary
    fourier_holds = abs(r_fourier) > 0.3
    summary = (
        f"Fourier's Law: J = -kappa * dT/dl\n\n"
        f"J-dT correlation: r={r_fourier:.3f}\n"
        f"kappa = {mean_kappa:.1f} +/- {std_kappa:.1f}\n"
        f"CV = {cv_kappa:.2f}\n\n"
        f"{'FOURIER LAW HOLDS' if fourier_holds else 'FOURIER LAW WEAK'}\n"
        f"{'kappa IS constant' if cv_kappa < 0.5 else 'kappa is VARIABLE'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=11, va='center', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 82: Thermal Conductivity (kappa={mean_kappa:.1f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase82_thermal_conductivity')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Fourier r={r_fourier:.3f}, kappa={mean_kappa:.1f} (CV={cv_kappa:.2f}). "
          f"{'FOURIER HOLDS' if fourier_holds else 'WEAK correlation'}.")
    print(f"{'='*70}")

    save_results('phase82_thermal_conductivity', {
        'experiment': 'Thermal Conductivity',
        'summary': {
            'fourier_r': float(r_fourier),
            'mean_kappa': float(mean_kappa),
            'cv_kappa': float(cv_kappa),
            'fourier_holds': bool(fourier_holds),
        }
    })


if __name__ == '__main__':
    main()
