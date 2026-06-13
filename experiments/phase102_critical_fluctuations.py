# -*- coding: utf-8 -*-
"""
Phase 102: Critical Fluctuations at Eta Transition
Near a phase transition, fluctuations peak (critical slowing down).
Measure variance of eta across prompts at each effective depth L.
If variance peaks at L0=22, it's a true phase transition.
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
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "The cosmic microwave background reveals the early universe",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
    "The Fibonacci sequence appears everywhere in nature",
    "Antibiotics kill bacteria by targeting cell walls",
    "The double slit experiment demonstrates wave particle duality",
    "Entropy measures the disorder of a system",
]


def measure_eta_per_prompt(model, tok, device, max_layer):
    """Return individual eta values for each prompt."""
    etas = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
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
                etas.append(1.0 - T_cold / T_hot)
            else:
                etas.append(0.0)
        else:
            etas.append(0.0)
    return etas


def main():
    print("=" * 70)
    print("Phase 102: Critical Fluctuations at Eta Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    layer_counts = list(range(4, 29))
    results = []

    for L in layer_counts:
        etas = measure_eta_per_prompt(model, tok, device, L)
        mean_e = np.mean(etas)
        std_e = np.std(etas)
        var_e = np.var(etas)
        # Susceptibility = variance / mean
        chi = var_e / (mean_e + 1e-10)
        # Binder cumulant U4 = 1 - <eta^4> / (3 * <eta^2>^2)
        m2 = np.mean(np.array(etas)**2)
        m4 = np.mean(np.array(etas)**4)
        binder = 1 - m4 / (3 * m2**2 + 1e-10) if m2 > 1e-10 else 0

        results.append({
            'L': L,
            'eta_mean': float(mean_e),
            'eta_std': float(std_e),
            'eta_var': float(var_e),
            'susceptibility': float(chi),
            'binder_cumulant': float(binder),
            'etas': [float(e) for e in etas],
        })
        print(f"  L={L:2d}: eta={mean_e:.4f}+/-{std_e:.4f}, chi={chi:.6f}, U4={binder:.4f}")

    Ls = np.array([r['L'] for r in results])
    means = np.array([r['eta_mean'] for r in results])
    stds = np.array([r['eta_std'] for r in results])
    vars_ = np.array([r['eta_var'] for r in results])
    chis = np.array([r['susceptibility'] for r in results])
    binders = np.array([r['binder_cumulant'] for r in results])

    # Find peak of susceptibility
    chi_peak_L = Ls[np.argmax(chis)]
    var_peak_L = Ls[np.argmax(vars_)]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) eta with error bars
    axes[0,0].errorbar(Ls, means, yerr=stds, fmt='o-', color='#c0392b',
                       markersize=4, capsize=2, linewidth=1.5)
    L_sm = np.linspace(4, 28, 200)
    axes[0,0].plot(L_sm, 1-1/np.sqrt(L_sm), '--', color='#2980b9', alpha=0.5)
    axes[0,0].axvline(x=22, color='#f39c12', linestyle='--', alpha=0.5, label='$L_0=22$')
    axes[0,0].set_xlabel('$L$')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) $\\eta \\pm \\sigma$')
    axes[0,0].legend(fontsize=8)

    # (b) Variance (order parameter fluctuations)
    axes[0,1].plot(Ls, vars_, 'o-', color='#8e44ad', markersize=5, linewidth=2)
    axes[0,1].axvline(x=var_peak_L, color='#f39c12', linestyle='--',
                      label=f'Peak: L={var_peak_L}')
    axes[0,1].axvline(x=22, color='gray', linestyle=':', alpha=0.5, label='$L_0=22$')
    axes[0,1].set_xlabel('$L$')
    axes[0,1].set_ylabel('$\\mathrm{Var}(\\eta)$')
    axes[0,1].set_title(f'(b) Variance (peak at L={var_peak_L})')
    axes[0,1].legend(fontsize=8)

    # (c) Susceptibility
    axes[0,2].plot(Ls, chis, 's-', color='#e74c3c', markersize=5, linewidth=2)
    axes[0,2].axvline(x=chi_peak_L, color='#f39c12', linestyle='--',
                      label=f'Peak: L={chi_peak_L}')
    axes[0,2].set_xlabel('$L$')
    axes[0,2].set_ylabel('$\\chi = \\mathrm{Var}/\\langle\\eta\\rangle$')
    axes[0,2].set_title(f'(c) Susceptibility (peak at L={chi_peak_L})')
    axes[0,2].legend(fontsize=8)

    # (d) Binder cumulant
    axes[1,0].plot(Ls, binders, 'D-', color='#27ae60', markersize=5, linewidth=2)
    axes[1,0].axhline(y=2/3, color='gray', linestyle=':', alpha=0.5, label='Gaussian (2/3)')
    axes[1,0].axvline(x=22, color='#f39c12', linestyle='--', alpha=0.5)
    axes[1,0].set_xlabel('$L$')
    axes[1,0].set_ylabel('$U_4$ (Binder)')
    axes[1,0].set_title('(d) Binder Cumulant')
    axes[1,0].legend(fontsize=8)

    # (e) Individual prompt trajectories
    for i in range(min(5, len(PROMPTS))):
        traj = [r['etas'][i] for r in results]
        axes[1,1].plot(Ls, traj, '-', alpha=0.5, linewidth=1,
                       label=PROMPTS[i][:20]+'...' if i < 3 else None)
    axes[1,1].plot(Ls, means, 'k-', linewidth=2.5, label='Mean')
    axes[1,1].axvline(x=22, color='#f39c12', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('$L$')
    axes[1,1].set_ylabel('$\\eta$')
    axes[1,1].set_title('(e) Individual Trajectories')
    axes[1,1].legend(fontsize=6)

    # (f) Summary
    summary = (
        f"Critical Fluctuation Analysis\n\n"
        f"Variance peak: L={var_peak_L}\n"
        f"Susceptibility peak: L={chi_peak_L}\n"
        f"Eta transition L0: 22\n\n"
        f"Match: {'YES' if abs(chi_peak_L - 22) <= 3 else 'NO'}\n\n"
        f"Peak variance: {vars_.max():.6f}\n"
        f"Peak chi: {chis.max():.6f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    is_critical = abs(chi_peak_L - 22) <= 3
    fig.suptitle(f'Phase 102: Critical Fluctuations '
                 f'($\\chi$ peak at L={chi_peak_L}, '
                 f'{"CONFIRMED" if is_critical else "SHIFTED"})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase102_critical_fluctuations')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Variance peak: L={var_peak_L}")
    print(f"Susceptibility peak: L={chi_peak_L}")
    print(f"Transition at L0=22: {'CONFIRMED' if is_critical else 'SHIFTED'}")
    print(f"{'='*70}")

    save_results('phase102_critical_fluctuations', {
        'experiment': 'Critical Fluctuations at Eta Transition',
        'results': [{k: v for k, v in r.items() if k != 'etas'} for r in results],
        'summary': {
            'var_peak_L': int(var_peak_L),
            'chi_peak_L': int(chi_peak_L),
            'is_critical': is_critical,
            'max_chi': float(chis.max()),
            'max_var': float(vars_.max()),
        }
    })


if __name__ == '__main__':
    main()
