# -*- coding: utf-8 -*-
"""
Phase 97: Eta Convergence Phase Transition
Phase 95 showed eta suddenly converges to 1-1/sqrt(L) around L=20-24.
This looks like a phase transition. Measure the order parameter
(eta deviation from theory) to find the critical layer count L*.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
]


def measure_eta_at_depth(model, tok, device, max_layer):
    """Measure eta using layers 0..max_layer."""
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

    return float(np.mean(etas)) if etas else 0.0


def main():
    print("=" * 70)
    print("Phase 97: Eta Convergence Phase Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    # Dense sampling around the transition region
    layer_counts = list(range(4, 29))
    results = []

    for L in layer_counts:
        eta = measure_eta_at_depth(model, tok, device, L)
        theory = 1.0 - 1.0 / np.sqrt(L)
        deviation = abs(eta - theory)
        results.append({
            'L': L,
            'eta': float(eta),
            'theory': float(theory),
            'deviation': float(deviation),
        })
        print(f"  L={L:2d}: eta={eta:.4f}, theory={theory:.4f}, dev={deviation:.4f}")

    Ls = np.array([r['L'] for r in results])
    etas = np.array([r['eta'] for r in results])
    devs = np.array([r['deviation'] for r in results])

    # Find L*: layer where deviation drops below 0.05
    L_star = None
    for r in results:
        if r['deviation'] < 0.05:
            L_star = r['L']
            break

    # Fit sigmoid to the convergence
    def sigmoid(x, L0, k, ymin, ymax):
        return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))

    try:
        popt, _ = curve_fit(sigmoid, Ls, etas, p0=[20, 0.5, 0.3, 0.85], maxfev=10000)
        L0_fit, k_fit = popt[0], popt[1]
        sig_pred = sigmoid(Ls, *popt)
        ss_res = np.sum((etas - sig_pred)**2)
        ss_tot = np.sum((etas - np.mean(etas))**2)
        sig_r2 = 1 - ss_res / (ss_tot + 1e-10)
    except Exception:
        L0_fit, k_fit, sig_r2 = 20, 0.5, 0
        sig_pred = etas

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) eta vs L with transition
    axes[0].plot(Ls, etas, 'o-', color='#c0392b', markersize=5, linewidth=1.5, label='Measured')
    L_smooth = np.linspace(4, 28, 200)
    axes[0].plot(L_smooth, 1.0 - 1.0/np.sqrt(L_smooth), '--', color='#2980b9',
                 linewidth=2, label='$1-1/\\sqrt{L}$')
    if sig_r2 > 0.5:
        axes[0].plot(L_smooth, sigmoid(L_smooth, *popt), ':', color='#27ae60',
                     linewidth=2, label=f'Sigmoid ($L_0={L0_fit:.1f}$)')
    if L_star:
        axes[0].axvline(x=L_star, color='#f39c12', linestyle='--', alpha=0.7,
                        label=f'$L^* = {L_star}$')
    axes[0].set_xlabel('Effective Layer Count $L$')
    axes[0].set_ylabel('$\\eta$')
    axes[0].set_title('(a) Eta Phase Transition')
    axes[0].legend(fontsize=7)

    # (b) Deviation (order parameter)
    colors_d = ['#27ae60' if d < 0.05 else '#c0392b' for d in devs]
    axes[1].bar(range(len(results)), devs, color=colors_d, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels([str(r['L']) for r in results], fontsize=7)
    axes[1].axhline(y=0.05, color='#f39c12', linestyle='--', label='Threshold (0.05)')
    axes[1].set_xlabel('$L$')
    axes[1].set_ylabel('$|\\eta - \\eta_{theory}|$')
    axes[1].set_title('(b) Order Parameter')
    axes[1].legend(fontsize=8)

    # (c) d(eta)/dL - the "susceptibility"
    deta_dL = np.gradient(etas, Ls)
    axes[2].plot(Ls, deta_dL, 'o-', color='#8e44ad', markersize=5, linewidth=1.5)
    peak_L = Ls[np.argmax(deta_dL)]
    axes[2].axvline(x=peak_L, color='#f39c12', linestyle='--',
                    label=f'Peak at $L={peak_L}$')
    axes[2].set_xlabel('$L$')
    axes[2].set_ylabel('$d\\eta/dL$')
    axes[2].set_title(f'(c) Susceptibility (peak at L={peak_L})')
    axes[2].legend(fontsize=8)

    fig.suptitle(f'Phase 97: Eta Convergence ($L^*={L_star}$, sigmoid $L_0={L0_fit:.1f}$)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase97_eta_transition')
    plt.close()

    print(f"\n{'='*70}")
    print(f"L* (convergence): {L_star}")
    print(f"Sigmoid midpoint L0: {L0_fit:.1f}")
    print(f"Susceptibility peak: L={peak_L}")
    print(f"Sigmoid R2: {sig_r2:.4f}")
    print(f"{'='*70}")

    save_results('phase97_eta_transition', {
        'experiment': 'Eta Convergence Phase Transition',
        'results': results,
        'summary': {
            'L_star': L_star,
            'L0_sigmoid': float(L0_fit),
            'k_sigmoid': float(k_fit),
            'sigmoid_r2': float(sig_r2),
            'susceptibility_peak': int(peak_L),
        }
    })


if __name__ == '__main__':
    main()
