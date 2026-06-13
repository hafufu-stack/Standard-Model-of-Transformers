# -*- coding: utf-8 -*-
"""
Phase 125: Thermodynamic Equation of State
Find the equation relating kT, eta, and entropy S.
Does the Transformer obey PV=NkT-like universal relations?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
]


def main():
    print("=" * 70)
    print("Phase 125: Thermodynamic Equation of State")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect kT, eta, S, U at each layer
    all_kT = [[] for _ in range(n_layers)]
    all_S = [[] for _ in range(n_layers)]
    all_eta = [[] for _ in range(n_layers)]
    all_U = [[] for _ in range(n_layers)]  # internal energy = mean |h|^2

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            # Output entropy (S)
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(S): S = 0

            T_vals.append(S)
            all_S[li].append(S)

            # kT from Boltzmann fit (use top-k as proxy)
            top_k = 100
            top_probs = torch.topk(probs, top_k).values
            log_probs = torch.log(top_probs + 1e-10)
            ranks = torch.arange(1, top_k + 1, dtype=torch.float32)
            # E ~ rank, P ~ exp(-E/kT) => log P ~ -rank/kT
            if log_probs.std() > 0.01:
                slope = np.polyfit(ranks.numpy(), log_probs.cpu().numpy(), 1)[0]
                kT = -1.0 / (slope + 1e-10)
            else:
                kT = 0.1
            kT = max(0.01, min(kT, 100))
            all_kT[li].append(float(kT))

            # Internal energy U = mean |h|^2
            U = (h ** 2).mean().item()
            all_U[li].append(float(U))

        # Eta at each layer
        for li in range(n_layers):
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                if T_hot > 0.01:
                    all_eta[li].append(1.0 - T_cold / T_hot)
                else:
                    all_eta[li].append(0.0)
            else:
                all_eta[li].append(0.0)

    # Average
    avg_kT = [np.mean(v) if v else 0 for v in all_kT]
    avg_S = [np.mean(v) if v else 0 for v in all_S]
    avg_eta = [np.mean(v) if v else 0 for v in all_eta]
    avg_U = [np.mean(v) if v else 0 for v in all_U]

    # Try to find equation of state: f(kT, eta, S) = 0
    # Test: S = a * kT * (1 - eta) + b
    X = np.array([[avg_kT[li], avg_eta[li], avg_kT[li] * (1 - avg_eta[li])]
                  for li in range(4, n_layers)])
    y = np.array([avg_S[li] for li in range(4, n_layers)])

    reg = LinearRegression()
    reg.fit(X, y)
    r2_fit = reg.score(X, y)
    y_pred = reg.predict(X)

    # Also test: S = a * kT^alpha * (1 - eta)^beta
    # Log-linear fit
    valid = [(avg_kT[li], avg_eta[li], avg_S[li]) for li in range(4, n_layers)
             if avg_kT[li] > 0.01 and avg_S[li] > 0.1 and avg_eta[li] < 0.99]
    if len(valid) > 5:
        log_kT = np.log([v[0] for v in valid])
        log_1meta = np.log([1 - v[1] + 1e-10 for v in valid])
        log_S = np.log([v[2] for v in valid])

        X_log = np.column_stack([log_kT, log_1meta])
        reg_log = LinearRegression()
        reg_log.fit(X_log, log_S)
        r2_log = reg_log.score(X_log, log_S)
        alpha = reg_log.coef_[0]
        beta = reg_log.coef_[1]
    else:
        alpha, beta, r2_log = 0, 0, 0

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers = np.arange(n_layers)

    # (a) State variables profile
    ax_a = axes[0, 0]
    ax_a.plot(layers, avg_kT, 'o-', color='#c0392b', markersize=3, label='$kT$')
    ax_a2 = ax_a.twinx()
    ax_a2.plot(layers, avg_S, 's-', color='#2980b9', markersize=3, label='$S$')
    ax_a.set_xlabel('Layer')
    ax_a.set_ylabel('$kT$', color='#c0392b')
    ax_a2.set_ylabel('$S$ (entropy)', color='#2980b9')
    ax_a.set_title('(a) State Variables')
    ax_a.axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')

    # (b) S vs kT phase diagram
    colors_b = plt.cm.coolwarm(np.linspace(0, 1, n_layers))
    for li in range(4, n_layers):
        axes[0, 1].scatter(avg_kT[li], avg_S[li], c=[colors_b[li]], s=50,
                          edgecolors='black', zorder=5)
        axes[0, 1].annotate(f'{li}', (avg_kT[li], avg_S[li]),
                           fontsize=6, alpha=0.7)
    axes[0, 1].set_xlabel('$kT$')
    axes[0, 1].set_ylabel('$S$')
    axes[0, 1].set_title('(b) Phase Diagram ($S$ vs $kT$)')

    # (c) kT vs eta
    for li in range(4, n_layers):
        axes[0, 2].scatter(avg_eta[li], avg_kT[li], c=[colors_b[li]], s=50,
                          edgecolors='black', zorder=5)
        axes[0, 2].annotate(f'{li}', (avg_eta[li], avg_kT[li]),
                           fontsize=6, alpha=0.7)
    axes[0, 2].set_xlabel('$\\eta$')
    axes[0, 2].set_ylabel('$kT$')
    axes[0, 2].set_title('(c) Phase Diagram ($kT$ vs $\\eta$)')

    # (d) Equation of state fit
    axes[1, 0].scatter(y, y_pred, c=range(len(y)), cmap='coolwarm', s=50, edgecolors='black')
    axes[1, 0].plot([min(y), max(y)], [min(y), max(y)], 'k--')
    axes[1, 0].set_xlabel('$S$ (actual)')
    axes[1, 0].set_ylabel('$S$ (predicted)')
    axes[1, 0].set_title(f'(d) Linear EoS ($R^2={r2_fit:.3f}$)')

    # (e) 3D-like view: eta, kT, S
    sc = axes[1, 1].scatter(avg_eta[4:], avg_kT[4:], c=avg_S[4:], s=80,
                            cmap='viridis', edgecolors='black')
    plt.colorbar(sc, ax=axes[1, 1], label='$S$')
    axes[1, 1].set_xlabel('$\\eta$')
    axes[1, 1].set_ylabel('$kT$')
    axes[1, 1].set_title('(e) State Space ($\\eta$, $kT$, $S$)')

    # (f) Summary
    summary = (
        f"Equation of State\n\n"
        f"Linear: S = f(kT, eta, kT*(1-eta))\n"
        f"  R2 = {r2_fit:.3f}\n"
        f"  Coefficients: {reg.coef_}\n\n"
        f"Power law: S ~ kT^a * (1-eta)^b\n"
        f"  R2 = {r2_log:.3f}\n"
        f"  alpha = {alpha:.3f}\n"
        f"  beta = {beta:.3f}\n\n"
        f"Best fit: S ~ kT^{alpha:.2f} * (1-eta)^{beta:.2f}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle(f'Phase 125: Equation of State (R2={max(r2_fit, r2_log):.3f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase125_eos')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Linear R2: {r2_fit:.3f}")
    print(f"Power law: S ~ kT^{alpha:.2f} * (1-eta)^{beta:.2f}, R2={r2_log:.3f}")
    print(f"{'='*70}")

    save_results('phase125_eos', {
        'experiment': 'Equation of State',
        'avg_kT': [float(v) for v in avg_kT],
        'avg_S': [float(v) for v in avg_S],
        'avg_eta': [float(v) for v in avg_eta],
        'avg_U': [float(v) for v in avg_U],
        'summary': {
            'linear_r2': float(r2_fit),
            'powerlaw_r2': float(r2_log),
            'alpha': float(alpha),
            'beta': float(beta),
        }
    })


if __name__ == '__main__':
    main()
