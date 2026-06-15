# -*- coding: utf-8 -*-
"""
Phase 184: Entropy Production Rate
Measure irreversible entropy production rate sigma_irr per layer.
Verify the second law: sigma_irr >= 0 at each layer.
Map the pattern of thermodynamic irreversibility across the architecture.
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
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "DNA stores hereditary genetic information",
    "Entropy measures disorder in physical systems",
    "The Turing machine is a fundamental model of computation",
    "Superconductors carry current with zero resistance",
    "The human genome contains approximately three billion base pairs",
    "Artificial neural networks are inspired by biological neurons",
]


def main():
    print("=" * 70)
    print("Phase 184: Entropy Production Rate")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_sigma = []  # Entropy production rate per layer
    all_Q = []  # Heat exchange
    all_W = []  # Work done
    all_dS = []
    all_T = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_vals, T_vals, S_vals = [], [], []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()

            U_vals.append(U if not np.isnan(U) else 0)
            T_vals.append(T if not np.isnan(T) else 0)
            S_vals.append(S if not np.isnan(S) else 0)

        # Compute entropy production rate per layer transition
        sigma_vals = []
        Q_vals = []
        W_vals = []
        dS_vals = []
        for i in range(n_layers - 1):
            dU = U_vals[i+1] - U_vals[i]
            dS = S_vals[i+1] - S_vals[i]
            T_avg = (T_vals[i] + T_vals[i+1]) / 2 + 1e-10

            # First law: dU = Q - W  (Q = TdS for reversible)
            Q = T_avg * dS  # Heat exchange
            W = Q - dU      # Work done by layer

            # Irreversible entropy production: sigma = dS - Q/T
            # sigma_irr = dS_total - dS_reversible
            sigma = dS - Q / (T_avg + 1e-10) if abs(T_avg) > 1e-6 else 0

            sigma_vals.append(sigma if not np.isnan(sigma) else 0)
            Q_vals.append(Q if not np.isnan(Q) else 0)
            W_vals.append(W if not np.isnan(W) else 0)
            dS_vals.append(dS if not np.isnan(dS) else 0)

        all_sigma.append(sigma_vals)
        all_Q.append(Q_vals)
        all_W.append(W_vals)
        all_dS.append(dS_vals)
        all_T.append(T_vals)

    sigma_mean = np.mean(all_sigma, axis=0)
    sigma_std = np.std(all_sigma, axis=0)
    Q_mean = np.mean(all_Q, axis=0)
    W_mean = np.mean(all_W, axis=0)
    dS_mean = np.mean(all_dS, axis=0)
    T_mean = np.mean(all_T, axis=0)

    layers_t = np.arange(n_layers - 1) + 0.5

    # 2nd law check
    n_violations = sum(1 for s in sigma_mean if s < -0.01)
    total_sigma = sum(sigma_mean)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Entropy production rate
    axes[0, 0].fill_between(layers_t, sigma_mean - sigma_std, sigma_mean + sigma_std,
                            alpha=0.3, color='#e74c3c')
    axes[0, 0].plot(layers_t, sigma_mean, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', label='$\\sigma_{irr} = 0$ (reversible)')
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('$\\sigma_{irr}$')
    axes[0, 0].set_title('(a) Entropy Production Rate')
    axes[0, 0].legend(fontsize=8)

    # (b) Cumulative entropy production
    sigma_cumul = np.cumsum(sigma_mean)
    axes[0, 1].plot(layers_t, sigma_cumul, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$\\Sigma_{irr}$ (cumulative)')
    axes[0, 1].set_title('(b) Cumulative Irreversibility')

    # (c) Heat vs Work decomposition
    axes[0, 2].plot(layers_t, Q_mean, 'o-', color='#e74c3c', markersize=3, linewidth=2, label='$Q$ (Heat)')
    axes[0, 2].plot(layers_t, W_mean, 's-', color='#3498db', markersize=3, linewidth=2, label='$W$ (Work)')
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].axhline(y=0, color='gray', linewidth=0.5)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Energy')
    axes[0, 2].set_title('(c) First Law: Q and W')
    axes[0, 2].legend(fontsize=8)

    # (d) dS vs Q/T (reversible vs irreversible)
    Q_over_T = [Q_mean[i] / (T_mean[i] + 1e-10) for i in range(n_layers - 1)]
    axes[1, 0].scatter(Q_over_T, dS_mean, c=layers_t, cmap='coolwarm', s=60, edgecolors='black')
    lim = max(max(abs(min(Q_over_T)), abs(max(Q_over_T))),
              max(abs(min(dS_mean)), abs(max(dS_mean)))) * 1.1
    axes[1, 0].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='Reversible ($dS = Q/T$)')
    axes[1, 0].set_xlabel('$Q/T$ (Reversible entropy change)')
    axes[1, 0].set_ylabel('$dS$ (Actual entropy change)')
    axes[1, 0].set_title('(d) Irreversibility Map')
    axes[1, 0].legend(fontsize=8)

    # (e) Sigma per prompt (heatmap-style)
    sigma_arr = np.array(all_sigma[:10])  # First 10 prompts
    im = axes[1, 1].imshow(sigma_arr, aspect='auto', cmap='RdBu_r',
                            extent=[0, n_layers-1, len(sigma_arr), 0])
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    plt.colorbar(im, ax=axes[1, 1], label='$\\sigma_{irr}$')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Prompt')
    axes[1, 1].set_title('(e) Prompt-Resolved Entropy Production')

    # (f) Summary
    pre_sigma = np.mean(sigma_mean[:L0])
    post_sigma = np.mean(sigma_mean[L0:])
    summary = (
        f"Entropy Production Rate\n\n"
        f"Total sigma_irr: {total_sigma:.4f}\n"
        f"2nd law violations: {n_violations}/{len(sigma_mean)}\n\n"
        f"Pre-L0 mean sigma: {pre_sigma:.5f}\n"
        f"Post-L0 mean sigma: {post_sigma:.5f}\n\n"
        f"{'MORE' if abs(post_sigma) > abs(pre_sigma) else 'LESS'}\n"
        f"irreversible post-transition\n\n"
        f"2nd Law: {'SATISFIED' if n_violations < 3 else 'VIOLATED'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 184: Entropy Production Rate', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase184_entropy_production_rate')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Total sigma_irr: {total_sigma:.4f}")
    print(f"2nd law violations: {n_violations}/{len(sigma_mean)}")
    print(f"Pre-L0: {pre_sigma:.5f}, Post-L0: {post_sigma:.5f}")
    print(f"{'=' * 70}")

    save_results('phase184_entropy_production_rate', {
        'experiment': 'Entropy Production Rate',
        'sigma_mean': [float(x) for x in sigma_mean],
        'sigma_cumulative': [float(x) for x in sigma_cumul],
        'Q_mean': [float(x) for x in Q_mean],
        'W_mean': [float(x) for x in W_mean],
        'total_sigma': float(total_sigma),
        'n_violations': n_violations,
        'pre_L0_sigma': float(pre_sigma),
        'post_L0_sigma': float(post_sigma),
    })


if __name__ == '__main__':
    main()
