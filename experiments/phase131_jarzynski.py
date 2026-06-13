# -*- coding: utf-8 -*-
"""
Phase 131: Jarzynski Equality Test
For non-equilibrium processes, the Jarzynski equality relates
free energy differences to work:
  exp(-beta * delta_F) = <exp(-beta * W)>
where W is the work done along non-equilibrium trajectories.
Test if this holds for layer-to-layer transitions.
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
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
    "Climate change affects global ecosystems",
    "The discovery of antibiotics revolutionized medicine",
]


def main():
    print("=" * 70)
    print("Phase 131: Jarzynski Equality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # At each layer transition, compute:
    # W = work = change in "energy" = change in -log P(correct)
    # delta_F = free energy diff from equilibrium measurement

    # Collect per-prompt entropy at each layer
    all_S = []  # [prompt][layer]
    all_logp = []  # [prompt][layer] = log probability of top token

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_vals = []
        logp_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            logp = torch.log(probs.max() + 1e-10).item()
            S_vals.append(S if not np.isnan(S) else 0)
            logp_vals.append(logp if not np.isnan(logp) else -20)
        all_S.append(S_vals)
        all_logp.append(logp_vals)

    # For each layer transition L -> L+1:
    # W_i = -logp(L+1) - (-logp(L)) = "work" done on prompt i
    # delta_F_eq = -mean(logp(L+1)) + mean(logp(L)) = equilibrium free energy change
    # Jarzynski: exp(-W_mean) should be close to exp(-delta_F_eq)

    jarzynski_ratio = []  # <exp(-W)> / exp(-delta_F)
    W_mean = []
    delta_F = []
    W_var = []

    beta = 1.0  # effective inverse temperature

    for li in range(n_layers - 1):
        W_vals = []
        for pi in range(len(PROMPTS)):
            w = (-all_logp[pi][li + 1]) - (-all_logp[pi][li])
            W_vals.append(w)

        W_avg = np.mean(W_vals)
        W_v = np.var(W_vals)
        dF = W_avg  # second law: delta_F <= <W>

        # Jarzynski: exp(-beta * delta_F) = <exp(-beta * W)>
        exp_neg_W = np.mean([np.exp(-beta * w) for w in W_vals])
        jarzynski_F = -np.log(exp_neg_W + 1e-20) / beta

        # Ratio
        ratio = exp_neg_W / (np.exp(-beta * W_avg) + 1e-20)

        jarzynski_ratio.append(float(ratio))
        W_mean.append(float(W_avg))
        delta_F.append(float(jarzynski_F))
        W_var.append(float(W_v))

    layers_t = np.arange(n_layers - 1) + 0.5

    # Dissipated work W_diss = <W> - delta_F (always >= 0 by 2nd law)
    W_diss = [wm - df for wm, df in zip(W_mean, delta_F)]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Work profile
    axes[0,0].plot(layers_t, W_mean, 'o-', color='#c0392b', markersize=4, linewidth=2,
                   label='$\\langle W \\rangle$')
    axes[0,0].plot(layers_t, delta_F, 's-', color='#2980b9', markersize=4, linewidth=2,
                   label='$\\Delta F$ (Jarzynski)')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,0].axhline(y=0, color='gray', linewidth=0.5)
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Energy')
    axes[0,0].set_title('(a) Work vs Free Energy')
    axes[0,0].legend(fontsize=8)

    # (b) Jarzynski ratio
    axes[0,1].plot(layers_t, jarzynski_ratio, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0,1].axhline(y=1, color='black', linestyle='--', label='Jarzynski satisfied')
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$\\langle e^{-W} \\rangle / e^{-\\langle W \\rangle}$')
    axes[0,1].set_title('(b) Jarzynski Ratio')
    axes[0,1].legend(fontsize=8)

    # (c) Dissipated work
    colors_c = ['#c0392b' if d > 0 else '#2980b9' for d in W_diss]
    axes[0,2].bar(layers_t, W_diss, color=colors_c, alpha=0.7, edgecolor='black', width=0.8)
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=0, color='black', linewidth=0.5)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$W_{diss} = \\langle W \\rangle - \\Delta F$')
    axes[0,2].set_title('(c) Dissipated Work (Irreversibility)')

    # (d) Work variance
    axes[1,0].plot(layers_t, W_var, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('Var($W$)')
    axes[1,0].set_title('(d) Work Fluctuations')

    # (e) Crooks relation check: log(<exp(-W)>) vs <W>
    log_exp_W = [np.log(r * np.exp(-w) + 1e-20) for r, w in zip(jarzynski_ratio, W_mean)]
    axes[1,1].scatter(W_mean, log_exp_W, c=layers_t, cmap='coolwarm', s=60, edgecolors='black')
    axes[1,1].plot([min(W_mean), max(W_mean)], [min(W_mean), max(W_mean)], 'k--', alpha=0.5)
    axes[1,1].set_xlabel('$\\langle W \\rangle$')
    axes[1,1].set_ylabel('$\\log\\langle e^{-W} \\rangle$')
    axes[1,1].set_title('(e) Crooks Consistency')

    # (f) Summary
    pre_diss = np.mean(W_diss[:int(L0)])
    post_diss = np.mean(W_diss[int(L0):])
    jr_mean = np.mean(jarzynski_ratio)

    summary = (
        f"Jarzynski Equality Test\n\n"
        f"Mean Jarzynski ratio: {jr_mean:.3f}\n"
        f"(1.0 = exact equality)\n\n"
        f"Dissipation pre-L0: {pre_diss:.4f}\n"
        f"Dissipation post-L0: {post_diss:.4f}\n\n"
        f"{'MORE' if post_diss > pre_diss else 'LESS'} irreversible\n"
        f"after transition\n\n"
        f"Jarzynski {'HOLDS' if 0.5 < jr_mean < 2.0 else 'VIOLATED'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 131: Jarzynski Equality', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase131_jarzynski')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Jarzynski ratio: {jr_mean:.3f}")
    print(f"Dissipation: pre={pre_diss:.4f}, post={post_diss:.4f}")
    print(f"{'='*70}")

    save_results('phase131_jarzynski', {
        'experiment': 'Jarzynski Equality',
        'jarzynski_ratio': jarzynski_ratio,
        'W_mean': W_mean,
        'delta_F': delta_F,
        'W_diss': W_diss,
        'summary': {
            'jr_mean': float(jr_mean),
            'pre_diss': float(pre_diss),
            'post_diss': float(post_diss),
        }
    })


if __name__ == '__main__':
    main()
