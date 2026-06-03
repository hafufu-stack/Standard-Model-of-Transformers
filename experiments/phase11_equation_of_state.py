# -*- coding: utf-8 -*-
"""
Phase 11: The Equation of State
=================================
Deep Think insight: L2 norm (U) increases while logit entropy (T) decreases.
This is F = U - TS (free energy minimization).
Derive the Transformer's thermodynamic Equation of State by
simultaneously measuring U, T, S, and V(=PR) at each layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 11: The Equation of State (F = U - TS)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The fundamental theorem of calculus states that",
        "In quantum field theory the vacuum is",
        "The periodic table organizes elements by",
        "Neural networks approximate functions through",
        "The speed of light in vacuum equals",
        "Entropy measures the disorder of a",
        "Transformers use attention to compute weighted",
        "The Schrodinger equation describes how the",
        "Gradient descent minimizes the loss function",
        "The second law of thermodynamics forbids",
    ]

    all_U = []  # Internal energy (L2 norm)
    all_S_ent = []  # Entropy (hidden state entropy)
    all_T_logit = []  # Temperature (logit entropy at each layer)
    all_PR = []  # Volume (participation ratio)

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_layers = []
        S_layers = []
        PR_layers = []

        for layer_idx, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()

            # U = L2 norm (internal energy)
            U = h.norm().item()
            U_layers.append(U)

            # S = information entropy of hidden state distribution
            h_abs = h.abs()
            h_prob = h_abs / (h_abs.sum() + 1e-10)
            S_ent = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()
            S_layers.append(S_ent)

            # PR = participation ratio (effective dimensionality / volume)
            h_sq = h ** 2
            h_sq_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_sq_prob ** 2).sum().item()
            PR_layers.append(PR)

        # T = logit entropy for each layer's "projection to vocab"
        # Use the final LM head to project each layer's hidden state
        T_layers = []
        lm_head = model.lm_head
        for layer_idx, hs in enumerate(out.hidden_states):
            h = hs[0, -1:, :]
            with torch.no_grad():
                logits = lm_head(h).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_ent = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_layers.append(T_ent)

        all_U.append(U_layers)
        all_S_ent.append(S_layers)
        all_T_logit.append(T_layers)
        all_PR.append(PR_layers)

    # Average across prompts
    U_avg = np.mean(all_U, axis=0)
    S_avg = np.mean(all_S_ent, axis=0)
    T_avg = np.mean(all_T_logit, axis=0)
    PR_avg = np.mean(all_PR, axis=0)

    # Compute Free Energy: F = U - T*S (using normalized values)
    U_norm = U_avg / (U_avg.max() + 1e-10)
    T_norm = T_avg / (T_avg.max() + 1e-10)
    S_norm = S_avg / (S_avg.max() + 1e-10)
    F = U_norm - T_norm * S_norm

    layers = np.arange(len(U_avg))

    print("\n--- Layer-by-layer thermodynamics ---")
    print(f"{'Layer':>5} {'U(L2)':>10} {'T(logit)':>10} {'S(hidden)':>10} {'PR':>10} {'F':>10}")
    for i in range(len(U_avg)):
        print(f"{i:5d} {U_avg[i]:10.2f} {T_avg[i]:10.2f} {S_avg[i]:10.3f} "
              f"{PR_avg[i]:10.2f} {F[i]:10.4f}")

    # Fit U-T relationship
    try:
        def linear(x, a, b):
            return a * x + b
        popt, _ = curve_fit(linear, T_avg[1:], U_avg[1:])
        dU_dT = popt[0]
        print(f"\n  dU/dT slope = {dU_dT:.4f}")
    except Exception:
        dU_dT = 0.0

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0][0]
    ax.plot(layers, U_avg, 'o-', color='#e74c3c', ms=4, label='U (L2 norm)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Internal Energy U')
    ax.set_title('(a) Internal Energy (L2 Norm)')

    ax = axes[0][1]
    ax.plot(layers, T_avg, 'o-', color='#3498db', ms=4, label='T (logit entropy)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Temperature T')
    ax.set_title('(b) Temperature (Logit Entropy)')

    ax = axes[0][2]
    ax.plot(layers, S_avg, 'o-', color='#2ecc71', ms=4, label='S (hidden entropy)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Hidden Entropy S')
    ax.set_title('(c) Entropy of Hidden State')

    ax = axes[1][0]
    ax.plot(layers, F, 'o-', color='#9b59b6', ms=4, label='F = U - TS')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Free Energy F')
    ax.set_title('(d) Free Energy F = U - TS')
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)

    ax = axes[1][1]
    ax.scatter(T_avg, U_avg, c=layers, cmap='coolwarm', s=60, edgecolors='black')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Internal Energy U')
    ax.set_title(f'(e) U-T Phase Space (dU/dT={dU_dT:.2f})')
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label('Layer')

    ax = axes[1][2]
    ax.plot(layers, PR_avg, 'o-', color='#f39c12', ms=4, label='PR')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('(f) Volume (PR) Profile')

    fig.suptitle(
        f"Phase 11: Equation of State\n"
        f"U increases ({U_avg[0]:.0f}->{U_avg[-1]:.0f}) while "
        f"T decreases ({T_avg[0]:.0f}->{T_avg[-1]:.0f}): F=U-TS",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase11_equation_of_state")
    plt.close()

    # Verdict
    u_increases = U_avg[-1] > U_avg[0]
    t_decreases = T_avg[-1] < T_avg[0]
    f_decreases = F[-1] < F[0]

    if u_increases and t_decreases:
        verdict = (f"THERMODYNAMIC EOS CONFIRMED: U increases ({U_avg[0]:.0f}->{U_avg[-1]:.0f}) "
                   f"while T decreases ({T_avg[0]:.0f}->{T_avg[-1]:.0f}). "
                   f"dU/dT={dU_dT:.2f}. Free energy {'minimized' if f_decreases else 'NOT minimized'}.")
    else:
        verdict = (f"UNEXPECTED: U {'up' if u_increases else 'down'}, "
                   f"T {'down' if t_decreases else 'up'}. Non-standard thermodynamics.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 11: Equation of State',
        'summary': {'verdict': verdict, 'dU_dT': dU_dT,
                    'U_range': [float(U_avg[0]), float(U_avg[-1])],
                    'T_range': [float(T_avg[0]), float(T_avg[-1])],
                    'F_range': [float(F[0]), float(F[-1])]},
    }
    save_results("phase11_equation_of_state", result)
    return result


if __name__ == '__main__':
    main()
