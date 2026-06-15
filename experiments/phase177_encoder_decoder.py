# -*- coding: utf-8 -*-
"""
Phase 177: Encoder-Decoder Duality
Study heat exchange dynamics at L0 boundary within decoder-only model.
Pre-L0 layers act as "encoder" (absorbing entropy), post-L0 as "decoder"
(concentrating information). Measure energy/entropy flow at L0.
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
]


def main():
    print("=" * 70)
    print("Phase 177: Encoder-Decoder Duality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21  # Critical layer

    all_U, all_T, all_S = [], [], []
    all_dU, all_dT, all_dS = [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_vals, T_vals, S_vals = [], [], []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()
            U = h.norm().item()

            # Hidden state entropy
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S_h = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()

            U_vals.append(U if not np.isnan(U) else 0)
            T_vals.append(T if not np.isnan(T) else 0)
            S_vals.append(S_h if not np.isnan(S_h) else 0)

        all_U.append(U_vals)
        all_T.append(T_vals)
        all_S.append(S_vals)

        # Layer-to-layer changes
        dU = [U_vals[i+1] - U_vals[i] for i in range(n_layers-1)]
        dT = [T_vals[i+1] - T_vals[i] for i in range(n_layers-1)]
        dS = [S_vals[i+1] - S_vals[i] for i in range(n_layers-1)]
        all_dU.append(dU)
        all_dT.append(dT)
        all_dS.append(dS)

    U_mean = np.mean(all_U, axis=0)
    T_mean = np.mean(all_T, axis=0)
    S_mean = np.mean(all_S, axis=0)
    dU_mean = np.mean(all_dU, axis=0)
    dT_mean = np.mean(all_dT, axis=0)
    dS_mean = np.mean(all_dS, axis=0)

    # Encoder (pre-L0) vs Decoder (post-L0) statistics
    enc_dU = np.mean(dU_mean[:L0])
    dec_dU = np.mean(dU_mean[L0:])
    enc_dT = np.mean(dT_mean[:L0])
    dec_dT = np.mean(dT_mean[L0:])
    enc_dS = np.mean(dS_mean[:L0])
    dec_dS = np.mean(dS_mean[L0:])

    # Heat exchange at L0 (Q = T * dS)
    Q_at_L0 = T_mean[L0] * dS_mean[min(L0, len(dS_mean)-1)]
    energy_jump_L0 = dU_mean[min(L0, len(dU_mean)-1)]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers = np.arange(n_layers)
    layers_t = np.arange(n_layers - 1) + 0.5

    # (a) Energy flow
    axes[0, 0].fill_between(layers_t[:L0], dU_mean[:L0], alpha=0.3, color='#e74c3c', label='Encoder')
    axes[0, 0].fill_between(layers_t[L0:], dU_mean[L0:], alpha=0.3, color='#3498db', label='Decoder')
    axes[0, 0].plot(layers_t, dU_mean, 'k-', linewidth=1.5)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$ (Heat Exchanger)')
    axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('$\\Delta U$ per layer')
    axes[0, 0].set_title('(a) Energy Flow: Encoder vs Decoder')
    axes[0, 0].legend(fontsize=7)

    # (b) Entropy flow
    axes[0, 1].fill_between(layers_t[:L0], dS_mean[:L0], alpha=0.3, color='#e74c3c')
    axes[0, 1].fill_between(layers_t[L0:], dS_mean[L0:], alpha=0.3, color='#3498db')
    axes[0, 1].plot(layers_t, dS_mean, 'k-', linewidth=1.5)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].axhline(y=0, color='gray', linewidth=0.5)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$\\Delta S$ per layer')
    axes[0, 1].set_title('(b) Entropy Flow')

    # (c) Temperature profile
    axes[0, 2].plot(layers, T_mean, 'o-', color='#e74c3c', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].fill_between(layers[:L0+1], T_mean[:L0+1], alpha=0.1, color='#e74c3c')
    axes[0, 2].fill_between(layers[L0:], T_mean[L0:], alpha=0.1, color='#3498db')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Temperature $T$')
    axes[0, 2].set_title('(c) Temperature: Hot Encoder -> Cold Decoder')

    # (d) Heat exchange at L0
    Q_profile = [T_mean[i] * dS_mean[i] for i in range(n_layers - 1)]
    colors_q = ['#e74c3c' if q > 0 else '#3498db' for q in Q_profile]
    axes[1, 0].bar(layers_t, Q_profile, color=colors_q, alpha=0.7, edgecolor='black', width=0.8)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$Q = T \\cdot \\Delta S$')
    axes[1, 0].set_title('(d) Heat Exchange Profile')

    # (e) Thermodynamic cycle T-S diagram
    axes[1, 1].plot(S_mean, T_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1, 1].scatter(S_mean[0], T_mean[0], s=100, c='green', marker='^', zorder=5, label='Input')
    axes[1, 1].scatter(S_mean[-1], T_mean[-1], s=100, c='red', marker='v', zorder=5, label='Output')
    axes[1, 1].scatter(S_mean[L0], T_mean[L0], s=150, c='#f39c12', marker='*', zorder=5, label='$L_0$ (exchanger)')
    axes[1, 1].set_xlabel('Hidden State Entropy $S$')
    axes[1, 1].set_ylabel('Temperature $T$')
    axes[1, 1].set_title('(e) T-S Diagram (Thermodynamic Cycle)')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = (
        f"Encoder-Decoder Duality\n\n"
        f"ENCODER (L<{L0}):\n"
        f"  dU/layer = {enc_dU:.3f} (absorbs energy)\n"
        f"  dT/layer = {enc_dT:.3f}\n"
        f"  dS/layer = {enc_dS:.4f}\n\n"
        f"DECODER (L>{L0}):\n"
        f"  dU/layer = {dec_dU:.3f}\n"
        f"  dT/layer = {dec_dT:.3f}\n"
        f"  dS/layer = {dec_dS:.4f}\n\n"
        f"Heat at L0: Q = {Q_at_L0:.3f}\n"
        f"L0 = Heat Exchanger"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 177: Encoder-Decoder Duality', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase177_encoder_decoder')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Encoder (L<{L0}): dU={enc_dU:.3f}, dT={enc_dT:.3f}, dS={enc_dS:.4f}")
    print(f"Decoder (L>{L0}): dU={dec_dU:.3f}, dT={dec_dT:.3f}, dS={dec_dS:.4f}")
    print(f"Heat exchange at L0: Q={Q_at_L0:.3f}")
    print(f"{'=' * 70}")

    save_results('phase177_encoder_decoder', {
        'experiment': 'Encoder-Decoder Duality',
        'L0': L0,
        'encoder': {'dU': float(enc_dU), 'dT': float(enc_dT), 'dS': float(enc_dS)},
        'decoder': {'dU': float(dec_dU), 'dT': float(dec_dT), 'dS': float(dec_dS)},
        'Q_at_L0': float(Q_at_L0),
        'U_mean': [float(x) for x in U_mean],
        'T_mean': [float(x) for x in T_mean],
        'S_mean': [float(x) for x in S_mean],
    })


if __name__ == '__main__':
    main()
