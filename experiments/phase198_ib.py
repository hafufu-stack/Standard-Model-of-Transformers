# -*- coding: utf-8 -*-
"""
Phase 198: Information Bottleneck Phase Diagram
=================================================
The Information Bottleneck (IB) theory predicts a phase transition
between memorization and compression. Where does this transition
occur in the layer stack?

Measure I(X;T) [input information preserved] and I(T;Y) [output
information available] at each layer, constructing the IB plane.

The IB curve I(T;Y) vs I(X;T) should show compression in later layers.
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
]


def main():
    print("=" * 70)
    print("Phase 198: Information Bottleneck Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21

    all_IxT = []  # I(X;T) - input info preserved
    all_ITy = []  # I(T;Y) - output info available
    all_compression = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Get initial representation and final output
        h0 = out.hidden_states[0][0, -1, :].float()
        final_logits = out.logits[0, -1, :].float()
        final_probs = torch.softmax(final_logits, dim=-1)

        IxT_vals = []
        ITy_vals = []
        comp_vals = []

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            # I(X;T) proxy: cosine similarity with input
            # High cosine = high mutual info with input (memorization)
            cos_input = torch.nn.functional.cosine_similarity(
                h.unsqueeze(0), h0.unsqueeze(0)).item()
            IxT = max(0, cos_input)  # Clamp to [0,1]
            IxT_vals.append(IxT)

            # I(T;Y) proxy: how well this layer's logits predict the output
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)

            # KL from this layer to final = "information gap"
            kl = (final_probs * torch.log((final_probs + 1e-10) / (probs + 1e-10))).sum().item()
            # I(T;Y) = H(Y) - H(Y|T) ~ 1 - KL(final||current)/H(final)
            H_final = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
            ITy = max(0, 1 - kl / (H_final + 1e-10))
            ITy_vals.append(ITy if not np.isnan(ITy) else 0)

            # Compression ratio
            h_ent = -(h ** 2 / (h ** 2).sum() * torch.log(h ** 2 / (h ** 2).sum() + 1e-10)).sum().item()
            comp_vals.append(h_ent if not np.isnan(h_ent) else 0)

        all_IxT.append(IxT_vals)
        all_ITy.append(ITy_vals)
        all_compression.append(comp_vals)

    IxT_mean = np.mean(all_IxT, axis=0)
    ITy_mean = np.mean(all_ITy, axis=0)
    comp_mean = np.mean(all_compression, axis=0)

    # Phase transition: where does the system switch from memorization to compression?
    # dI(X;T)/dl < 0 and dI(T;Y)/dl > 0 simultaneously
    dIxT = np.gradient(IxT_mean)
    dITy = np.gradient(ITy_mean)

    # IB transition: layer where dIxT becomes negative and dITy is positive
    transition_layer = None
    for i in range(1, len(dIxT)):
        if dIxT[i] < 0 and dITy[i] > 0:
            transition_layer = i
            break

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) IB plane: I(T;Y) vs I(X;T)
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    for i in range(n_layers - 1):
        axes[0, 0].annotate('', xy=(IxT_mean[i+1], ITy_mean[i+1]),
                            xytext=(IxT_mean[i], ITy_mean[i]),
                            arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    axes[0, 0].scatter(IxT_mean, ITy_mean, c=np.arange(n_layers), cmap='viridis',
                        s=40, edgecolors='black', zorder=5)
    axes[0, 0].scatter([IxT_mean[0]], [ITy_mean[0]], s=150, marker='s', c='green',
                        edgecolors='black', zorder=10, label='Layer 0')
    axes[0, 0].scatter([IxT_mean[-1]], [ITy_mean[-1]], s=150, marker='*', c='red',
                        edgecolors='black', zorder=10, label=f'Layer {n_layers-1}')
    if L0 < n_layers:
        axes[0, 0].scatter([IxT_mean[L0]], [ITy_mean[L0]], s=150, marker='D', c='#f39c12',
                            edgecolors='black', zorder=10, label=f'$L_0$={L0}')
    axes[0, 0].set_xlabel('$I(X;T)$ - Input Info Preserved')
    axes[0, 0].set_ylabel('$I(T;Y)$ - Output Info Available')
    axes[0, 0].set_title('(a) Information Bottleneck Plane')
    axes[0, 0].legend(fontsize=7)

    # (b) I(X;T) and I(T;Y) profiles
    axes[0, 1].plot(np.arange(n_layers), IxT_mean, 'o-', color='#e74c3c', markersize=4,
                    linewidth=2, label='$I(X;T)$ (input)')
    axes[0, 1].plot(np.arange(n_layers), ITy_mean, 's-', color='#3498db', markersize=4,
                    linewidth=2, label='$I(T;Y)$ (output)')
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    if transition_layer:
        axes[0, 1].axvline(x=transition_layer, color='#2ecc71', linewidth=2, linestyle=':',
                            label=f'IB transition (L={transition_layer})')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Mutual Information Proxy')
    axes[0, 1].set_title('(b) Information Profiles')
    axes[0, 1].legend(fontsize=7)

    # (c) Compression profile
    axes[0, 2].plot(np.arange(n_layers), comp_mean, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0, 2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Hidden State Entropy')
    axes[0, 2].set_title('(c) Compression Profile')

    # (d) Derivatives
    axes[1, 0].plot(np.arange(n_layers), dIxT, 'o-', color='#e74c3c', markersize=3, linewidth=2,
                    label='$dI(X;T)/dl$')
    axes[1, 0].plot(np.arange(n_layers), dITy, 's-', color='#3498db', markersize=3, linewidth=2,
                    label='$dI(T;Y)/dl$')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Rate of Change')
    axes[1, 0].set_title('(d) Information Flow Rates')
    axes[1, 0].legend(fontsize=7)

    # (e) IB tradeoff: I(T;Y) / I(X;T)
    tradeoff = ITy_mean / (IxT_mean + 1e-10)
    axes[1, 1].plot(np.arange(n_layers), tradeoff, 'o-', color='#2ecc71', markersize=4, linewidth=2)
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('$I(T;Y) / I(X;T)$')
    axes[1, 1].set_title('(e) IB Efficiency (output/input ratio)')

    # (f) Summary
    summary = (
        f"Information Bottleneck\n\n"
        f"IB transition layer: {transition_layer}\n"
        f"L0 (thermodynamic): {L0}\n\n"
        f"I(X;T) at L0: {IxT_mean[L0]:.4f}\n"
        f"I(T;Y) at L0: {ITy_mean[L0]:.4f}\n\n"
        f"IB efficiency at L0:\n"
        f"  {tradeoff[L0]:.4f}\n\n"
        f"Max IB efficiency:\n"
        f"  {max(tradeoff):.4f} at L={np.argmax(tradeoff)}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 198: Information Bottleneck Phase Diagram', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase198_ib')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"IB transition: layer {transition_layer}")
    print(f"I(X;T) at L0: {IxT_mean[L0]:.4f}, I(T;Y) at L0: {ITy_mean[L0]:.4f}")
    print(f"Max IB efficiency: {max(tradeoff):.4f} at L={np.argmax(tradeoff)}")
    print(f"{'=' * 70}")

    save_results('phase198_ib', {
        'experiment': 'Information Bottleneck',
        'IxT_mean': [float(x) for x in IxT_mean],
        'ITy_mean': [float(x) for x in ITy_mean],
        'summary': {
            'transition_layer': int(transition_layer) if transition_layer else None,
            'IxT_at_L0': float(IxT_mean[L0]),
            'ITy_at_L0': float(ITy_mean[L0]),
            'max_efficiency': float(max(tradeoff)),
            'max_eff_layer': int(np.argmax(tradeoff)),
        }
    })


if __name__ == '__main__':
    main()
