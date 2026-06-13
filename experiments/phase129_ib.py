# -*- coding: utf-8 -*-
"""
Phase 129: Information Bottleneck
Measure mutual information I(X;T_l) and I(T_l;Y) at each layer.
X = input tokens, T_l = hidden state at layer l, Y = output distribution.
The Information Bottleneck predicts I(X;T) decreases while I(T;Y)
increases. How does this relate to the phase transition?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
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
    print("Phase 129: Information Bottleneck")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # Collect hidden states and output distributions at each layer
    all_hidden = []  # [prompt][layer] = PCA-reduced vector
    all_input_ids = []
    all_output_probs = []  # [prompt] = final output distribution

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        all_input_ids.append(inp['input_ids'][0].cpu().numpy())

        # Final output
        final_probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        all_output_probs.append(final_probs.cpu().numpy())

        # Hidden at each layer (use last token, PCA to 50d later)
        hs = [out.hidden_states[li][0, -1, :].float().cpu().numpy() for li in range(n_layers)]
        all_hidden.append(hs)

    # Compute proxy for I(X; T_l): cosine similarity between input embedding and T_l
    # Higher sim = more input info retained
    I_X_T = []
    for li in range(n_layers):
        sims = []
        for pi in range(len(PROMPTS)):
            h0 = all_hidden[pi][0]  # input embedding
            hl = all_hidden[pi][li]
            cos = np.dot(h0, hl) / (np.linalg.norm(h0) * np.linalg.norm(hl) + 1e-10)
            sims.append(cos)
        I_X_T.append(float(np.mean(sims)))

    # Compute proxy for I(T_l; Y): how much does T_l predict the output?
    # Use KL divergence between layer-l logits and final logits
    I_T_Y = []
    for li in range(n_layers):
        kls = []
        for pi in range(len(PROMPTS)):
            hs_tensor = torch.tensor(all_hidden[pi][li], device=device).to(model.dtype)
            with torch.no_grad():
                normed = model.model.norm(hs_tensor.unsqueeze(0).unsqueeze(0))
                logits = model.lm_head(normed).squeeze().float()
            probs_l = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_y = all_output_probs[pi]
            # Negative KL = more info about Y
            kl = np.sum(probs_y * np.log((probs_y + 1e-10) / (probs_l + 1e-10)))
            kls.append(float(kl) if not np.isnan(kl) else 10)
        # I(T;Y) proxy = -KL (higher = more info)
        I_T_Y.append(float(-np.mean(kls)))

    layers = np.arange(n_layers)

    # Compression: rate of I(X;T) decrease
    dI_X = np.gradient(I_X_T)
    # Prediction: rate of I(T;Y) increase
    dI_Y = np.gradient(I_T_Y)

    # Phase transition signature
    pre_dIX = np.mean(dI_X[:int(L0)])
    post_dIX = np.mean(dI_X[int(L0):])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) I(X;T) profile
    axes[0, 0].plot(layers, I_X_T, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('$I(X; T_l)$ proxy')
    axes[0, 0].set_title('(a) Input Information Retention')
    axes[0, 0].legend()

    # (b) I(T;Y) profile
    axes[0, 1].plot(layers, I_T_Y, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$I(T_l; Y)$ proxy')
    axes[0, 1].set_title('(b) Output Prediction Quality')

    # (c) Information plane
    axes[0, 2].scatter(I_X_T, I_T_Y, c=layers, cmap='coolwarm', s=60, edgecolors='black')
    for i in range(0, n_layers, 4):
        axes[0, 2].annotate(f'{i}', (I_X_T[i], I_T_Y[i]), fontsize=7)
    axes[0, 2].set_xlabel('$I(X; T)$ (input info)')
    axes[0, 2].set_ylabel('$I(T; Y)$ (output info)')
    axes[0, 2].set_title('(c) Information Plane')

    # (d) Compression rate
    colors_d = ['#c0392b' if d < 0 else '#2980b9' for d in dI_X]
    axes[1, 0].bar(layers, dI_X, color=colors_d, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('$dI(X;T)/dL$')
    axes[1, 0].set_title('(d) Compression Rate')

    # (e) Prediction rate
    colors_e = ['#27ae60' if d > 0 else '#7f8c8d' for d in dI_Y]
    axes[1, 1].bar(layers, dI_Y, color=colors_e, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('$dI(T;Y)/dL$')
    axes[1, 1].set_title('(e) Prediction Building Rate')

    # (f) Summary
    summary = (
        f"Information Bottleneck\n\n"
        f"I(X;T): {I_X_T[0]:.3f} -> {I_X_T[-1]:.3f}\n"
        f"I(T;Y): {I_T_Y[0]:.3f} -> {I_T_Y[-1]:.3f}\n\n"
        f"Compression pre-L0: {pre_dIX:.4f}/layer\n"
        f"Compression post-L0: {post_dIX:.4f}/layer\n\n"
        f"Transition accelerates\n"
        f"{'COMPRESSION' if abs(post_dIX) > abs(pre_dIX) else 'PREDICTION'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 129: Information Bottleneck', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase129_ib')
    plt.close()

    print(f"\n{'='*70}")
    print(f"I(X;T): {I_X_T[0]:.3f} -> {I_X_T[-1]:.3f}")
    print(f"I(T;Y): {I_T_Y[0]:.3f} -> {I_T_Y[-1]:.3f}")
    print(f"{'='*70}")

    save_results('phase129_ib', {
        'experiment': 'Information Bottleneck',
        'I_X_T': [float(v) for v in I_X_T],
        'I_T_Y': [float(v) for v in I_T_Y],
        'summary': {
            'I_X_T_start': float(I_X_T[0]),
            'I_X_T_end': float(I_X_T[-1]),
            'I_T_Y_start': float(I_T_Y[0]),
            'I_T_Y_end': float(I_T_Y[-1]),
        }
    })


if __name__ == '__main__':
    main()
