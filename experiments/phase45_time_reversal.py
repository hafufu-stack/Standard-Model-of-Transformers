# -*- coding: utf-8 -*-
"""
Phase 45: Holographic Time-Reversal
Reconstruct intermediate layer states from final output using SVD/linear regression.
Measures how much information is preserved at the boundary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 45: Holographic Time-Reversal")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration through",
        "In the standard model of particle physics, quarks interact via the strong force mediated by",
        "The evolutionary theory of natural selection proposed by Darwin explains how species",
        "The blockchain technology underlying Bitcoin uses cryptographic hash functions to ensure",
        "Quantum entanglement allows two particles to be correlated regardless of the distance",
        "The Renaissance period in Europe was characterized by a revival of interest in",
        "Machine learning algorithms can be broadly classified into supervised, unsupervised, and",
        "The human immune system consists of innate and adaptive components that work together to",
        "Climate models predict that global temperatures will rise by several degrees Celsius",
        "The double-slit experiment demonstrates the wave-particle duality of matter and",
    ]

    n_layers = len(model.model.layers)
    all_hidden_states = []  # (n_prompts, n_layers+1, hidden_dim)

    print(f"\n--- Extracting hidden states from {len(prompts)} prompts ---")
    for i, prompt in enumerate(prompts):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Extract last-token hidden state at each layer
        states = []
        for hs in out.hidden_states:
            states.append(hs[0, -1, :].cpu().float().numpy())
        all_hidden_states.append(states)

    all_hidden_states = np.array(all_hidden_states)  # (n_prompts, n_layers+1, hidden_dim)
    n_prompts, total_layers, hidden_dim = all_hidden_states.shape
    print(f"  Shape: {all_hidden_states.shape}")

    # Split into train and test
    n_train = max(2, n_prompts * 2 // 3)
    train_states = all_hidden_states[:n_train]
    test_states = all_hidden_states[n_train:]
    n_test = len(test_states)

    print(f"  Train: {n_train} prompts, Test: {n_test} prompts")

    # For each layer, learn linear map: final_state -> intermediate_state
    final_layer_idx = total_layers - 1

    reconstruction_results = []

    print(f"\n--- Learning reverse maps (final -> each layer) ---")
    for target_layer in range(total_layers):
        # X = final layer states, Y = target layer states
        X_train = train_states[:, final_layer_idx, :]  # (n_train, hidden_dim)
        Y_train = train_states[:, target_layer, :]     # (n_train, hidden_dim)

        # Use SVD-based pseudoinverse for linear regression: Y = X @ W
        # W = (X^T X)^{-1} X^T Y = pinv(X) @ Y
        try:
            # Regularized least squares
            XtX = X_train.T @ X_train + 1e-4 * np.eye(hidden_dim)
            XtY = X_train.T @ Y_train
            W = np.linalg.solve(XtX, XtY)

            # Test reconstruction
            X_test = test_states[:, final_layer_idx, :]
            Y_test = test_states[:, target_layer, :]
            Y_pred = X_test @ W

            # Cosine similarity
            cos_sims = []
            for j in range(n_test):
                cos = np.dot(Y_test[j], Y_pred[j]) / (np.linalg.norm(Y_test[j]) * np.linalg.norm(Y_pred[j]) + 1e-10)
                cos_sims.append(cos)
            mean_cos = np.mean(cos_sims)

            # L2 relative error
            l2_errors = []
            for j in range(n_test):
                err = np.linalg.norm(Y_test[j] - Y_pred[j]) / (np.linalg.norm(Y_test[j]) + 1e-10)
                l2_errors.append(err)
            mean_l2 = np.mean(l2_errors)

            # SVD of W to check information flow
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            effective_rank = np.sum(S > S[0] * 0.01) if len(S) > 0 else 0

            success = True
        except Exception as e:
            mean_cos = 0
            mean_l2 = 1.0
            effective_rank = 0
            success = False

        reconstruction_results.append({
            'target_layer': target_layer,
            'cosine_similarity': float(mean_cos),
            'l2_relative_error': float(mean_l2),
            'effective_rank': int(effective_rank),
            'success': success,
        })

        if target_layer % 7 == 0:
            print(f"  Layer {target_layer}: cos_sim={mean_cos:.4f}, "
                  f"L2_err={mean_l2:.4f}, rank={effective_rank}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers_plot = [r['target_layer'] for r in reconstruction_results]
    cos_sims_plot = [r['cosine_similarity'] for r in reconstruction_results]
    l2_errors_plot = [r['l2_relative_error'] for r in reconstruction_results]
    ranks_plot = [r['effective_rank'] for r in reconstruction_results]

    # (a) Cosine similarity
    axes[0, 0].plot(layers_plot, cos_sims_plot, marker='o', color='#2ecc71',
                    markersize=3, linewidth=1.5)
    axes[0, 0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Target Layer')
    axes[0, 0].set_ylabel('Cosine Similarity')
    axes[0, 0].set_title('(a) Reconstruction Accuracy (Cosine)')
    axes[0, 0].set_ylim(-0.1, 1.1)

    # (b) L2 relative error
    axes[0, 1].plot(layers_plot, l2_errors_plot, marker='s', color='#e74c3c',
                    markersize=3, linewidth=1.5)
    axes[0, 1].set_xlabel('Target Layer')
    axes[0, 1].set_ylabel('Relative L2 Error')
    axes[0, 1].set_title('(b) Reconstruction Error')

    # (c) Effective rank of reverse map
    axes[1, 0].bar(layers_plot, ranks_plot, color='#3498db', alpha=0.7)
    axes[1, 0].set_xlabel('Target Layer')
    axes[1, 0].set_ylabel('Effective Rank')
    axes[1, 0].set_title('(c) Information Channels (SVD Rank)')

    # (d) Information preservation heatmap
    # How well each layer pair can be predicted
    n_sample_layers = min(total_layers, 14)
    step = max(1, total_layers // n_sample_layers)
    sample_layers = list(range(0, total_layers, step))

    info_matrix = np.zeros((len(sample_layers), len(sample_layers)))
    for si, source_layer in enumerate(sample_layers):
        for ti, target_layer in enumerate(sample_layers):
            X_tr = train_states[:, source_layer, :]
            Y_tr = train_states[:, target_layer, :]
            try:
                XtX = X_tr.T @ X_tr + 1e-4 * np.eye(hidden_dim)
                W = np.linalg.solve(XtX, X_tr.T @ Y_tr)
                X_te = test_states[:, source_layer, :]
                Y_te = test_states[:, target_layer, :]
                Y_pr = X_te @ W
                cos_vals = []
                for j in range(n_test):
                    c = np.dot(Y_te[j], Y_pr[j]) / (np.linalg.norm(Y_te[j]) * np.linalg.norm(Y_pr[j]) + 1e-10)
                    cos_vals.append(c)
                info_matrix[si, ti] = np.mean(cos_vals)
            except Exception:
                info_matrix[si, ti] = 0

    im = axes[1, 1].imshow(info_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(len(sample_layers)))
    axes[1, 1].set_xticklabels([f'L{l}' for l in sample_layers], fontsize=6, rotation=45)
    axes[1, 1].set_yticks(range(len(sample_layers)))
    axes[1, 1].set_yticklabels([f'L{l}' for l in sample_layers], fontsize=6)
    axes[1, 1].set_xlabel('Target Layer')
    axes[1, 1].set_ylabel('Source Layer')
    axes[1, 1].set_title('(d) Cross-Layer Information Flow')
    plt.colorbar(im, ax=axes[1, 1], label='Cosine Sim')

    fig.suptitle('Phase 45: Holographic Time-Reversal', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase45_time_reversal')
    plt.close()

    # === Verdict ===
    early_cos = np.mean([r['cosine_similarity'] for r in reconstruction_results[:total_layers//3]])
    mid_cos = np.mean([r['cosine_similarity'] for r in reconstruction_results[total_layers//3:2*total_layers//3]])
    late_cos = np.mean([r['cosine_similarity'] for r in reconstruction_results[2*total_layers//3:]])

    print(f"\n{'='*70}")
    print(f"VERDICT: Reverse reconstruction from final layer: "
          f"Early={early_cos:.3f}, Mid={mid_cos:.3f}, Late={late_cos:.3f}. "
          f"{'Strong holographic encoding' if early_cos > 0.5 else 'Weak holographic encoding'}: "
          f"final layer retains {early_cos*100:.0f}% of early-layer information.")
    print(f"{'='*70}")

    save_results('phase45_time_reversal', {
        'experiment': 'Holographic Time-Reversal',
        'reconstruction_results': reconstruction_results,
        'summary': {
            'early_layer_cos': float(early_cos),
            'mid_layer_cos': float(mid_cos),
            'late_layer_cos': float(late_cos),
        }
    })


if __name__ == '__main__':
    main()
