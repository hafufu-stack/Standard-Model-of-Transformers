# -*- coding: utf-8 -*-
"""
Phase 148: The Loschmidt Echo
Measure time-reversal asymmetry precisely by perturbing the final hidden state
and propagating backward. How much information is lost (irreversibly)?
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
]


def main():
    print("=" * 70)
    print("Phase 148: Loschmidt Echo")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # For each prompt, collect hidden states at every layer
    # Then for each layer L, add small perturbation epsilon to h_L,
    # propagate forward to h_final, and measure how much the output changes.
    # The "echo" measures sensitivity to perturbation at different depths.

    epsilon_sizes = [0.001, 0.01, 0.1, 1.0]
    all_echoes = {eps: [[] for _ in range(n_layers)] for eps in epsilon_sizes}
    all_fidelity = {eps: [[] for _ in range(n_layers)] for eps in epsilon_sizes}

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Baseline: get unperturbed final output
        with torch.no_grad():
            out_base = model(**inp, output_hidden_states=True)
        h_final_base = out_base.hidden_states[-1][0, -1, :].float()
        logits_base = out_base.logits[0, -1, :].float()
        probs_base = torch.softmax(logits_base, dim=-1)

        # For each layer, inject perturbation and measure echo
        for li in range(n_layers):
            if li >= len(model.model.layers):
                # Last layer = output embedding, skip
                for eps in epsilon_sizes:
                    all_echoes[eps][li].append(0)
                    all_fidelity[eps][li].append(1)
                continue

            for eps in epsilon_sizes:
                # Create hook that adds perturbation at layer li
                perturb_dir = torch.randn(1, 1, model.config.hidden_size).to(device)
                perturb_dir = perturb_dir / (perturb_dir.norm() + 1e-10) * eps

                def make_hook(perturbation):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                            p = perturbation.to(h.dtype)
                            h[:, -1:, :] += p
                            return (h,) + output[1:]
                        return output
                    return hook_fn

                hook = model.model.layers[li].register_forward_hook(
                    make_hook(perturb_dir))

                with torch.no_grad():
                    out_pert = model(**inp, output_hidden_states=True)
                hook.remove()

                h_final_pert = out_pert.hidden_states[-1][0, -1, :].float()
                logits_pert = out_pert.logits[0, -1, :].float()
                probs_pert = torch.softmax(logits_pert, dim=-1)

                # Loschmidt echo = overlap between original and perturbed
                cos_sim = torch.nn.functional.cosine_similarity(
                    h_final_base.unsqueeze(0), h_final_pert.unsqueeze(0)).item()

                # KL divergence between probability distributions
                kl = torch.sum(probs_base * torch.log(
                    (probs_base + 1e-10) / (probs_pert + 1e-10))).item()
                kl = max(0, min(kl, 100))  # clip

                all_echoes[eps][li].append(float(kl))
                all_fidelity[eps][li].append(float(cos_sim))

    # Averages
    avg_echo = {eps: [np.mean(v) if v else 0 for v in all_echoes[eps]] for eps in epsilon_sizes}
    avg_fid = {eps: [np.mean(v) if v else 0 for v in all_fidelity[eps]] for eps in epsilon_sizes}

    layers = np.arange(n_layers)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Echo (KL divergence) vs layer for different epsilon
    for eps in epsilon_sizes:
        axes[0,0].plot(layers, avg_echo[eps], 'o-', markersize=3, linewidth=2,
                      label=f'$\\epsilon$={eps}')
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,0].set_xlabel('Perturbation Layer')
    axes[0,0].set_ylabel('KL Divergence')
    axes[0,0].set_title('(a) Loschmidt Echo')
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_yscale('log')

    # (b) Fidelity (cosine sim) vs layer
    for eps in epsilon_sizes:
        axes[0,1].plot(layers, avg_fid[eps], 'o-', markersize=3, linewidth=2,
                      label=f'$\\epsilon$={eps}')
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Perturbation Layer')
    axes[0,1].set_ylabel('Cosine Similarity')
    axes[0,1].set_title('(b) Fidelity')
    axes[0,1].legend(fontsize=8)

    # (c) Lyapunov exponent: d(log KL)/d(layer) for eps=0.01
    eps_ref = 0.01
    echo_ref = avg_echo[eps_ref]
    log_echo = [np.log(e + 1e-10) for e in echo_ref]
    lyapunov = np.gradient(log_echo)
    axes[0,2].plot(layers, lyapunov, 'o-', color='#c0392b', markersize=3, linewidth=2)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=0, color='gray', linewidth=0.5)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$\\lambda$ (Lyapunov)')
    axes[0,2].set_title('(c) Local Lyapunov Exponent')

    # (d) Echo ratio: post/pre transition
    pre_echo = np.mean(echo_ref[:20])
    post_echo = np.mean(echo_ref[20:])
    axes[1,0].bar(['Pre-$L_0$', 'Post-$L_0$'], [pre_echo, post_echo],
                  color=['#2980b9', '#c0392b'], alpha=0.8, edgecolor='black')
    axes[1,0].set_ylabel('Mean KL Divergence')
    axes[1,0].set_title(f'(d) Echo Asymmetry: {post_echo/pre_echo:.2f}x')

    # (e) Irreversibility profile
    irrev = [1 - f for f in avg_fid[0.1]]
    colors_e = ['#c0392b' if r > 0.05 else '#27ae60' for r in irrev]
    axes[1,1].bar(layers, irrev, color=colors_e, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Irreversibility (1 - cosine)')
    axes[1,1].set_title('(e) Irreversibility per Layer')

    # (f) Summary
    max_echo_layer = np.argmax(echo_ref)
    summary = (
        f"Loschmidt Echo\n\n"
        f"Most sensitive layer: L{max_echo_layer}\n"
        f"Echo asymmetry: {post_echo/(pre_echo+1e-10):.2f}x\n\n"
        f"Pre-L0 KL: {pre_echo:.4f}\n"
        f"Post-L0 KL: {post_echo:.4f}\n\n"
        f"{'IRREVERSIBLE' if post_echo > pre_echo else 'REVERSIBLE'}\n"
        f"post-transition\n\n"
        f"Max Lyapunov: {max(lyapunov):.3f}\n"
        f"Min Lyapunov: {min(lyapunov):.3f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 148: Loschmidt Echo (Time-Reversal Asymmetry)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase148_loschmidt')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Most sensitive layer: L{max_echo_layer}")
    print(f"Echo asymmetry: {post_echo/(pre_echo+1e-10):.2f}x")
    print(f"{'='*70}")

    save_results('phase148_loschmidt', {
        'experiment': 'Loschmidt Echo',
        'echo_eps0.01': echo_ref,
        'summary': {
            'most_sensitive_layer': int(max_echo_layer),
            'echo_asymmetry': float(post_echo/(pre_echo+1e-10)),
            'pre_echo': float(pre_echo),
            'post_echo': float(post_echo),
        }
    })


if __name__ == '__main__':
    main()
