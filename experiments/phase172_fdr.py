# -*- coding: utf-8 -*-
"""
Phase 172: Fluctuation-Dissipation Relation at Each Layer
Measure the FDR (chi/C_v ratio) at each layer.
In equilibrium, FDR = 1. Deviations indicate non-equilibrium.
Where does the system deviate most from equilibrium?
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
    "DNA encodes the instructions for all living organisms",
    "Thermodynamics governs the flow of energy and entropy",
    "The brain processes information through neural circuits",
    "Climate change is driven by greenhouse gas emissions",
]


def main():
    print("=" * 70)
    print("Phase 172: Fluctuation-Dissipation at Each Layer")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    noise_scale = 0.5
    n_perturb = 5

    all_S = np.zeros((len(PROMPTS), n_layers))
    all_chi = np.zeros((len(PROMPTS), n_layers))  # Response to perturbation

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)

        # Baseline
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            all_S[pi, li] = S if not np.isnan(S) else 0

        # Perturbed: inject noise at each layer and measure response
        for li in range(min(n_layers - 1, len(model.model.layers))):
            responses = []
            for _ in range(n_perturb):
                hook = None
                def make_hook(target_layer, scale):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            noise = torch.randn_like(h) * scale
                            return (h + noise,) + output[1:]
                        return output
                    return hook_fn

                hook = model.model.layers[li].register_forward_hook(
                    make_hook(li, noise_scale))

                with torch.no_grad():
                    out_p = model(**inp, output_hidden_states=True)
                hook.remove()

                hs_p = out_p.hidden_states[-1]
                with torch.no_grad():
                    normed_p = model.model.norm(hs_p[:, -1:, :])
                    logits_p = model.lm_head(normed_p).squeeze().float()
                probs_p = torch.softmax(logits_p, dim=-1)
                S_p = -(probs_p * torch.log(probs_p + 1e-10)).sum().item()
                S_p = S_p if not np.isnan(S_p) else 0
                responses.append(abs(S_p - all_S[pi, -1]))

            all_chi[pi, li] = np.mean(responses)

    # Fluctuation: variance of S across prompts at each layer
    var_S = np.var(all_S, axis=0)
    mean_chi = np.mean(all_chi, axis=0)

    # FDR: chi / (beta * C_v) ~ chi / var_S
    # Simplified: FDR ratio = chi / var_S (normalized)
    fdr = np.zeros(n_layers)
    for li in range(n_layers):
        if var_S[li] > 1e-6 and li < len(mean_chi):
            fdr[li] = mean_chi[li] / (var_S[li] + 1e-10)
        else:
            fdr[li] = 0

    layers = np.arange(n_layers)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Fluctuation (var S) vs layer
    axes[0,0].plot(layers, var_S, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$C_v$ = Var($S$)')
    axes[0,0].set_title('(a) Fluctuation (Heat Capacity)')
    axes[0,0].legend()

    # (b) Response (chi) vs layer
    axes[0,1].plot(layers[:len(mean_chi)], mean_chi, 'o-', color='#c0392b',
                  markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer of Perturbation')
    axes[0,1].set_ylabel('$\\chi$ = Response')
    axes[0,1].set_title('(b) Dissipation (Response)')

    # (c) FDR ratio
    valid_fdr = fdr[:len(mean_chi)]
    fdr_colors = ['#27ae60' if abs(f - 1) < 0.5 else '#c0392b' for f in valid_fdr]
    axes[0,2].bar(layers[:len(valid_fdr)], valid_fdr, color=fdr_colors,
                  alpha=0.7, edgecolor='black')
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=1, color='black', linewidth=2, linestyle='-', label='Equilibrium')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('FDR = $\\chi / C_v$')
    axes[0,2].set_title('(c) FD Ratio')
    axes[0,2].legend()

    # (d) Pre vs Post FDR
    pre_fdr = np.mean(valid_fdr[:20])
    post_fdr = np.mean(valid_fdr[20:])
    axes[1,0].bar(['Pre-$L_0$', 'Post-$L_0$'], [pre_fdr, post_fdr],
                  color=['#2980b9', '#c0392b'], alpha=0.8, edgecolor='black')
    axes[1,0].axhline(y=1, color='black', linewidth=1, linestyle='--')
    axes[1,0].set_ylabel('Mean FDR')
    axes[1,0].set_title('(d) FDR by Phase')

    # (e) Chi vs C_v scatter
    for li in range(min(len(mean_chi), n_layers)):
        color = '#2980b9' if li < 20 else '#c0392b'
        axes[1,1].scatter(var_S[li], mean_chi[li], c=color, s=50,
                         edgecolors='black', alpha=0.7)
    axes[1,1].set_xlabel('$C_v$ (Fluctuation)')
    axes[1,1].set_ylabel('$\\chi$ (Response)')
    axes[1,1].set_title('(e) Fluctuation vs Response')
    # FDR=1 line
    x_range = np.linspace(0, max(var_S) * 1.1, 100)
    axes[1,1].plot(x_range, x_range, '--', color='gray', label='FDR=1')
    axes[1,1].legend()

    # (f) Summary
    max_fdr_layer = np.argmax(valid_fdr[4:]) + 4
    summary = (
        f"Fluctuation-Dissipation Relation\n\n"
        f"Pre-L0 FDR: {pre_fdr:.3f}\n"
        f"Post-L0 FDR: {post_fdr:.3f}\n\n"
        f"Max FDR layer: L{max_fdr_layer}\n"
        f"  (FDR={valid_fdr[max_fdr_layer]:.3f})\n\n"
        f"Equilibrium (FDR=1):\n"
        f"{'VIOLATED' if abs(pre_fdr - 1) > 0.3 else 'HOLDS'} pre-L0\n"
        f"{'VIOLATED' if abs(post_fdr - 1) > 0.3 else 'HOLDS'} post-L0"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 172: Fluctuation-Dissipation Relation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase172_fdr')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-L0 FDR: {pre_fdr:.3f}")
    print(f"Post-L0 FDR: {post_fdr:.3f}")
    print(f"Max FDR: L{max_fdr_layer} ({valid_fdr[max_fdr_layer]:.3f})")
    print(f"{'='*70}")

    save_results('phase172_fdr', {
        'experiment': 'FDR at Each Layer',
        'summary': {
            'pre_fdr': float(pre_fdr),
            'post_fdr': float(post_fdr),
            'max_fdr_layer': int(max_fdr_layer),
        }
    })


if __name__ == '__main__':
    main()
