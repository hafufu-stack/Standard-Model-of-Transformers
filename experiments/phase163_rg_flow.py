# -*- coding: utf-8 -*-
"""
Phase 163: Renormalization Group Flow
Coarse-grain layers by pairs and triplets, re-measure eta.
If the phase transition survives coarse-graining, this proves
it is a TRUE critical phenomenon (scale-invariant).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


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
    print("Phase 163: Renormalization Group Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect raw S profiles
    all_S = np.zeros((len(PROMPTS), n_layers))
    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
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

    mean_S = np.mean(all_S, axis=0)

    # Compute eta from mean_S at different coarse-graining scales
    def compute_eta_from_S(S_vals):
        """Compute running eta from a sequence of S values."""
        n = len(S_vals)
        eta = []
        for i in range(n):
            subset = S_vals[:i+1]
            if len(subset) >= 3:
                T_hot = max(subset)
                T_cold = min(subset[len(subset)//2:])
                e = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                e = 0
            eta.append(e)
        return np.array(eta)

    # Scale 1: every layer (original)
    eta_1 = compute_eta_from_S(mean_S)

    # Scale 2: every 2nd layer
    S_2 = mean_S[::2]
    eta_2 = compute_eta_from_S(S_2)

    # Scale 3: every 3rd layer
    S_3 = mean_S[::3]
    eta_3 = compute_eta_from_S(S_3)

    # Scale 4: every 4th layer
    S_4 = mean_S[::4]
    eta_4 = compute_eta_from_S(S_4)

    # Fit sigmoid to each and extract L0
    results = {}
    for scale, eta, S_vals in [(1, eta_1, mean_S), (2, eta_2, S_2),
                                (3, eta_3, S_3), (4, eta_4, S_4)]:
        n = len(eta)
        try:
            skip = max(2, n // 6)
            Ls = np.arange(skip, n)
            popt, _ = curve_fit(sigmoid, Ls, eta[skip:],
                                p0=[n * 0.75, 0.5, 0, 0.9], maxfev=10000)
            L0 = popt[0]
            r2 = 1 - np.sum((eta[skip:] - sigmoid(Ls, *popt))**2) / (
                np.sum((eta[skip:] - np.mean(eta[skip:]))**2) + 1e-10)
        except:
            L0 = n * 0.75
            r2 = 0

        L0_real = L0 * scale  # Back to original layer index
        results[scale] = {
            'n_points': n,
            'L0': float(L0),
            'L0_real': float(L0_real),
            'L0_ratio': float(L0 / n),
            'R2': float(r2),
            'eta': eta.tolist(),
            'final_eta': float(eta[-1]),
        }
        print(f"  Scale {scale}: {n} points, L0={L0:.1f} (real={L0_real:.1f}), "
              f"L0/n={L0/n:.3f}, R2={r2:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_s = {1: '#2980b9', 2: '#c0392b', 3: '#27ae60', 4: '#f39c12'}

    # (a) Eta at each scale (normalized x-axis)
    for scale, r in results.items():
        x = np.linspace(0, 1, len(r['eta']))
        axes[0,0].plot(x, r['eta'], 'o-', color=colors_s[scale], markersize=4,
                      linewidth=2, label=f'Scale {scale} (L0/n={r["L0_ratio"]:.3f})')
    axes[0,0].set_xlabel('Relative Depth')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) RG Flow: Normalized Eta')
    axes[0,0].legend(fontsize=7)

    # (b) L0/n ratio at each scale
    scales = list(results.keys())
    ratios = [results[s]['L0_ratio'] for s in scales]
    axes[0,1].bar(range(len(scales)), ratios,
                  color=[colors_s[s] for s in scales], alpha=0.8, edgecolor='black')
    axes[0,1].set_xticks(range(len(scales)))
    axes[0,1].set_xticklabels([f'Scale {s}' for s in scales])
    mean_ratio = np.mean([r for r in ratios if 0 < r < 2])
    axes[0,1].axhline(y=mean_ratio, color='black', linestyle='--',
                      label=f'Mean={mean_ratio:.3f}')
    axes[0,1].set_ylabel('$L_0 / n$')
    axes[0,1].set_title('(b) Scale Invariance of L0')
    axes[0,1].legend()

    # (c) R2 at each scale
    r2s = [results[s]['R2'] for s in scales]
    axes[0,2].bar(range(len(scales)), r2s,
                  color=[colors_s[s] for s in scales], alpha=0.8, edgecolor='black')
    axes[0,2].set_xticks(range(len(scales)))
    axes[0,2].set_xticklabels([f'Scale {s}' for s in scales])
    axes[0,2].set_ylabel('$R^2$')
    axes[0,2].set_title('(c) Fit Quality vs Scale')

    # (d) L0_real (in original layer coords)
    L0_reals = [results[s]['L0_real'] for s in scales]
    axes[1,0].bar(range(len(scales)), L0_reals,
                  color=[colors_s[s] for s in scales], alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(scales)))
    axes[1,0].set_xticklabels([f'Scale {s}' for s in scales])
    axes[1,0].axhline(y=21.7, color='gray', linestyle='--', label='L0=21.7')
    axes[1,0].set_ylabel('$L_0$ (original)')
    axes[1,0].set_title('(d) L0 in Original Coordinates')
    axes[1,0].legend()

    # (e) S profiles at different scales
    for scale, r in results.items():
        S_vals = mean_S[::scale]
        x = np.arange(len(S_vals)) * scale
        axes[1,1].plot(x, S_vals, 'o-', color=colors_s[scale], markersize=4,
                      linewidth=2, label=f'Scale {scale}')
    axes[1,1].axvline(x=21.7, color='gray', linewidth=1, linestyle='--')
    axes[1,1].set_xlabel('Original Layer')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) Coarse-Grained Entropy')
    axes[1,1].legend(fontsize=8)

    # (f) Summary
    ratio_cv = np.std([r for r in ratios if 0 < r < 2]) / (mean_ratio + 1e-10)
    summary = (
        f"Renormalization Group Flow\n\n"
        + "\n".join(f"Scale {s}: L0/n={results[s]['L0_ratio']:.3f} "
                    f"(R2={results[s]['R2']:.3f})"
                    for s in scales)
        + f"\n\nMean L0/n: {mean_ratio:.3f}\n"
        f"CV: {ratio_cv:.3f}\n\n"
        f"Phase transition is\n"
        f"{'SCALE-INVARIANT' if ratio_cv < 0.15 else 'SCALE-DEPENDENT'}\n"
        f"(RG flow {'preserves' if ratio_cv < 0.15 else 'breaks'} L0/n)"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 163: Renormalization Group Flow',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase163_rg_flow')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Mean L0/n={mean_ratio:.3f}, CV={ratio_cv:.3f}")
    print(f"Scale invariant: {'YES' if ratio_cv < 0.15 else 'NO'}")
    print(f"{'='*70}")

    save_results('phase163_rg_flow', {
        'experiment': 'RG Flow',
        'results': {str(s): {k: v for k, v in r.items() if k != 'eta'}
                    for s, r in results.items()},
        'summary': {
            'mean_ratio': float(mean_ratio),
            'ratio_cv': float(ratio_cv),
        }
    })


if __name__ == '__main__':
    main()
