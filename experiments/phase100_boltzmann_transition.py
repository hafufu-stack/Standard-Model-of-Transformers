# -*- coding: utf-8 -*-
"""
Phase 100: Boltzmann Distribution Before and After Eta Transition
Phase 97 found eta phase transition at L0=21.7.
Phase 91 found Boltzmann universality (R2>0.8 at all scales).
Question: Does the Boltzmann R2 change qualitatively at L0?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "The human genome encodes three billion base pairs",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Climate change affects global ecosystems",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
]


def boltzmann(E, A, kT):
    return A * np.exp(-E / (kT + 1e-10))


def measure_boltzmann_at_layer(model, tok, device, layer_idx):
    """Measure Boltzmann fit R2 at a specific layer."""
    all_norms = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hs = out.hidden_states[layer_idx]
        h = hs[0, -1, :].float()

        # Measure activation distribution (neuron-level)
        norms = h.abs().cpu().numpy()
        all_norms.extend(norms.tolist())

    all_norms = np.array(all_norms)
    all_norms = all_norms[all_norms > 0]

    if len(all_norms) < 50:
        return 0, 0

    hist, edges = np.histogram(all_norms, bins=30, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    hv = hist[mask]
    bc = centers[mask]

    if len(bc) < 5:
        return 0, 0

    try:
        popt, _ = curve_fit(boltzmann, bc, hv, p0=[hv[0], np.mean(all_norms)],
                            maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
        pred = boltzmann(bc, *popt)
        ss_res = np.sum((hv - pred)**2)
        ss_tot = np.sum((hv - np.mean(hv))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return float(r2), float(popt[1])
    except Exception:
        return 0, 0


def main():
    print("=" * 70)
    print("Phase 100: Boltzmann Distribution vs Eta Transition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    results = []
    L0 = 21.7  # From Phase 97

    for li in range(n_layers):
        r2, kT = measure_boltzmann_at_layer(model, tok, device, li)
        phase = 'pre-transition' if li < L0 else 'post-transition'
        results.append({
            'layer': li,
            'boltzmann_r2': float(r2),
            'kT': float(kT),
            'phase': phase,
        })
        print(f"  L{li:2d}: R2={r2:.4f}, kT={kT:.4f} [{phase}]")

    pre = [r for r in results if r['phase'] == 'pre-transition']
    post = [r for r in results if r['phase'] == 'post-transition']

    mean_r2_pre = np.mean([r['boltzmann_r2'] for r in pre])
    mean_r2_post = np.mean([r['boltzmann_r2'] for r in post])
    mean_kT_pre = np.mean([r['kT'] for r in pre])
    mean_kT_post = np.mean([r['kT'] for r in post])

    # Also measure entropy at each layer
    entropies = []
    for li in range(n_layers):
        ents = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(S):
                ents.append(S)
        entropies.append(float(np.mean(ents)) if ents else 0)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers = [r['layer'] for r in results]
    r2s = [r['boltzmann_r2'] for r in results]
    kTs = [r['kT'] for r in results]

    # (a) R2 profile
    colors_bar = ['#2980b9' if l < L0 else '#c0392b' for l in layers]
    axes[0,0].bar(layers, r2s, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0={L0:.0f}$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Boltzmann $R^2$')
    axes[0,0].set_title(f'(a) Boltzmann Fit (pre={mean_r2_pre:.3f}, post={mean_r2_post:.3f})')
    axes[0,0].legend()

    # (b) kT profile
    axes[0,1].plot(layers, kTs, 'o-', color='#8e44ad', markersize=4)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Effective $kT$')
    axes[0,1].set_title('(b) Effective Temperature')

    # (c) Entropy profile
    axes[0,2].plot(layers, entropies, 'o-', color='#27ae60', markersize=4)
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Entropy $S$')
    axes[0,2].set_title('(c) Entropy Profile')

    # (d) R2 vs Entropy scatter
    axes[1,0].scatter(entropies, r2s, c=layers, cmap='viridis', s=60, edgecolors='black')
    cb = plt.colorbar(axes[1,0].collections[0], ax=axes[1,0], shrink=0.7)
    cb.set_label('Layer')
    axes[1,0].set_xlabel('Entropy $S$')
    axes[1,0].set_ylabel('Boltzmann $R^2$')
    axes[1,0].set_title('(d) $R^2$ vs Entropy')

    # (e) Pre vs Post comparison
    categories = ['Pre-transition\n(L < L0)', 'Post-transition\n(L >= L0)']
    r2_vals = [mean_r2_pre, mean_r2_post]
    kT_vals_comp = [mean_kT_pre, mean_kT_post]
    x = np.arange(2)
    axes[1,1].bar(x - 0.2, r2_vals, 0.35, color='#3498db', alpha=0.8, label='$R^2$')
    ax_twin = axes[1,1].twinx()
    ax_twin.bar(x + 0.2, kT_vals_comp, 0.35, color='#e74c3c', alpha=0.8, label='$kT$')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(categories)
    axes[1,1].set_ylabel('$R^2$', color='#3498db')
    ax_twin.set_ylabel('$kT$', color='#e74c3c')
    axes[1,1].set_title('(e) Phase Comparison')

    # (f) Summary
    r2_change = mean_r2_post - mean_r2_pre
    kT_change = mean_kT_post / (mean_kT_pre + 1e-10)
    summary = (
        f"Boltzmann vs Eta Transition\n\n"
        f"L0 = {L0:.0f}\n\n"
        f"Pre-transition (L < {L0:.0f}):\n"
        f"  R2 = {mean_r2_pre:.3f}\n"
        f"  kT = {mean_kT_pre:.3f}\n\n"
        f"Post-transition (L >= {L0:.0f}):\n"
        f"  R2 = {mean_r2_post:.3f}\n"
        f"  kT = {mean_kT_post:.3f}\n\n"
        f"R2 change: {r2_change:+.3f}\n"
        f"kT ratio: {kT_change:.2f}x"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle(f'Phase 100: Boltzmann Distribution vs Eta Transition '
                 f'($\\Delta R^2={r2_change:+.3f}$)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase100_boltzmann_transition')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Pre-transition:  R2={mean_r2_pre:.3f}, kT={mean_kT_pre:.3f}")
    print(f"Post-transition: R2={mean_r2_post:.3f}, kT={mean_kT_post:.3f}")
    print(f"R2 change: {r2_change:+.3f}")
    print(f"kT cooling ratio: {kT_change:.2f}x")
    print(f"{'='*70}")

    save_results('phase100_boltzmann_transition', {
        'experiment': 'Boltzmann vs Eta Transition',
        'results': results,
        'entropies': entropies,
        'summary': {
            'L0': float(L0),
            'r2_pre': float(mean_r2_pre),
            'r2_post': float(mean_r2_post),
            'r2_change': float(r2_change),
            'kT_pre': float(mean_kT_pre),
            'kT_post': float(mean_kT_post),
            'kT_ratio': float(kT_change),
        }
    })


if __name__ == '__main__':
    main()
