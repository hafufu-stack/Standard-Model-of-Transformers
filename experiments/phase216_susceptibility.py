# -*- coding: utf-8 -*-
"""
Phase 216: Susceptibility Divergence
=======================================
At a true critical point, specific heat C_v and susceptibility chi
diverge (show a peak). Test this for L0.

C_v(l) = |dU/dT| at each layer
chi(l) = d(OP)/d(sigma) at each layer (response to noise)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

MODEL_SIZES = ['0.5B', '1.5B']
NOISE_SIGMAS = [0.01, 0.05, 0.10]  # Multiple field strengths for chi


def compute_specific_heat(model, tok, device):
    """Compute C_v(l) = |dU/dT| at each layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_U, all_T = [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        U_list, T_list = [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)
        all_U.append(U_list)
        all_T.append(T_list)

    n_hs = min(len(u) for u in all_U)
    mean_U = [float(np.mean([u[i] for u in all_U])) for i in range(n_hs)]
    mean_T = [float(np.mean([t[i] for t in all_T])) for i in range(n_hs)]

    # C_v = |dU/dT| via dU/dl / dT/dl
    Cv = []
    for i in range(n_hs - 1):
        dU = mean_U[i+1] - mean_U[i]
        dT = mean_T[i+1] - mean_T[i]
        if abs(dT) > 1e-6:
            cv = abs(dU / dT)
        else:
            cv = 0
        Cv.append(cv)

    return mean_U, mean_T, Cv


def compute_susceptibility(model, tok, device, sigma):
    """Compute chi(l) = d(P1)/d(sigma) at each layer."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    chi_per_layer = []
    for inject_layer in range(n_layers):
        delta_P1_values = []
        for prompt in PROMPTS:
            inp = tok(prompt, return_tensors='pt').to(device)
            input_ids = inp['input_ids']

            # Baseline P1
            with torch.no_grad():
                out = model(**inp)
            baseline_P1 = torch.softmax(out.logits[0, -1, :].float(), dim=-1).max().item()

            # Perturbed P1
            with torch.no_grad():
                hidden = model.model.embed_tokens(input_ids)
                seq_len = hidden.shape[1]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(hidden, position_ids)

                for li in range(n_layers):
                    layer = model.model.layers[li]
                    layer_out = layer(hidden, position_embeddings=position_embeddings)
                    hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]
                    if li == inject_layer:
                        noise = torch.randn_like(hidden.float()) * sigma
                        hidden = (hidden.float() + noise).to(hidden.dtype)

                normed = norm_layer(hidden)
                logits = lm_head(normed)[0, -1, :].float()
            perturbed_P1 = torch.softmax(logits, dim=-1).max().item()

            delta_P1 = abs(perturbed_P1 - baseline_P1)
            delta_P1_values.append(delta_P1)

        chi = float(np.mean(delta_P1_values)) / (sigma + 1e-10)
        chi_per_layer.append(chi)

    return chi_per_layer


def main():
    print("=" * 70)
    print("Phase 216: Susceptibility Divergence")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_model = {}

    for size in MODEL_SIZES:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        n_layers = len(model.model.layers)

        # Specific heat
        print("  Computing specific heat C_v...")
        mean_U, mean_T, Cv = compute_specific_heat(model, tok, device)

        # Susceptibility at multiple field strengths
        chi_results = {}
        for sigma in NOISE_SIGMAS:
            print(f"  Computing susceptibility (sigma={sigma})...")
            chi = compute_susceptibility(model, tok, device, sigma)
            chi_results[str(sigma)] = chi

        # Find peaks
        Cv_peak = int(np.argmax(Cv)) if Cv else 0
        chi_peak = int(np.argmax(chi_results[str(NOISE_SIGMAS[1])])) if chi_results else 0

        print(f"  C_v peak: layer {Cv_peak}")
        print(f"  chi peak: layer {chi_peak}")

        results_by_model[size] = {
            'n_layers': n_layers,
            'mean_U': mean_U,
            'mean_T': mean_T,
            'Cv': Cv,
            'Cv_peak': Cv_peak,
            'chi': chi_results,
            'chi_peak': chi_peak,
        }

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) C_v profiles
    for size in MODEL_SIZES:
        r = results_by_model[size]
        x_norm = np.linspace(0, 1, len(r['Cv']))
        axes[0, 0].plot(x_norm, r['Cv'], '-', color=colors[size], lw=2,
                        label=f'{size} (peak@{r["Cv_peak"]})')
    axes[0, 0].set_xlabel('Normalized Layer')
    axes[0, 0].set_ylabel('Specific Heat C_v')
    axes[0, 0].set_title('(a) Specific Heat |dU/dT|')
    axes[0, 0].legend(fontsize=8)

    # (b) chi profiles (sigma=0.05)
    for size in MODEL_SIZES:
        r = results_by_model[size]
        chi = r['chi'][str(NOISE_SIGMAS[1])]
        x_norm = np.linspace(0, 1, len(chi))
        axes[0, 1].plot(x_norm, chi, '-', color=colors[size], lw=2,
                        label=f'{size} (peak@{r["chi_peak"]})')
    axes[0, 1].set_xlabel('Normalized Layer')
    axes[0, 1].set_ylabel('Susceptibility chi')
    axes[0, 1].set_title('(b) Susceptibility d(P1)/d(sigma)')
    axes[0, 1].legend(fontsize=8)

    # (c) chi at multiple field strengths (1.5B only)
    r15 = results_by_model['1.5B']
    for sigma in NOISE_SIGMAS:
        chi = r15['chi'][str(sigma)]
        axes[0, 2].plot(range(len(chi)), chi, '-', lw=2,
                        label=f'sigma={sigma}', alpha=0.8)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Susceptibility chi')
    axes[0, 2].set_title('(c) 1.5B: Field Strength Dependence')
    axes[0, 2].legend(fontsize=8)

    # (d) Peak positions comparison
    Cv_peaks = [results_by_model[s]['Cv_peak'] / results_by_model[s]['n_layers']
                for s in MODEL_SIZES]
    chi_peaks = [results_by_model[s]['chi_peak'] / results_by_model[s]['n_layers']
                 for s in MODEL_SIZES]
    x = np.arange(len(MODEL_SIZES))
    w = 0.35
    axes[1, 0].bar(x - w/2, Cv_peaks, w, label='C_v peak', color='#f39c12', alpha=0.7)
    axes[1, 0].bar(x + w/2, chi_peaks, w, label='chi peak', color='#9b59b6', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(MODEL_SIZES)
    axes[1, 0].set_ylabel('Peak Position (l/N)')
    axes[1, 0].set_title('(d) Peak Positions: Universal?')
    axes[1, 0].legend(fontsize=8)
    for i in range(len(MODEL_SIZES)):
        axes[1, 0].text(i - w/2, Cv_peaks[i] + 0.02, f'{Cv_peaks[i]:.2f}',
                        ha='center', fontsize=8)
        axes[1, 0].text(i + w/2, chi_peaks[i] + 0.02, f'{chi_peaks[i]:.2f}',
                        ha='center', fontsize=8)

    # (e) C_v and chi overlay for 1.5B
    ax1 = axes[1, 1]
    ax2t = ax1.twinx()
    l1 = ax1.plot(range(len(r15['Cv'])), r15['Cv'], '-', color='#f39c12',
                  lw=2, label='C_v')
    chi_05 = r15['chi'][str(NOISE_SIGMAS[1])]
    l2 = ax2t.plot(range(len(chi_05)), chi_05, '-', color='#9b59b6',
                   lw=2, label='chi')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('C_v', color='#f39c12')
    ax2t.set_ylabel('chi', color='#9b59b6')
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    axes[1, 1].set_title('(e) 1.5B: C_v and chi Overlay')

    # (f) Summary
    cv_diff = abs(Cv_peaks[0] - Cv_peaks[1])
    chi_diff = abs(chi_peaks[0] - chi_peaks[1])
    summary = (
        f"Susceptibility Divergence\n\n"
        f"C_v peaks:\n"
        f"  0.5B: {results_by_model['0.5B']['Cv_peak']} "
        f"({Cv_peaks[0]:.2f})\n"
        f"  1.5B: {results_by_model['1.5B']['Cv_peak']} "
        f"({Cv_peaks[1]:.2f})\n"
        f"  diff: {cv_diff:.3f}\n\n"
        f"chi peaks:\n"
        f"  0.5B: {results_by_model['0.5B']['chi_peak']} "
        f"({chi_peaks[0]:.2f})\n"
        f"  1.5B: {results_by_model['1.5B']['chi_peak']} "
        f"({chi_peaks[1]:.2f})\n"
        f"  diff: {chi_diff:.3f}\n\n"
        f"Peaks coincide: "
        f"{'YES' if cv_diff < 0.15 and chi_diff < 0.15 else 'NO'}\n"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 216: Susceptibility Divergence",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase216_susceptibility')
    plt.close()

    save_results('phase216_susceptibility', {
        'experiment': 'Susceptibility Divergence',
        'results': {s: {k: v for k, v in r.items()}
                    for s, r in results_by_model.items()},
    })


if __name__ == '__main__':
    main()
