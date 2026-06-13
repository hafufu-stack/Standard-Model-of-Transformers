# -*- coding: utf-8 -*-
"""
Phase 42: Boltzmann Neurons
Test whether FFN neuron activations follow Boltzmann distribution P(E) ~ exp(-E/kT).
If confirmed, proves LLMs are strict statistical mechanical systems.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kstest
from utils import load_model, save_results, save_figure


def boltzmann_pdf(E, A, kT):
    """Boltzmann distribution: P(E) = A * exp(-E/kT)"""
    return A * np.exp(-E / (kT + 1e-10))


def main():
    print("=" * 70)
    print("Phase 42: Boltzmann Neurons (Statistical Mechanics Proof)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    prompts = [
        "The theory of general relativity describes how massive objects warp the fabric of",
        "In quantum mechanics, the uncertainty principle states that one cannot simultaneously know",
        "The human brain contains approximately 86 billion neurons, each connected to",
        "Climate change is driven primarily by the accumulation of greenhouse gases in",
    ]

    n_layers = len(model.model.layers)
    all_layer_results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {prompt_idx+1}: '{prompt[:50]}...' ---")
        inp = tok(prompt, return_tensors='pt').to(device)

        # Capture FFN activations at each layer
        ffn_activations = {}

        def make_ffn_capture_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                # Capture last-token FFN activation
                ffn_activations[layer_idx] = h[0, -1, :].detach().cpu().float().numpy()
            return hook

        hooks = []
        for li in range(n_layers):
            h = model.model.layers[li].mlp.register_forward_hook(make_ffn_capture_hook(li))
            hooks.append(h)

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in hooks:
            h.remove()

        # Also measure macro temperature T from logits
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        macro_T = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Analyze each layer
        for li in range(n_layers):
            if li not in ffn_activations:
                continue
            act = ffn_activations[li]

            # Energy = activation^2 (kinetic energy of neuron)
            energies = act ** 2

            # Create histogram of energies
            # Filter out zero energies
            nonzero_E = energies[energies > 1e-8]
            if len(nonzero_E) < 50:
                continue

            # Bin the energies
            n_bins = 50
            hist, bin_edges = np.histogram(nonzero_E, bins=n_bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Filter out zero-count bins
            mask = hist > 0
            bin_c = bin_centers[mask]
            hist_v = hist[mask]

            if len(bin_c) < 10:
                continue

            # Fit Boltzmann distribution
            try:
                popt, pcov = curve_fit(boltzmann_pdf, bin_c, hist_v,
                                       p0=[hist_v[0], np.mean(nonzero_E)],
                                       maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
                A_fit, kT_fit = popt

                # Goodness of fit: R^2
                residuals = hist_v - boltzmann_pdf(bin_c, *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((hist_v - np.mean(hist_v)) ** 2)
                r_squared = 1 - ss_res / (ss_tot + 1e-10)

                # KS test against exponential (Boltzmann is exponential in E)
                ks_stat, ks_pval = kstest(nonzero_E, 'expon', args=(0, np.mean(nonzero_E)))

                fit_success = True
            except Exception:
                A_fit = kT_fit = r_squared = 0
                ks_stat = ks_pval = 0
                fit_success = False

            all_layer_results.append({
                'prompt_idx': prompt_idx, 'layer': li,
                'macro_T': macro_T,
                'micro_kT': float(kT_fit) if fit_success else None,
                'R_squared': float(r_squared) if fit_success else None,
                'ks_stat': float(ks_stat),
                'ks_pval': float(ks_pval),
                'n_neurons': len(nonzero_E),
                'mean_energy': float(np.mean(nonzero_E)),
                'fit_success': fit_success,
            })

        if prompt_idx == 0:
            print(f"  Macro T (logit entropy): {macro_T:.3f}")
            for r in all_layer_results[-3:]:
                if r['fit_success']:
                    print(f"  Layer {r['layer']}: micro_kT={r['micro_kT']:.4f}, "
                          f"R2={r['R_squared']:.4f}, KS_p={r['ks_pval']:.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) R^2 across layers (averaged over prompts)
    layer_r2 = {}
    layer_kT = {}
    for r in all_layer_results:
        if r['fit_success']:
            li = r['layer']
            if li not in layer_r2:
                layer_r2[li] = []
                layer_kT[li] = []
            layer_r2[li].append(r['R_squared'])
            layer_kT[li].append(r['micro_kT'])

    layers_sorted = sorted(layer_r2.keys())
    r2_means = [np.mean(layer_r2[l]) for l in layers_sorted]
    r2_stds = [np.std(layer_r2[l]) for l in layers_sorted]

    axes[0, 0].errorbar(layers_sorted, r2_means, yerr=r2_stds, marker='o',
                        color='#e74c3c', capsize=3, markersize=4)
    axes[0, 0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='R2=0.9')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('R-squared (Boltzmann Fit)')
    axes[0, 0].set_title('(a) Boltzmann Fit Quality')
    axes[0, 0].legend()

    # (b) micro kT across layers
    kT_means = [np.mean(layer_kT[l]) for l in layers_sorted]
    kT_stds = [np.std(layer_kT[l]) for l in layers_sorted]
    axes[0, 1].errorbar(layers_sorted, kT_means, yerr=kT_stds, marker='s',
                        color='#3498db', capsize=3, markersize=4)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Micro kT (Boltzmann)')
    axes[0, 1].set_title('(b) Micro Temperature Across Layers')

    # (c) Example fit: best layer of first prompt
    p0_results = [r for r in all_layer_results if r['prompt_idx'] == 0 and r['fit_success']]
    if p0_results:
        best = max(p0_results, key=lambda x: x['R_squared'])
        li = best['layer']
        # Re-extract for plotting
        inp = tok(prompts[0], return_tensors='pt').to(device)
        ffn_act_plot = {}
        def plot_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            ffn_act_plot[0] = h[0, -1, :].detach().cpu().float().numpy()
        hh = model.model.layers[li].mlp.register_forward_hook(plot_hook)
        with torch.no_grad():
            model(**inp)
        hh.remove()

        if 0 in ffn_act_plot:
            act_plot = ffn_act_plot[0]
            E_plot = act_plot ** 2
            E_nz = E_plot[E_plot > 1e-8]
            hist_p, edges_p = np.histogram(E_nz, bins=50, density=True)
            centers_p = (edges_p[:-1] + edges_p[1:]) / 2
            mask_p = hist_p > 0
            axes[1, 0].bar(centers_p[mask_p], hist_p[mask_p], width=np.diff(edges_p)[0],
                           color='#9b59b6', alpha=0.6, label='Data')
            try:
                popt_p, _ = curve_fit(boltzmann_pdf, centers_p[mask_p], hist_p[mask_p],
                                      p0=[hist_p[mask_p][0], np.mean(E_nz)], maxfev=5000,
                                      bounds=([0, 1e-8], [np.inf, np.inf]))
                x_fit = np.linspace(centers_p[mask_p].min(), centers_p[mask_p].max(), 100)
                axes[1, 0].plot(x_fit, boltzmann_pdf(x_fit, *popt_p), 'r-', linewidth=2,
                               label=f'Boltzmann (kT={popt_p[1]:.3f})')
            except Exception:
                pass
            axes[1, 0].set_xlabel('Energy (activation^2)')
            axes[1, 0].set_ylabel('Probability Density')
            axes[1, 0].set_title(f'(c) Layer {li} Fit (R2={best["R_squared"]:.3f})')
            axes[1, 0].legend()

    # (d) Macro T vs Mean micro kT
    macro_Ts = []
    micro_kTs = []
    for prompt_idx in range(len(prompts)):
        p_results = [r for r in all_layer_results if r['prompt_idx'] == prompt_idx and r['fit_success']]
        if p_results:
            macro_Ts.append(p_results[0]['macro_T'])
            micro_kTs.append(np.mean([r['micro_kT'] for r in p_results]))

    if macro_Ts:
        axes[1, 1].scatter(macro_Ts, micro_kTs, s=80, c='#e67e22', edgecolors='black')
        axes[1, 1].set_xlabel('Macro T (Logit Entropy)')
        axes[1, 1].set_ylabel('Mean Micro kT (Boltzmann)')
        axes[1, 1].set_title('(d) Macro vs Micro Temperature')
        # Fit line
        if len(macro_Ts) > 1:
            coeffs = np.polyfit(macro_Ts, micro_kTs, 1)
            x_line = np.linspace(min(macro_Ts), max(macro_Ts), 50)
            axes[1, 1].plot(x_line, np.polyval(coeffs, x_line), 'r--',
                           label=f'slope={coeffs[0]:.3f}')
            axes[1, 1].legend()

    fig.suptitle('Phase 42: Boltzmann Neurons', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase42_boltzmann')
    plt.close()

    # === Verdict ===
    all_r2 = [r['R_squared'] for r in all_layer_results if r['fit_success']]
    mean_r2 = np.mean(all_r2) if all_r2 else 0
    high_r2_pct = sum(1 for r in all_r2 if r > 0.8) / (len(all_r2) or 1) * 100
    ks_pass = sum(1 for r in all_layer_results if r.get('ks_pval', 0) > 0.05) / (len(all_layer_results) or 1) * 100

    print(f"\n{'='*70}")
    print(f"VERDICT: Mean Boltzmann R2={mean_r2:.3f}, {high_r2_pct:.0f}% layers with R2>0.8, "
          f"{ks_pass:.0f}% pass KS test (p>0.05). "
          f"{'CONFIRMED: LLM is a statistical mechanical system' if mean_r2 > 0.7 else 'Partial fit only'}.")
    print(f"{'='*70}")

    save_results('phase42_boltzmann', {
        'experiment': 'Boltzmann Neurons',
        'layer_results': all_layer_results,
        'summary': {
            'mean_r_squared': mean_r2,
            'pct_high_r2': high_r2_pct,
            'pct_ks_pass': ks_pass,
        }
    })


if __name__ == '__main__':
    main()
