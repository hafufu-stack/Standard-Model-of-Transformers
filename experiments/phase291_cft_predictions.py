# -*- coding: utf-8 -*-
"""
Phase 291: CFT Predictions from c=1
======================================
Phase 279 found central charge c ~ 1.0 (0.5B) and c ~ 0.89 (1.5B).
c=1 corresponds to a free boson CFT. This makes concrete predictions:

1. Entanglement entropy: S = (c/3) * log(l) = (1/3) * log(l)
2. Correlation functions: <O(l1) O(l2)> ~ |l1-l2|^(-2*Delta)
3. Operator spectrum: Delta_n = n (free boson)
4. Partition function: Z ~ (1/eta(q))^c

Test all predictions against real transformer data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPT = "The fundamental laws of physics describe how the universe works from the smallest particles to the largest structures"


def main():
    print("=" * 70)
    print("Phase 291: CFT Predictions from c=1")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        n_layers = len(model.model.layers)
        D = model.config.hidden_size

        inp = tok(PROMPT, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # ===== TEST 1: Entanglement entropy S = (c/3) * log(l) =====
        # Compute SVD entropy of layer representations
        svd_entropies = []
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0].float().cpu().numpy()  # (seq, D)
            _, s, _ = np.linalg.svd(h, full_matrices=False)
            s_norm = s / (s.sum() + 1e-10)
            valid = s_norm > 1e-15
            entropy = -np.sum(s_norm[valid] * np.log(s_norm[valid]))
            svd_entropies.append(float(entropy))

        # Fit: S(l) = (c_eff/3) * log(l) + const
        layers_arr = np.arange(1, n_layers + 1)
        log_layers = np.log(layers_arr)
        ent_to_fit = svd_entropies[1:]  # skip embedding
        slope, intercept, r_ent, p_ent, _ = stats.linregress(log_layers, ent_to_fit)
        c_from_entropy = 3 * slope  # c = 3 * slope

        print(f"  Prediction 1: S = (c/3)*log(l)")
        print(f"    c_eff = {c_from_entropy:.4f} (predict: 1.0)")
        print(f"    R2 = {r_ent**2:.4f}")

        # ===== TEST 2: Correlation function decay =====
        # Inter-layer correlations: C(dl) = <h(l) . h(l+dl)> / (|h(l)||h(l+dl)|)
        corr_data = {}
        for dl in range(1, min(8, n_layers)):
            corrs = []
            for li in range(n_layers - dl):
                h1 = out.hidden_states[li + 1][0, -1, :].float()
                h2 = out.hidden_states[li + 1 + dl][0, -1, :].float()
                cos = torch.nn.functional.cosine_similarity(
                    h1.unsqueeze(0), h2.unsqueeze(0)).item()
                corrs.append(cos)
            corr_data[dl] = float(np.mean(corrs))

        # Fit: C(dl) ~ dl^(-2*Delta)
        dl_arr = np.array(list(corr_data.keys()), dtype=float)
        corr_arr = np.array(list(corr_data.values()))
        # Only fit positive correlations
        valid_mask = corr_arr > 0
        if valid_mask.sum() >= 2:
            log_dl = np.log(dl_arr[valid_mask])
            log_corr = np.log(corr_arr[valid_mask])
            slope_corr, int_corr, r_corr, p_corr, _ = stats.linregress(log_dl, log_corr)
            delta_scaling = -slope_corr / 2  # Delta = -slope/2
        else:
            slope_corr, r_corr, p_corr, delta_scaling = 0, 0, 1, 0

        print(f"  Prediction 2: C(dl) ~ dl^(-2*Delta)")
        print(f"    Delta = {delta_scaling:.4f} (predict: 1.0 for free boson)")
        print(f"    R2 = {r_corr**2:.4f}")

        # ===== TEST 3: Operator spectrum (SVD singular values) =====
        # For c=1 free boson, operator dimensions should scale as Delta_n = n
        h_final = out.hidden_states[-1][0].float().cpu().numpy()
        _, s_final, _ = np.linalg.svd(h_final, full_matrices=False)
        # Normalize singular values
        s_norm_final = s_final / s_final[0]
        # Expected: s_n ~ n^(-Delta) for scaling operators
        n_ops = min(20, len(s_norm_final))
        n_arr = np.arange(1, n_ops + 1, dtype=float)
        log_n = np.log(n_arr)
        log_s = np.log(s_norm_final[:n_ops] + 1e-15)

        slope_spec, _, r_spec, p_spec, _ = stats.linregress(log_n, log_s)
        spectral_delta = -slope_spec  # Power law decay exponent

        print(f"  Prediction 3: Operator spectrum Delta_n ~ n")
        print(f"    Spectral decay = {spectral_delta:.4f}")
        print(f"    R2 = {r_spec**2:.4f}")

        # ===== TEST 4: Scaling dimension universality =====
        # Compare c values from different methods
        c_values = {
            'entropy_scaling': round(c_from_entropy, 4),
            'correlation_delta': round(delta_scaling * 2, 4),  # c ~ 2*Delta for free boson
            'spectral': round(spectral_delta, 4),
        }
        c_mean = float(np.mean(list(c_values.values())))
        c_std = float(np.std(list(c_values.values())))

        print(f"  Summary of c estimates: {c_values}")
        print(f"    Mean c = {c_mean:.4f} +/- {c_std:.4f}")

        all_results[size] = {
            'n_layers': n_layers,
            'D': D,
            'entropy': {
                'c_eff': round(c_from_entropy, 4),
                'R2': round(r_ent**2, 4),
                'p': round(float(p_ent), 6),
                'profile': [round(s, 4) for s in svd_entropies],
            },
            'correlation': {
                'delta': round(delta_scaling, 4),
                'R2': round(r_corr**2, 4),
                'decay_exponent': round(slope_corr, 4),
                'data': {str(k): round(v, 4) for k, v in corr_data.items()},
            },
            'spectrum': {
                'spectral_delta': round(spectral_delta, 4),
                'R2': round(r_spec**2, 4),
                'top_singular': [round(float(s), 4) for s in s_norm_final[:10]],
            },
            'c_estimates': c_values,
            'c_mean': round(c_mean, 4),
            'c_std': round(c_std, 4),
        }

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Entropy scaling: S vs log(l)
    for si, (size, data) in enumerate(all_results.items()):
        layers = np.arange(len(data['entropy']['profile']))
        axes[0, 0].plot(layers, data['entropy']['profile'], 'o-', color=colors[size],
                       lw=2, markersize=4, label=f'{size} (c_eff={data["entropy"]["c_eff"]:.2f})')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('SVD Entropy')
    axes[0, 0].set_title('(a) Entanglement Entropy', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Correlation decay: C(dl) vs dl (log-log)
    for size, data in all_results.items():
        dls = [int(k) for k in data['correlation']['data'].keys()]
        corrs = [data['correlation']['data'][str(k)] for k in dls]
        positive = [(d, c) for d, c in zip(dls, corrs) if c > 0]
        if positive:
            axes[0, 1].loglog([p[0] for p in positive], [p[1] for p in positive],
                            'o-', color=colors[size], lw=2,
                            label=f'{size} (Delta={data["correlation"]["delta"]:.2f})')
    axes[0, 1].set_xlabel('Layer Separation dl')
    axes[0, 1].set_ylabel('Correlation C(dl)')
    axes[0, 1].set_title('(b) Correlation Decay', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Singular value spectrum (log-log)
    for size, data in all_results.items():
        s_vals = data['spectrum']['top_singular']
        axes[0, 2].semilogy(range(1, len(s_vals)+1), s_vals, 'o-', color=colors[size],
                           lw=2, markersize=4,
                           label=f'{size} (Delta={data["spectrum"]["spectral_delta"]:.2f})')
    axes[0, 2].set_xlabel('Mode Index n')
    axes[0, 2].set_ylabel('Normalized Singular Value')
    axes[0, 2].set_title('(c) Operator Spectrum', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) c estimates comparison
    methods = list(next(iter(all_results.values()))['c_estimates'].keys())
    x = np.arange(len(methods))
    w = 0.35
    for i, (size, data) in enumerate(all_results.items()):
        vals = [data['c_estimates'][m] for m in methods]
        axes[1, 0].bar(x + i*w - w/2, vals, w, label=size, color=colors[size])
    axes[1, 0].axhline(1.0, color='gold', ls='--', lw=2, label='c=1 (free boson)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=8)
    axes[1, 0].set_ylabel('Central Charge c')
    axes[1, 0].set_title('(d) c from Multiple Methods', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) S vs log(l) fit
    for size, data in all_results.items():
        profile = data['entropy']['profile'][1:]  # skip layer 0
        n = len(profile)
        log_l = np.log(np.arange(1, n+1))
        axes[1, 1].plot(log_l, profile, 'o', color=colors[size], markersize=4)
        # Fit line
        c = data['entropy']['c_eff']
        r2 = data['entropy']['R2']
        fit_y = (c/3) * log_l + (profile[0] - (c/3)*log_l[0])
        axes[1, 1].plot(log_l, fit_y, '--', color=colors[size], alpha=0.5,
                       label=f'{size}: S=(c/3)ln(l), c={c:.2f}, R2={r2:.3f}')
    axes[1, 1].set_xlabel('log(Layer)')
    axes[1, 1].set_ylabel('SVD Entropy')
    axes[1, 1].set_title('(e) CFT Entropy Law Fit', fontweight='bold')
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "CFT PREDICTIONS (c=1)\n\n"
    for size, data in all_results.items():
        txt += f"{size}:\n"
        txt += f"  c(entropy)   = {data['c_estimates']['entropy_scaling']:.3f}\n"
        txt += f"  c(corr)      = {data['c_estimates']['correlation_delta']:.3f}\n"
        txt += f"  c(spectrum)  = {data['c_estimates']['spectral']:.3f}\n"
        txt += f"  c_mean       = {data['c_mean']:.3f}\n\n"
    txt += "Free boson: c=1\n"
    txt += "Meaning: 1 effective\n"
    txt += "degree of freedom per\n"
    txt += "layer transition"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 291: CFT Predictions from c=1 (Free Boson)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase291_cft_predictions')
    plt.close()

    save_results('phase291_cft_predictions', {
        'experiment': 'CFT Predictions from c=1',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
