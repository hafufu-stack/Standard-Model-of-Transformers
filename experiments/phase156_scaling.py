# -*- coding: utf-8 -*-
"""
Phase 156: Scaling Law of L0
Measure L0/L ratio across different model sizes in the Qwen2.5 family.
If L0/L is constant, this is a universal scaling law.
Uses only locally available models.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import save_results, save_figure


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


def safe_str(s):
    return s.encode('ascii', errors='replace').decode('ascii')


# Models to try (locally available)
MODELS = [
    ("Qwen/Qwen2.5-0.5B", "0.5B"),
    ("Qwen/Qwen2.5-1.5B", "1.5B"),
    ("Qwen/Qwen2.5-3B", "3B"),
    ("Qwen/Qwen2.5-7B", "7B"),
]


def analyze_model(model_name, label, prompts, device):
    """Analyze a single model and return L0, n_layers, etc."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading {label}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map=device, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as e:
        print(f"    SKIP: {e}")
        return None

    n_layers = len(model.model.layers) + 1
    print(f"    {n_layers} layers")

    all_eta = [[] for _ in range(n_layers)]

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(min(n_layers, len(out.hidden_states))):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_vals.append(S if not np.isnan(S) else 0)

        for li in range(n_layers):
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0
            all_eta[li].append(eta)

    avg_eta = [np.mean(v) if v else 0 for v in all_eta]

    # Fit sigmoid
    try:
        skip = max(2, n_layers // 8)
        Ls = np.arange(skip, n_layers)
        popt, _ = curve_fit(sigmoid, Ls, avg_eta[skip:],
                            p0=[n_layers * 0.75, 0.5, 0, 0.9], maxfev=10000)
        L0 = popt[0]
        sig_pred = sigmoid(Ls, *popt)
        r2 = 1 - np.sum((np.array(avg_eta[skip:]) - sig_pred)**2) / (
            np.sum((np.array(avg_eta[skip:]) - np.mean(avg_eta[skip:]))**2) + 1e-10)
    except:
        L0 = n_layers * 0.75
        r2 = 0

    result = {
        'model': model_name,
        'label': label,
        'n_layers': n_layers,
        'L0': float(L0),
        'L0_ratio': float(L0 / n_layers),
        'R2': float(r2),
        'eta': avg_eta,
        'final_eta': float(avg_eta[-1]),
    }

    print(f"    L0={L0:.1f}, L0/L={L0/n_layers:.3f}, R2={r2:.3f}")

    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return result


def main():
    print("=" * 70)
    print("Phase 156: Scaling Law of L0")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for model_name, label in MODELS:
        r = analyze_model(model_name, label, PROMPTS, device)
        if r is not None:
            results[label] = r

    if len(results) < 2:
        print("WARNING: Need at least 2 models for scaling law")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_m = {'0.5B': '#e74c3c', '1.5B': '#2980b9', '3B': '#27ae60', '7B': '#f39c12'}

    # (a) Eta profiles for all models
    for label, r in results.items():
        axes[0,0].plot(np.arange(r['n_layers']) / r['n_layers'],
                      r['eta'], 'o-', color=colors_m.get(label, 'gray'),
                      markersize=3, linewidth=2,
                      label=f"{label} (L0/L={r['L0_ratio']:.3f})")
    axes[0,0].set_xlabel('Relative Depth (l/L)')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Normalized Eta Profiles')
    axes[0,0].legend(fontsize=8)

    # (b) L0 vs n_layers
    labels = list(results.keys())
    n_layers_list = [results[l]['n_layers'] for l in labels]
    L0_list = [results[l]['L0'] for l in labels]
    bar_c = [colors_m.get(l, 'gray') for l in labels]
    axes[0,1].scatter(n_layers_list, L0_list, c=bar_c, s=150, edgecolors='black', zorder=5)
    for i, label in enumerate(labels):
        axes[0,1].annotate(label, (n_layers_list[i], L0_list[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    # Fit linear
    if len(n_layers_list) >= 2:
        slope, intercept = np.polyfit(n_layers_list, L0_list, 1)
        xr = np.linspace(min(n_layers_list)-2, max(n_layers_list)+2, 100)
        axes[0,1].plot(xr, slope * xr + intercept, '--', color='gray',
                      label=f'L0 = {slope:.2f}*L + {intercept:.1f}')
    axes[0,1].set_xlabel('Total Layers')
    axes[0,1].set_ylabel('$L_0$')
    axes[0,1].set_title('(b) L0 Scaling')
    axes[0,1].legend()

    # (c) L0/L ratio
    ratios = [results[l]['L0_ratio'] for l in labels]
    axes[0,2].bar(range(len(labels)), ratios, color=bar_c, alpha=0.8, edgecolor='black')
    axes[0,2].set_xticks(range(len(labels)))
    axes[0,2].set_xticklabels(labels)
    mean_ratio = np.mean(ratios)
    axes[0,2].axhline(y=mean_ratio, color='black', linestyle='--',
                      label=f'Mean={mean_ratio:.3f}')
    axes[0,2].set_ylabel('$L_0 / L$')
    axes[0,2].set_title('(c) Universal Ratio')
    axes[0,2].legend()

    # (d) R2 by model
    r2s = [results[l]['R2'] for l in labels]
    axes[1,0].bar(range(len(labels)), r2s, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,0].set_xticks(range(len(labels)))
    axes[1,0].set_xticklabels(labels)
    axes[1,0].set_ylabel('$R^2$')
    axes[1,0].set_title('(d) Sigmoid Fit Quality')

    # (e) Final eta by model
    final_etas = [results[l]['final_eta'] for l in labels]
    axes[1,1].bar(range(len(labels)), final_etas, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(labels)))
    axes[1,1].set_xticklabels(labels)
    axes[1,1].set_ylabel('$\\eta_{final}$')
    axes[1,1].set_title('(e) Final Efficiency')

    # (f) Summary
    ratio_cv = np.std(ratios) / (np.mean(ratios) + 1e-10) if ratios else 0
    summary = (
        f"Scaling Law of L0\n\n"
        + "\n".join(f"{l}: L={results[l]['n_layers']}, "
                    f"L0={results[l]['L0']:.1f}, "
                    f"L0/L={results[l]['L0_ratio']:.3f}"
                    for l in labels)
        + f"\n\nMean L0/L: {mean_ratio:.3f}\n"
        f"CV: {ratio_cv:.3f}\n\n"
        f"L0/L is {'UNIVERSAL' if ratio_cv < 0.15 else 'MODEL-SPECIFIC'}\n"
        f"({'constant' if ratio_cv < 0.15 else 'varies'} across sizes)"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 156: Scaling Law of L0',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase156_scaling')
    plt.close()

    print(f"\n{'='*70}")
    for l in labels:
        print(f"  {l}: L={results[l]['n_layers']}, L0={results[l]['L0']:.1f}, "
              f"L0/L={results[l]['L0_ratio']:.3f}")
    print(f"  Mean L0/L={mean_ratio:.3f}, CV={ratio_cv:.3f}")
    print(f"{'='*70}")

    save_results('phase156_scaling', {
        'experiment': 'Scaling Law of L0',
        'results': {l: {k: v for k, v in r.items() if k != 'eta'}
                    for l, r in results.items()},
        'summary': {
            'mean_ratio': float(mean_ratio),
            'ratio_cv': float(ratio_cv),
        }
    })


if __name__ == '__main__':
    main()
