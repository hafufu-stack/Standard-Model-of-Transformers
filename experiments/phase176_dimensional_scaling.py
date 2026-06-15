# -*- coding: utf-8 -*-
"""
Phase 176: Dimensional Scaling
Test how universality class and universal constants scale with model dimension.
Use 3 available models (0.5B, 1.1B, 1.5B) to extrapolate to 7B+.
Key question: Does beta (critical exponent) shift from 2D XY to 3D Ising?
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
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]


def sigmoid(x, L_max, k, x0):
    return L_max / (1 + np.exp(-k * (x - x0)))


def measure_model(model, tok, device, prompts):
    """Measure key thermodynamic quantities for a model."""
    n_layers = len(model.model.layers) + 1
    all_U = []
    all_T = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        U_vals, T_vals = [], []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            U_vals.append(hs[0, -1, :].float().norm().item())
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_vals.append(T if not np.isnan(T) else 0.0)
        all_U.append(U_vals)
        all_T.append(T_vals)

    U_mean = np.mean(all_U, axis=0)
    T_mean = np.mean(all_T, axis=0)

    # Sigmoid fit for U (order parameter)
    U_norm = (U_mean - U_mean.min()) / (U_mean.max() - U_mean.min() + 1e-10)
    layers = np.arange(n_layers)
    try:
        popt, _ = curve_fit(sigmoid, layers, U_norm, p0=[1.0, 0.5, n_layers * 0.7], maxfev=5000)
        L0 = popt[2]
        L0_ratio = L0 / n_layers
        U_fit = sigmoid(layers, *popt)
        r2 = 1 - np.sum((U_norm - U_fit) ** 2) / np.sum((U_norm - np.mean(U_norm)) ** 2)
    except Exception:
        L0 = n_layers * 0.75
        L0_ratio = 0.75
        r2 = 0.0

    # Carnot efficiency
    T_hot = np.mean(T_mean[:3])
    T_cold = np.mean(T_mean[-3:])
    eta = 1 - T_cold / (T_hot + 1e-10)

    # Cv estimation (slope dU/dT)
    cv_vals = []
    for i in range(1, n_layers):
        dU = U_mean[i] - U_mean[i - 1]
        dT = T_mean[i] - T_mean[i - 1]
        if abs(dT) > 1e-6:
            cv_vals.append(dU / dT)
    Cv_mean = np.mean(cv_vals) if cv_vals else 0

    # Critical exponent beta from transition steepness
    # beta ~ 1/(2*k) where k is sigmoid steepness
    try:
        beta_crit = 1.0 / (2.0 * abs(popt[1]) + 1e-10)
    except Exception:
        beta_crit = 0.0

    return {
        'L0': float(L0), 'L0_ratio': float(L0_ratio), 'eta': float(eta),
        'Cv_mean': float(Cv_mean), 'beta_crit': float(beta_crit),
        'r2_sigmoid': float(r2), 'n_layers': n_layers,
        'T_hot': float(T_hot), 'T_cold': float(T_cold),
        'U_mean': [float(x) for x in U_mean],
        'T_mean': [float(x) for x in T_mean],
    }


def main():
    print("=" * 70)
    print("Phase 176: Dimensional Scaling")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_all = []

    # Qwen2.5-1.5B
    print("\n--- Qwen2.5-1.5B ---")
    model, tok = load_model(device=device, size='1.5B')
    d_model = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    r = measure_model(model, tok, device, PROMPTS)
    r['name'] = 'Qwen2.5-1.5B'
    r['d_model'] = d_model
    r['n_params'] = float(n_params)
    results_all.append(r)
    print(f"  d={d_model}, L0/L={r['L0_ratio']:.3f}, eta={r['eta']:.4f}, beta={r['beta_crit']:.4f}")
    del model; torch.cuda.empty_cache()

    # Qwen2.5-0.5B
    print("\n--- Qwen2.5-0.5B ---")
    model, tok = load_model(device=device, size='0.5B')
    d_model = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    r = measure_model(model, tok, device, PROMPTS)
    r['name'] = 'Qwen2.5-0.5B'
    r['d_model'] = d_model
    r['n_params'] = float(n_params)
    results_all.append(r)
    print(f"  d={d_model}, L0/L={r['L0_ratio']:.3f}, eta={r['eta']:.4f}, beta={r['beta_crit']:.4f}")
    del model; torch.cuda.empty_cache()

    # TinyLlama-1.1B
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tinyllama_path = os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
        snaps = os.path.join(tinyllama_path, "snapshots")
        snap = os.listdir(snaps)[0]
        mid = os.path.join(snaps, snap)
        print("\n--- TinyLlama-1.1B ---")
        tok3 = AutoTokenizer.from_pretrained(mid, local_files_only=True)
        model3 = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map=device, local_files_only=True)
        model3.eval()
        d_model = model3.config.hidden_size
        n_params = sum(p.numel() for p in model3.parameters()) / 1e9
        r = measure_model(model3, tok3, device, PROMPTS)
        r['name'] = 'TinyLlama-1.1B'
        r['d_model'] = d_model
        r['n_params'] = float(n_params)
        results_all.append(r)
        print(f"  d={d_model}, L0/L={r['L0_ratio']:.3f}, eta={r['eta']:.4f}, beta={r['beta_crit']:.4f}")
        del model3; torch.cuda.empty_cache()
    except Exception as e:
        print(f"TinyLlama not available: {e}")

    # === Scaling extrapolation ===
    d_models = [r['d_model'] for r in results_all]
    n_params_list = [r['n_params'] for r in results_all]
    etas = [r['eta'] for r in results_all]
    betas = [r['beta_crit'] for r in results_all]
    L0_ratios = [r['L0_ratio'] for r in results_all]
    Cvs = [r['Cv_mean'] for r in results_all]

    # Extrapolate to 7B (d_model ~ 4096)
    d_7b = 4096
    n_7b = 7.0

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) eta vs d_model
    axes[0, 0].scatter(d_models, etas, s=120, c='#3498db', edgecolors='black', zorder=5)
    for i, r in enumerate(results_all):
        axes[0, 0].annotate(r['name'], (d_models[i], etas[i]),
                            textcoords="offset points", xytext=(5, 10), fontsize=8)
    if len(d_models) >= 2:
        z = np.polyfit(d_models, etas, 1)
        d_range = np.linspace(min(d_models) * 0.5, d_7b * 1.1, 50)
        axes[0, 0].plot(d_range, np.polyval(z, d_range), '--', color='gray', alpha=0.5)
        eta_7b = np.polyval(z, d_7b)
        axes[0, 0].scatter([d_7b], [eta_7b], s=150, marker='*', c='#e74c3c', zorder=10, label=f'7B pred: {eta_7b:.3f}')
    axes[0, 0].set_xlabel('$d_{model}$')
    axes[0, 0].set_ylabel('$\\eta$')
    axes[0, 0].set_title('(a) Carnot Efficiency Scaling')
    axes[0, 0].legend(fontsize=8)

    # (b) beta vs d_model
    axes[0, 1].scatter(d_models, betas, s=120, c='#e74c3c', edgecolors='black', zorder=5)
    axes[0, 1].axhline(y=0.161, color='blue', linestyle='--', alpha=0.5, label='2D XY ($\\beta=0.161$)')
    axes[0, 1].axhline(y=0.326, color='green', linestyle='--', alpha=0.5, label='3D Ising ($\\beta=0.326$)')
    axes[0, 1].axhline(y=0.5, color='purple', linestyle='--', alpha=0.5, label='Mean Field ($\\beta=0.5$)')
    for i, r in enumerate(results_all):
        axes[0, 1].annotate(r['name'], (d_models[i], betas[i]),
                            textcoords="offset points", xytext=(5, 10), fontsize=8)
    axes[0, 1].set_xlabel('$d_{model}$')
    axes[0, 1].set_ylabel('$\\beta_{crit}$')
    axes[0, 1].set_title('(b) Critical Exponent Scaling')
    axes[0, 1].legend(fontsize=7)

    # (c) L0/L vs d_model
    axes[0, 2].scatter(d_models, L0_ratios, s=120, c='#2ecc71', edgecolors='black', zorder=5)
    axes[0, 2].axhline(y=0.75, color='black', linestyle='--', alpha=0.3, label='$L_0/L = 0.75$')
    for i, r in enumerate(results_all):
        axes[0, 2].annotate(r['name'], (d_models[i], L0_ratios[i]),
                            textcoords="offset points", xytext=(5, 10), fontsize=8)
    axes[0, 2].set_xlabel('$d_{model}$')
    axes[0, 2].set_ylabel('$L_0 / L$')
    axes[0, 2].set_title('(c) Critical Layer Ratio Scaling')
    axes[0, 2].legend(fontsize=8)

    # (d) Cv vs d_model
    axes[1, 0].scatter(d_models, Cvs, s=120, c='#f39c12', edgecolors='black', zorder=5)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    for i, r in enumerate(results_all):
        axes[1, 0].annotate(r['name'], (d_models[i], Cvs[i]),
                            textcoords="offset points", xytext=(5, 10), fontsize=8)
    axes[1, 0].set_xlabel('$d_{model}$')
    axes[1, 0].set_ylabel('$C_v$')
    axes[1, 0].set_title('(d) Specific Heat Scaling')

    # (e) U profile comparison
    for r in results_all:
        layers_norm = np.linspace(0, 1, len(r['U_mean']))
        U_norm = (np.array(r['U_mean']) - min(r['U_mean'])) / (max(r['U_mean']) - min(r['U_mean']) + 1e-10)
        axes[1, 1].plot(layers_norm, U_norm, 'o-', markersize=3, label=r['name'], linewidth=2)
    axes[1, 1].axvline(x=0.75, color='black', linestyle='--', alpha=0.3, label='$L_0/L$')
    axes[1, 1].set_xlabel('Normalized Layer $l/L$')
    axes[1, 1].set_ylabel('Normalized $U$')
    axes[1, 1].set_title('(e) Universal U Profile')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    summary = "Dimensional Scaling\n\n"
    for r in results_all:
        summary += f"{r['name']}:\n  d={r['d_model']}, L0/L={r['L0_ratio']:.3f}\n  eta={r['eta']:.3f}, beta={r['beta_crit']:.3f}\n"
    summary += f"\nPrediction for 7B:\n"
    if len(d_models) >= 2:
        summary += f"  eta -> {eta_7b:.3f}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 176: Dimensional Scaling', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase176_dimensional_scaling')
    plt.close()

    print(f"\n{'=' * 70}")
    for r in results_all:
        print(f"{r['name']}: d={r['d_model']}, eta={r['eta']:.4f}, beta={r['beta_crit']:.4f}, L0/L={r['L0_ratio']:.3f}")
    print(f"{'=' * 70}")

    save_results('phase176_dimensional_scaling', {
        'experiment': 'Dimensional Scaling',
        'models': results_all,
    })


if __name__ == '__main__':
    main()
