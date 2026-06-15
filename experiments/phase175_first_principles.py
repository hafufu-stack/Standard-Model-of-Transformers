# -*- coding: utf-8 -*-
"""
Phase 175: First Principles eta Derivation
Test whether Carnot efficiency eta=0.813 can be derived from
architectural parameters alone: d_model, d_ffn, n_layers.
Candidate formulas: 1 - d_model/d_ffn, 1 - sqrt(d_model/d_ffn), etc.
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
]


def measure_eta(model, tok, device, prompts):
    """Measure Carnot efficiency: eta = 1 - T_cold / T_hot."""
    n_layers = len(model.model.layers) + 1
    all_T = []
    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_vals.append(T if not np.isnan(T) else 0.0)
        all_T.append(T_vals)

    T_array = np.array(all_T)
    T_hot = np.mean(T_array[:, :3])  # First 3 layers
    T_cold = np.mean(T_array[:, -3:])  # Last 3 layers
    eta = 1.0 - T_cold / (T_hot + 1e-10)
    return float(eta), float(T_hot), float(T_cold)


def main():
    print("=" * 70)
    print("Phase 175: First Principles eta Derivation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Measure eta for all 3 models
    model_configs = []

    # Qwen2.5-1.5B: d_model=1536, d_ffn=8960, n_layers=28
    print("\n--- Qwen2.5-1.5B ---")
    model, tok = load_model(device=device, size='1.5B')
    d_model = model.config.hidden_size
    d_ffn = model.config.intermediate_size
    n_layers = model.config.num_hidden_layers
    eta, T_hot, T_cold = measure_eta(model, tok, device, PROMPTS)
    model_configs.append({
        'name': 'Qwen2.5-1.5B', 'd_model': d_model, 'd_ffn': d_ffn,
        'n_layers': n_layers, 'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
    })
    print(f"  d_model={d_model}, d_ffn={d_ffn}, L={n_layers}, eta={eta:.4f}")
    del model
    torch.cuda.empty_cache()

    # Qwen2.5-0.5B: d_model=896, d_ffn=4864, n_layers=24
    print("\n--- Qwen2.5-0.5B ---")
    model, tok = load_model(device=device, size='0.5B')
    d_model = model.config.hidden_size
    d_ffn = model.config.intermediate_size
    n_layers = model.config.num_hidden_layers
    eta, T_hot, T_cold = measure_eta(model, tok, device, PROMPTS)
    model_configs.append({
        'name': 'Qwen2.5-0.5B', 'd_model': d_model, 'd_ffn': d_ffn,
        'n_layers': n_layers, 'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
    })
    print(f"  d_model={d_model}, d_ffn={d_ffn}, L={n_layers}, eta={eta:.4f}")
    del model
    torch.cuda.empty_cache()

    # TinyLlama-1.1B: d_model=2048, d_ffn=5632, n_layers=22
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tinyllama_path = os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
        if os.path.exists(tinyllama_path):
            snaps = os.path.join(tinyllama_path, "snapshots")
            snap = os.listdir(snaps)[0]
            mid = os.path.join(snaps, snap)
            print("\n--- TinyLlama-1.1B ---")
            tok3 = AutoTokenizer.from_pretrained(mid, local_files_only=True)
            model3 = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, device_map=device, local_files_only=True)
            model3.eval()
            d_model = model3.config.hidden_size
            d_ffn = model3.config.intermediate_size
            n_layers = model3.config.num_hidden_layers
            eta, T_hot, T_cold = measure_eta(model3, tok3, device, PROMPTS)
            model_configs.append({
                'name': 'TinyLlama-1.1B', 'd_model': d_model, 'd_ffn': d_ffn,
                'n_layers': n_layers, 'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
            })
            print(f"  d_model={d_model}, d_ffn={d_ffn}, L={n_layers}, eta={eta:.4f}")
            del model3
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"TinyLlama not available: {e}")

    # === Test candidate formulas ===
    formulas = {}
    for cfg in model_configs:
        d, f, L = cfg['d_model'], cfg['d_ffn'], cfg['n_layers']
        formulas[cfg['name']] = {
            'measured_eta': cfg['eta'],
            '1 - d/f': 1 - d / f,
            '1 - sqrt(d/f)': 1 - np.sqrt(d / f),
            '1 - (d/f)^2': 1 - (d / f) ** 2,
            '1 - 1/sqrt(L)': 1 - 1 / np.sqrt(L),
            '1 - d/(d+f)': 1 - d / (d + f),
        }

    # Find best formula
    formula_names = ['1 - d/f', '1 - sqrt(d/f)', '1 - (d/f)^2', '1 - 1/sqrt(L)', '1 - d/(d+f)']
    formula_errors = {}
    for fn in formula_names:
        errors = []
        for name, vals in formulas.items():
            errors.append(abs(vals[fn] - vals['measured_eta']))
        formula_errors[fn] = np.mean(errors)

    best_formula = min(formula_errors, key=formula_errors.get)
    best_error = formula_errors[best_formula]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Measured eta per model
    names = [c['name'] for c in model_configs]
    etas = [c['eta'] for c in model_configs]
    bars = axes[0, 0].bar(names, etas, color=['#3498db', '#e74c3c', '#2ecc71'][:len(names)],
                          edgecolor='black', alpha=0.8)
    axes[0, 0].axhline(y=0.813, color='black', linestyle='--', label='$\\eta = 0.813$')
    for bar, e in zip(bars, etas):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{e:.3f}', ha='center', fontsize=10)
    axes[0, 0].set_ylabel('$\\eta$')
    axes[0, 0].set_title('(a) Measured Carnot Efficiency')
    axes[0, 0].legend(fontsize=8)

    # (b) d_model / d_ffn ratio
    ratios = [c['d_model'] / c['d_ffn'] for c in model_configs]
    axes[0, 1].scatter(ratios, etas, s=120, c=['#3498db', '#e74c3c', '#2ecc71'][:len(names)],
                       edgecolors='black', zorder=5)
    for i, n in enumerate(names):
        axes[0, 1].annotate(n, (ratios[i], etas[i]), textcoords="offset points",
                            xytext=(5, 10), fontsize=8)
    # Plot best formula line
    r_range = np.linspace(min(ratios) * 0.8, max(ratios) * 1.2, 50)
    if best_formula == '1 - d/f':
        axes[0, 1].plot(r_range, 1 - r_range, 'k--', alpha=0.5, label=best_formula)
    elif best_formula == '1 - sqrt(d/f)':
        axes[0, 1].plot(r_range, 1 - np.sqrt(r_range), 'k--', alpha=0.5, label=best_formula)
    elif best_formula == '1 - (d/f)^2':
        axes[0, 1].plot(r_range, 1 - r_range**2, 'k--', alpha=0.5, label=best_formula)
    axes[0, 1].set_xlabel('$d_{model} / d_{ffn}$')
    axes[0, 1].set_ylabel('$\\eta$')
    axes[0, 1].set_title('(b) $\\eta$ vs Dimension Ratio')
    axes[0, 1].legend(fontsize=8)

    # (c) Formula comparison
    fn_labels = list(formula_errors.keys())
    fn_vals = list(formula_errors.values())
    colors_c = ['#2ecc71' if v == best_error else '#95a5a6' for v in fn_vals]
    axes[0, 2].barh(fn_labels, fn_vals, color=colors_c, edgecolor='black')
    axes[0, 2].set_xlabel('Mean Absolute Error')
    axes[0, 2].set_title('(c) Formula Error Ranking')
    axes[0, 2].invert_yaxis()

    # (d) T_hot and T_cold
    x = np.arange(len(names))
    w = 0.35
    axes[1, 0].bar(x - w/2, [c['T_hot'] for c in model_configs], w, label='$T_{hot}$', color='#e74c3c', alpha=0.8)
    axes[1, 0].bar(x + w/2, [c['T_cold'] for c in model_configs], w, label='$T_{cold}$', color='#3498db', alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title('(d) $T_{hot}$ vs $T_{cold}$')
    axes[1, 0].legend(fontsize=8)

    # (e) Predicted vs measured eta
    for fn, marker in zip(formula_names[:4], ['o', 's', '^', 'D']):
        pred = [formulas[n][fn] for n in [c['name'] for c in model_configs]]
        axes[1, 1].scatter(etas, pred, marker=marker, s=80, label=fn, alpha=0.7)
    axes[1, 1].plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.3)
    axes[1, 1].set_xlabel('Measured $\\eta$')
    axes[1, 1].set_ylabel('Predicted $\\eta$')
    axes[1, 1].set_title('(e) Predicted vs Measured')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    summary = (
        f"First Principles eta\n\n"
        f"Best formula: {best_formula}\n"
        f"Mean error: {best_error:.4f}\n\n"
    )
    for cfg in model_configs:
        pred = formulas[cfg['name']][best_formula]
        summary += f"{cfg['name']}:\n  measured={cfg['eta']:.4f}, pred={pred:.4f}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 175: First Principles $\\eta$ Derivation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase175_first_principles')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Best formula: {best_formula} (error={best_error:.4f})")
    for cfg in model_configs:
        pred = formulas[cfg['name']][best_formula]
        print(f"  {cfg['name']}: measured={cfg['eta']:.4f}, predicted={pred:.4f}")
    print(f"{'=' * 70}")

    save_results('phase175_first_principles', {
        'experiment': 'First Principles eta Derivation',
        'model_configs': model_configs,
        'formulas': formulas,
        'formula_errors': formula_errors,
        'best_formula': best_formula,
        'best_error': float(best_error),
    })


if __name__ == '__main__':
    main()
