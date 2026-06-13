# -*- coding: utf-8 -*-
"""
Phase 99: Eta Phase Transition Universality
Phase 97 found sigmoid transition with L0=21.7 for Qwen 1.5B (28 layers).
Test if L0/L_total ratio is universal across models:
- Qwen 1.5B: L=28, L0=21.7 -> ratio=0.776
- Qwen 0.5B: L=24 -> predicted L0=18.6
- TinyLlama: L=22 -> predicted L0=17.1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
]

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
    "The cosmic microwave background reveals the early universe",
    "General relativity describes gravity as spacetime curvature",
]


def measure_eta_at_depth(model, tok, device, max_layer, norm_layer, lm_head):
    """Measure eta using layers 0..max_layer."""
    etas = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(min(max_layer + 1, len(out.hidden_states))):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(T):
                T_vals.append(T)

        if len(T_vals) >= 4:
            T_hot = max(T_vals)
            T_cold = min(T_vals[len(T_vals)//2:])
            if T_hot > 0.01:
                etas.append(1.0 - T_cold / T_hot)

    return float(np.mean(etas)) if etas else 0.0


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


def main():
    print("=" * 70)
    print("Phase 99: Eta Phase Transition Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_model_data = {}

    # Include Phase 97 result for Qwen 1.5B
    all_model_data['Qwen2.5-1.5B'] = {
        'n_layers': 28, 'L0': 21.7, 'ratio': 21.7/28,
        'source': 'Phase 97',
    }

    for model_id, model_name in MODELS:
        print(f"\n--- {model_name} ---")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map=device,
                trust_remote_code=True)
            model.eval()
        except Exception as e:
            print(f"  Failed: {str(e)[:80]}")
            continue

        n_layers = len(model.model.layers) + 1
        norm_layer = model.model.norm
        lm_head = model.lm_head

        layer_counts = list(range(4, n_layers))
        etas_by_L = []

        for L in layer_counts:
            eta = measure_eta_at_depth(model, tok, device, L, norm_layer, lm_head)
            etas_by_L.append({'L': L, 'eta': float(eta)})
            theory = 1.0 - 1.0/np.sqrt(L)
            print(f"  L={L:2d}: eta={eta:.4f} (theory={theory:.4f})")

        Ls = np.array([r['L'] for r in etas_by_L])
        etas = np.array([r['eta'] for r in etas_by_L])

        try:
            popt, _ = curve_fit(sigmoid, Ls, etas,
                                p0=[n_layers*0.7, 0.5, np.min(etas), np.max(etas)],
                                maxfev=10000)
            L0_fit = popt[0]
            sig_pred = sigmoid(Ls, *popt)
            ss_res = np.sum((etas - sig_pred)**2)
            ss_tot = np.sum((etas - np.mean(etas))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            ratio = L0_fit / (n_layers - 1)
            print(f"  Sigmoid: L0={L0_fit:.1f}, ratio={ratio:.3f}, R2={r2:.4f}")
        except Exception as e:
            print(f"  Fit failed: {e}")
            L0_fit = n_layers * 0.7
            r2 = 0
            ratio = 0.7

        all_model_data[model_name] = {
            'n_layers': n_layers - 1,
            'L0': float(L0_fit),
            'ratio': float(ratio),
            'r2': float(r2),
            'etas_by_L': etas_by_L,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Qwen2.5-1.5B': '#c0392b', 'Qwen2.5-0.5B': '#2980b9', 'TinyLlama-1.1B': '#27ae60'}

    # (a) All models' eta curves
    for mname, data in all_model_data.items():
        if 'etas_by_L' in data:
            Ls = [r['L'] for r in data['etas_by_L']]
            etas = [r['eta'] for r in data['etas_by_L']]
            axes[0].plot(Ls, etas, 'o-', color=colors.get(mname, 'gray'),
                        markersize=4, linewidth=1.5, label=f'{mname} (L={data["n_layers"]})')
    L_sm = np.linspace(4, 30, 200)
    axes[0].plot(L_sm, 1-1/np.sqrt(L_sm), 'k--', alpha=0.4, label='$1-1/\\sqrt{L}$')
    axes[0].set_xlabel('Effective Layer Count')
    axes[0].set_ylabel('$\\eta$')
    axes[0].set_title('(a) Eta vs Layer Count')
    axes[0].legend(fontsize=7)

    # (b) L0/L_total ratio comparison
    names = list(all_model_data.keys())
    ratios = [all_model_data[m]['ratio'] for m in names]
    bar_colors = [colors.get(m, 'gray') for m in names]
    axes[1].bar(range(len(names)), ratios, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylabel('$L_0 / L_{total}$')
    mean_ratio = np.mean(ratios)
    cv_ratio = np.std(ratios) / (mean_ratio + 1e-10)
    axes[1].axhline(y=mean_ratio, color='black', linestyle='--',
                    label=f'Mean = {mean_ratio:.3f}')
    axes[1].set_title(f'(b) Transition Ratio (CV={cv_ratio:.3f})')
    axes[1].legend()

    # (c) Summary
    is_universal = cv_ratio < 0.15
    summary = f"Eta Phase Transition\n\n"
    for m in names:
        d = all_model_data[m]
        summary += f"{m}:\n  L={d['n_layers']}, L0={d['L0']:.1f}, ratio={d['ratio']:.3f}\n"
    summary += f"\nMean ratio: {mean_ratio:.3f}\nCV: {cv_ratio:.3f}\n"
    summary += f"\n{'UNIVERSAL' if is_universal else 'MODEL-DEPENDENT'}"

    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('(c) Summary')

    fig.suptitle(f'Phase 99: Eta Transition Universality '
                 f'($L_0/L$ = {mean_ratio:.3f}, CV = {cv_ratio:.3f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase99_eta_universality')
    plt.close()

    print(f"\n{'='*70}")
    print(f"L0/L ratios: {[f'{r:.3f}' for r in ratios]}")
    print(f"Mean ratio: {mean_ratio:.3f}, CV: {cv_ratio:.3f}")
    print(f"VERDICT: {'UNIVERSAL' if is_universal else 'MODEL-DEPENDENT'}")
    print(f"{'='*70}")

    save_results('phase99_eta_universality', {
        'experiment': 'Eta Phase Transition Universality',
        'models': {m: {k: v for k, v in d.items() if k != 'etas_by_L'}
                   for m, d in all_model_data.items()},
        'etas_by_model': {m: d.get('etas_by_L', []) for m, d in all_model_data.items()},
        'summary': {
            'mean_ratio': float(mean_ratio),
            'cv': float(cv_ratio),
            'is_universal': is_universal,
        }
    })


if __name__ == '__main__':
    main()
