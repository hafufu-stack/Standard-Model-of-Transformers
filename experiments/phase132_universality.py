# -*- coding: utf-8 -*-
"""
Phase 132: Universality across Architectures
Test if the eta phase transition exists in different model architectures:
- Qwen2.5-1.5B (our main model)
- GPT-2 (if available locally)
This tests true universality of the Standard Model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
]


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


def compute_eta_profile(model, tok, prompts, device, model_name, norm_fn, head_fn):
    """Compute eta profile for any model."""
    model.eval()

    # Determine number of layers from hidden states output
    test_inp = tok(prompts[0], return_tensors='pt').to(device)
    with torch.no_grad():
        test_out = model(**test_inp, output_hidden_states=True)
    n_layers = len(test_out.hidden_states)

    all_etas = [[] for _ in range(n_layers)]

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = norm_fn(hs[:, -1:, :])
                logits = head_fn(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_vals.append(T if not np.isnan(T) else 0)

        for li in range(n_layers):
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                if T_hot > 0.01:
                    all_etas[li].append(1.0 - T_cold / T_hot)
                else:
                    all_etas[li].append(0.0)
            else:
                all_etas[li].append(0.0)

    avg_etas = [np.mean(v) if v else 0 for v in all_etas]

    # Fit sigmoid
    Ls = np.arange(4, n_layers)
    eta_fit = np.array(avg_etas[4:])
    try:
        popt, _ = curve_fit(sigmoid, Ls, eta_fit,
                            p0=[n_layers*0.7, 0.5, np.min(eta_fit), np.max(eta_fit)],
                            maxfev=10000)
        L0 = popt[0]
        sig_pred = sigmoid(Ls, *popt)
        r2 = 1 - np.sum((eta_fit - sig_pred)**2) / (np.sum((eta_fit - np.mean(eta_fit))**2) + 1e-10)
    except:
        L0 = n_layers * 0.7
        r2 = 0

    return {
        'model_name': model_name,
        'n_layers': n_layers,
        'eta_profile': avg_etas,
        'L0': float(L0),
        'L0_ratio': float(L0 / n_layers),
        'r2': float(r2),
    }


def main():
    print("=" * 70)
    print("Phase 132: Universality across Architectures")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # Model 1: Qwen2.5-1.5B
    print("\n--- Qwen2.5-1.5B ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model_q = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16,
            device_map=device, local_files_only=True)
        tok_q = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B", local_files_only=True)
        r = compute_eta_profile(model_q, tok_q, PROMPTS, device, 'Qwen2.5-1.5B',
                               model_q.model.norm, model_q.lm_head)
        results['Qwen2.5-1.5B'] = r
        print(f"  L0={r['L0']:.1f}, L0/L={r['L0_ratio']:.3f}, R2={r['r2']:.4f}")
        del model_q
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Model 2: GPT-2
    print("\n--- GPT-2 ---")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model_g = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True).to(device)
        tok_g = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
        r = compute_eta_profile(model_g, tok_g, PROMPTS, device, 'GPT-2',
                               model_g.transformer.ln_f, model_g.lm_head)
        results['GPT-2'] = r
        print(f"  L0={r['L0']:.1f}, L0/L={r['L0_ratio']:.3f}, R2={r['r2']:.4f}")
        del model_g
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Model 3: Qwen2.5-0.5B (smaller)
    print("\n--- Qwen2.5-0.5B ---")
    try:
        model_s = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16,
            device_map=device, local_files_only=True)
        tok_s = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", local_files_only=True)
        r = compute_eta_profile(model_s, tok_s, PROMPTS, device, 'Qwen2.5-0.5B',
                               model_s.model.norm, model_s.lm_head)
        results['Qwen2.5-0.5B'] = r
        print(f"  L0={r['L0']:.1f}, L0/L={r['L0_ratio']:.3f}, R2={r['r2']:.4f}")
        del model_s
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

    if len(results) < 2:
        print("\nWARNING: Only tested on 1 model. Need at least 2 for universality.")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Qwen2.5-1.5B': '#c0392b', 'GPT-2': '#2980b9', 'Qwen2.5-0.5B': '#27ae60'}

    # (a) Eta profiles
    for name, r in results.items():
        n = r['n_layers']
        x = np.linspace(0, 1, n)
        axes[0].plot(x, r['eta_profile'], 'o-', color=colors.get(name, 'gray'),
                    markersize=3, label=f'{name} (L0/L={r["L0_ratio"]:.2f})')
    axes[0].set_xlabel('Normalized Depth (l/L)')
    axes[0].set_ylabel('$\\eta$')
    axes[0].set_title('(a) Universal Eta Profiles')
    axes[0].legend(fontsize=8)

    # (b) L0/L comparison
    names = list(results.keys())
    ratios = [results[n]['L0_ratio'] for n in names]
    r2s = [results[n]['r2'] for n in names]
    bar_c = [colors.get(n, 'gray') for n in names]
    axes[1].bar(range(len(names)), ratios, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=8, rotation=15)
    axes[1].set_ylabel('$L_0 / L$')
    if ratios:
        mean_ratio = np.mean(ratios)
        axes[1].axhline(y=mean_ratio, color='black', linestyle='--',
                        label=f'Mean={mean_ratio:.3f}')
    axes[1].set_title('(b) Transition Point Universality')
    axes[1].legend()

    # (c) Summary
    cv = np.std(ratios) / (np.mean(ratios) + 1e-10) if ratios else 0
    summary = (
        f"Universality Test\n\n"
        + "\n".join(f"{n}: L0/L={r['L0_ratio']:.3f} (R2={r['r2']:.3f})"
                    for n, r in results.items())
        + f"\n\nMean L0/L: {np.mean(ratios):.3f}" if ratios else ""
        + f"\nCV: {cv:.3f}\n\n"
        + f"{'UNIVERSAL' if cv < 0.15 else 'NOT UNIVERSAL'}"
    )
    axes[2].text(0.5, 0.5, summary, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].axis('off')
    axes[2].set_title('(c) Summary')

    fig.suptitle('Phase 132: Architecture Universality', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase132_universality')
    plt.close()

    print(f"\n{'='*70}")
    for n, r in results.items():
        print(f"  {n}: L0/L={r['L0_ratio']:.3f} (R2={r['r2']:.3f})")
    if ratios:
        print(f"  CV={cv:.3f}")
    print(f"{'='*70}")

    save_results('phase132_universality', {
        'experiment': 'Architecture Universality',
        'results': results,
        'summary': {
            'models_tested': len(results),
            'L0_ratios': {n: r['L0_ratio'] for n, r in results.items()},
            'cv': float(cv) if ratios else None,
        }
    })


if __name__ == '__main__':
    main()
