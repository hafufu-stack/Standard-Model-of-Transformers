# -*- coding: utf-8 -*-
"""
Phase 153: Carnot Limit of Alignment
Compare base model vs instruction-tuned model to see if
alignment reduces thermodynamic efficiency.
Use Qwen2.5-1.5B (base) vs Qwen2.5-1.5B-Instruct.
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


def analyze_model(model, tok, prompts, device, model_name, norm_fn, head_fn):
    """Compute full thermodynamic profile."""
    n_layers = len(list(model.model.layers)) + 1

    all_S = [[] for _ in range(n_layers)]
    all_eta = [[] for _ in range(n_layers)]

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(min(n_layers, len(out.hidden_states))):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = norm_fn(hs[:, -1:, :])
                logits = head_fn(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_vals.append(S if not np.isnan(S) else 0)
            all_S[li].append(T_vals[-1])

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
    avg_S = [np.mean(v) if v else 0 for v in all_S]

    # Fit sigmoid
    try:
        Ls = np.arange(4, n_layers)
        popt, _ = curve_fit(sigmoid, Ls, avg_eta[4:],
                            p0=[22, 0.5, 0, 0.9], maxfev=10000)
        L0 = popt[0]
        sig_pred = sigmoid(Ls, *popt)
        r2 = 1 - np.sum((np.array(avg_eta[4:]) - sig_pred)**2) / (
            np.sum((np.array(avg_eta[4:]) - np.mean(avg_eta[4:]))**2) + 1e-10)
    except:
        L0 = 22
        r2 = 0

    return {
        'name': model_name,
        'eta': avg_eta,
        'S': avg_S,
        'L0': float(L0),
        'R2': float(r2),
        'final_eta': float(avg_eta[-1]),
        'final_S': float(avg_S[-1]),
        'n_layers': n_layers,
    }


def main():
    print("=" * 70)
    print("Phase 153: Carnot Limit of Alignment")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    # Model 1: Base
    print("\n--- Qwen2.5-1.5B (Base) ---")
    try:
        model_b = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16,
            device_map=device, local_files_only=True)
        tok_b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
        r = analyze_model(model_b, tok_b, PROMPTS, device, 'Base',
                         model_b.model.norm, model_b.lm_head)
        results['base'] = r
        print(f"  L0={r['L0']:.1f}, R2={r['R2']:.3f}, eta_final={r['final_eta']:.3f}")
        del model_b
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Model 2: Instruct
    print("\n--- Qwen2.5-1.5B-Instruct ---")
    try:
        model_i = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16,
            device_map=device, local_files_only=True)
        tok_i = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
        r = analyze_model(model_i, tok_i, PROMPTS, device, 'Instruct',
                         model_i.model.norm, model_i.lm_head)
        results['instruct'] = r
        print(f"  L0={r['L0']:.1f}, R2={r['R2']:.3f}, eta_final={r['final_eta']:.3f}")
        del model_i
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

    if len(results) < 2:
        print("WARNING: Need both models for comparison")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'base': '#2980b9', 'instruct': '#c0392b'}

    # (a) Eta profiles
    for key, r in results.items():
        axes[0,0].plot(range(r['n_layers']), r['eta'], 'o-', color=colors.get(key, 'gray'),
                      markersize=3, linewidth=2, label=f"{r['name']} (L0={r['L0']:.1f})")
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Alignment Effect on Eta')
    axes[0,0].legend()

    # (b) S profiles
    for key, r in results.items():
        axes[0,1].plot(range(r['n_layers']), r['S'], 'o-', color=colors.get(key, 'gray'),
                      markersize=3, linewidth=2, label=r['name'])
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('$S$')
    axes[0,1].set_title('(b) Entropy Profiles')
    axes[0,1].legend()

    # (c) L0 comparison
    if len(results) == 2:
        keys = list(results.keys())
        L0s = [results[k]['L0'] for k in keys]
        r2s = [results[k]['R2'] for k in keys]
        bar_c = [colors.get(k, 'gray') for k in keys]
        axes[0,2].bar(range(len(keys)), L0s, color=bar_c, alpha=0.8, edgecolor='black')
        axes[0,2].set_xticks(range(len(keys)))
        axes[0,2].set_xticklabels([results[k]['name'] for k in keys])
        axes[0,2].set_ylabel('$L_0$')
        axes[0,2].set_title('(c) Critical Point Shift')

    # (d) Final eta comparison
    if len(results) == 2:
        etas_f = [results[k]['final_eta'] for k in keys]
        axes[1,0].bar(range(len(keys)), etas_f, color=bar_c, alpha=0.8, edgecolor='black')
        axes[1,0].set_xticks(range(len(keys)))
        axes[1,0].set_xticklabels([results[k]['name'] for k in keys])
        axes[1,0].set_ylabel('$\\eta_{final}$')
        axes[1,0].set_title('(d) Final Efficiency')

    # (e) Final S comparison
    if len(results) == 2:
        Ss_f = [results[k]['final_S'] for k in keys]
        axes[1,1].bar(range(len(keys)), Ss_f, color=bar_c, alpha=0.8, edgecolor='black')
        axes[1,1].set_xticks(range(len(keys)))
        axes[1,1].set_xticklabels([results[k]['name'] for k in keys])
        axes[1,1].set_ylabel('$S_{final}$')
        axes[1,1].set_title('(e) Final Entropy')

    # (f) Summary
    if len(results) == 2:
        base_r = results['base']
        inst_r = results['instruct']
        eta_tax = (base_r['final_eta'] - inst_r['final_eta']) / base_r['final_eta'] * 100
        summary = (
            f"Alignment Tax Analysis\n\n"
            f"Base: L0={base_r['L0']:.1f}, eta={base_r['final_eta']:.3f}\n"
            f"Instruct: L0={inst_r['L0']:.1f}, eta={inst_r['final_eta']:.3f}\n\n"
            f"Eta tax: {eta_tax:+.1f}%\n"
            f"L0 shift: {inst_r['L0']-base_r['L0']:+.1f}\n\n"
            f"Alignment {'REDUCES' if eta_tax > 0 else 'INCREASES'}\n"
            f"thermodynamic efficiency"
        )
    else:
        summary = "Insufficient models for comparison"
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 153: Carnot Limit of Alignment',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase153_alignment')
    plt.close()

    print(f"\n{'='*70}")
    for k, r in results.items():
        print(f"  {r['name']}: L0={r['L0']:.1f}, eta_final={r['final_eta']:.3f}")
    print(f"{'='*70}")

    save_results('phase153_alignment', {
        'experiment': 'Carnot Limit of Alignment',
        'results': {k: {kk: vv for kk, vv in v.items() if kk not in ['eta', 'S']}
                    for k, v in results.items()},
    })


if __name__ == '__main__':
    main()
