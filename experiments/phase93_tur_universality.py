# -*- coding: utf-8 -*-
"""
Phase 93: TUR Universality
Phase 92 showed TUR holds 10/10 on Qwen 1.5B with mean ratio 12.5.
Test on all 3 architectures to confirm as the 6th Universal Law.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
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
]


def measure_tur(model, tok, device, model_name):
    """Measure TUR for a single model."""
    results = []
    layers = model.model.layers if hasattr(model.model, 'layers') else model.model.decoder.layers

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # FFN norms
        ffn_norms = []
        hooks = []

        def make_hook(storage):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                storage.append(h[0, -1, :].detach().float().norm().item())
            return hook

        for layer in layers:
            mlp = layer.mlp if hasattr(layer, 'mlp') else (layer.fc1 if hasattr(layer, 'fc1') else None)
            if mlp:
                hooks.append(mlp.register_forward_hook(make_hook(ffn_norms)))

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in hooks:
            h.remove()

        hs_list = [out.hidden_states[li][0, -1, :].cpu().float()
                   for li in range(len(out.hidden_states))]

        # J = information current (1 - cos_sim between adjacent layers)
        J_vals = []
        for i in range(1, len(hs_list)):
            cos = torch.nn.functional.cosine_similarity(
                hs_list[i].unsqueeze(0), hs_list[i-1].unsqueeze(0)).item()
            J_vals.append(1.0 - cos)

        # F = driving force
        F_vals = ffn_norms[:len(J_vals)]

        # kT at each layer
        kT_vals = []
        for li in range(1, len(out.hidden_states)):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0.0
            kT_vals.append(T)

        if J_vals and F_vals:
            J = np.array(J_vals)
            F = np.array(F_vals[:len(J)])
            kT = np.array(kT_vals[:len(J)])

            var_J = np.var(J)
            var_F = np.var(F)
            mean_kT = np.mean(kT)
            product = var_J * var_F
            bound = 2 * mean_kT

            results.append({
                'var_J': float(var_J),
                'var_F': float(var_F),
                'product': float(product),
                'bound': float(bound),
                'satisfied': bool(product >= bound),
                'ratio': float(product / (bound + 1e-10)),
            })

    return results


def main():
    print("=" * 70)
    print("Phase 93: TUR Universality (3 Architectures)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_model_results = {}

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

        results = measure_tur(model, tok, device, model_name)
        n_sat = sum(1 for r in results if r['satisfied'])
        mean_ratio = np.mean([r['ratio'] for r in results])
        print(f"  TUR: {n_sat}/{len(results)} satisfied, mean ratio = {mean_ratio:.2f}")

        all_model_results[model_name] = {
            'results': results,
            'n_satisfied': n_sat,
            'n_total': len(results),
            'satisfaction_rate': float(n_sat / len(results)) if results else 0,
            'mean_ratio': float(mean_ratio),
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = list(all_model_results.keys())
    colors = ['#c0392b', '#2980b9', '#27ae60'][:len(model_names)]

    # (a) Satisfaction rate
    rates = [all_model_results[m]['satisfaction_rate'] for m in model_names]
    axes[0].bar(range(len(model_names)), rates, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, fontsize=9)
    axes[0].set_ylabel('TUR Satisfaction Rate')
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=1.0, color='gray', linestyle='--')
    axes[0].set_title('(a) TUR Satisfaction Rate')
    for i, r in enumerate(rates):
        axes[0].text(i, r + 0.02, f'{r:.0%}', ha='center', fontsize=11, fontweight='bold')

    # (b) Mean ratio
    mean_ratios = [all_model_results[m]['mean_ratio'] for m in model_names]
    axes[1].bar(range(len(model_names)), mean_ratios, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, fontsize=9)
    axes[1].set_ylabel('Mean TUR Ratio')
    axes[1].axhline(y=1.0, color='#c0392b', linestyle='--', label='TUR bound (1.0)')
    axes[1].set_title('(b) TUR Ratio (bound = 1.0)')
    axes[1].legend()

    # (c) Product vs Bound scatter
    for idx, m in enumerate(model_names):
        products = [r['product'] for r in all_model_results[m]['results']]
        bounds = [r['bound'] for r in all_model_results[m]['results']]
        axes[2].scatter(bounds, products, s=60, c=colors[idx], alpha=0.7,
                       edgecolors='black', label=m, zorder=5)
    max_val = max(max(r['product'] for m in model_names for r in all_model_results[m]['results']),
                  max(r['bound'] for m in model_names for r in all_model_results[m]['results'])) * 1.2
    axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='TUR boundary')
    axes[2].set_xlabel('$2kT$')
    axes[2].set_ylabel('$\\sigma^2(J) \\cdot \\sigma^2(F)$')
    axes[2].set_title('(c) Product vs Bound')
    axes[2].legend(fontsize=7)

    overall_rate = np.mean(rates)
    overall_ratio = np.mean(mean_ratios)
    cv = np.std(mean_ratios) / (np.mean(mean_ratios) + 1e-10)
    is_universal = all(r >= 0.8 for r in rates) and cv < 0.5

    fig.suptitle(f'Phase 93: TUR Universality (rate={overall_rate:.0%}, '
                 f'ratio={overall_ratio:.1f}, {"UNIVERSAL" if is_universal else "PARTIAL"})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase93_tur_universality')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Overall TUR satisfaction: {overall_rate:.0%}")
    print(f"Mean ratio across models: {overall_ratio:.2f}")
    print(f"Cross-model CV: {cv:.3f}")
    print(f"Verdict: TUR {'IS' if is_universal else 'IS NOT'} a universal law")
    print(f"{'='*70}")

    save_results('phase93_tur_universality', {
        'experiment': 'TUR Universality',
        'per_model': all_model_results,
        'summary': {
            'overall_rate': float(overall_rate),
            'overall_ratio': float(overall_ratio),
            'cv': float(cv),
            'is_universal': is_universal,
        }
    })


if __name__ == '__main__':
    main()
