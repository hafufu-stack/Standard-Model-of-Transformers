# -*- coding: utf-8 -*-
"""
Phase 86: First Principles Derivation of eta = 0.813
Investigate whether Carnot efficiency depends on architecture hyperparameters
(FFN expansion ratio, n_heads, d_model, n_layers).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
]

PROMPTS = [
    "The fundamental theorem of calculus connects differentiation and",
    "Quantum mechanics describes particles at the atomic scale",
    "The human genome contains three billion base pairs encoding",
    "Neural networks learn through layers of interconnected nodes",
    "Black holes form from gravitational collapse of massive stars",
    "The periodic table organizes chemical elements by number",
    "Evolution by natural selection operates on heritable variation",
    "Climate change affects ecosystems through rising temperatures",
]


def get_architecture_params(model, model_name):
    """Extract key architecture hyperparameters."""
    config = model.config
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_layers = config.num_hidden_layers

    # FFN intermediate size
    if hasattr(config, 'intermediate_size'):
        d_ffn = config.intermediate_size
    else:
        d_ffn = d_model * 4  # default assumption

    expansion_ratio = d_ffn / d_model
    return {
        'model_name': model_name,
        'd_model': d_model,
        'd_ffn': d_ffn,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'expansion_ratio': round(expansion_ratio, 3),
        'd_head': d_model // n_heads,
    }


def measure_eta(model, tok, device):
    """Measure Carnot efficiency eta = 1 - T_cold/T_hot for a model."""
    etas = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # T at each layer
        Ts = []
        for li, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(T) and T > 0:
                Ts.append(T)

        if len(Ts) >= 2:
            T_hot = max(Ts)
            T_cold = min(Ts)
            if T_hot > 0:
                eta = 1.0 - T_cold / T_hot
                etas.append(eta)

    return etas


def main():
    print("=" * 70)
    print("Phase 86: First Principles eta Derivation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    for model_id, model_name in MODELS:
        print(f"\n--- Loading {model_name} ---")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map=device,
                trust_remote_code=True,
            )
            model.eval()
        except Exception as e:
            print(f"  Failed to load: {str(e)[:80]}")
            continue

        params = get_architecture_params(model, model_name)
        etas = measure_eta(model, tok, device)

        if etas:
            params['eta_mean'] = float(np.mean(etas))
            params['eta_std'] = float(np.std(etas))
            params['eta_values'] = [float(e) for e in etas]
            print(f"  {model_name}: eta = {np.mean(etas):.4f} +/- {np.std(etas):.4f}")
            print(f"    d_model={params['d_model']}, d_ffn={params['d_ffn']}, "
                  f"expansion={params['expansion_ratio']}, heads={params['n_heads']}")
        all_results.append(params)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # === Analysis: find correlation between eta and architecture params ===
    valid = [r for r in all_results if 'eta_mean' in r]
    if len(valid) >= 2:
        etas = [r['eta_mean'] for r in valid]
        expansions = [r['expansion_ratio'] for r in valid]
        d_models = [r['d_model'] for r in valid]
        n_heads = [r['n_heads'] for r in valid]
        n_layers = [r['n_layers'] for r in valid]

        # Test correlations
        correlations = {}
        for name, vals in [('expansion_ratio', expansions), ('d_model', d_models),
                           ('n_heads', n_heads), ('n_layers', n_layers)]:
            if len(set(vals)) > 1:
                r, p = sp_stats.pearsonr(vals, etas)
                correlations[name] = {'r': float(r), 'p': float(p)}
                print(f"  Corr(eta, {name}): r={r:.3f}, p={p:.3f}")

        # Test hypothesis: eta = 1 - d_model/d_ffn (attention/FFN ratio)
        attn_ffn_ratios = [r['d_model'] / r['d_ffn'] for r in valid]
        predicted_etas = [1.0 - ratio for ratio in attn_ffn_ratios]
        print(f"\n  Hypothesis: eta = 1 - d_model/d_ffn")
        for r, pred in zip(valid, predicted_etas):
            actual = r['eta_mean']
            print(f"    {r['model_name']}: predicted={pred:.4f}, actual={actual:.4f}, "
                  f"error={abs(pred - actual):.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if len(valid) >= 2:
        names = [r['model_name'] for r in valid]
        etas_m = [r['eta_mean'] for r in valid]
        etas_s = [r['eta_std'] for r in valid]
        colors = ['#c0392b', '#2980b9', '#27ae60'][:len(valid)]

        # (a) eta by model
        axes[0].bar(range(len(valid)), etas_m, yerr=etas_s, color=colors,
                    alpha=0.8, edgecolor='black', capsize=5)
        axes[0].set_xticks(range(len(valid)))
        axes[0].set_xticklabels(names, fontsize=9)
        axes[0].set_ylabel('Carnot Efficiency $\\eta$')
        axes[0].set_title('(a) eta by Architecture')
        mean_eta = np.mean(etas_m)
        axes[0].axhline(y=mean_eta, color='gray', linestyle='--',
                        label=f'Mean = {mean_eta:.3f}')
        axes[0].legend()

        # (b) eta vs expansion ratio
        exp_ratios = [r['expansion_ratio'] for r in valid]
        axes[1].scatter(exp_ratios, etas_m, s=100, c=colors, edgecolors='black', zorder=5)
        for i, r in enumerate(valid):
            axes[1].annotate(r['model_name'], (exp_ratios[i], etas_m[i]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
        # Plot hypothesis line
        x_line = np.linspace(min(exp_ratios) - 0.5, max(exp_ratios) + 0.5, 50)
        y_hyp = [1.0 - 1.0/x for x in x_line]
        axes[1].plot(x_line, y_hyp, '--', color='gray', alpha=0.5,
                     label='$\\eta = 1 - 1/r_{exp}$')
        axes[1].set_xlabel('FFN Expansion Ratio ($d_{FFN}/d_{model}$)')
        axes[1].set_ylabel('$\\eta$')
        axes[1].set_title('(b) eta vs Expansion Ratio')
        axes[1].legend(fontsize=8)

        # (c) Predicted vs actual
        pred = [1.0 - r['d_model'] / r['d_ffn'] for r in valid]
        axes[2].scatter(pred, etas_m, s=100, c=colors, edgecolors='black', zorder=5)
        min_v = min(min(pred), min(etas_m)) - 0.05
        max_v = max(max(pred), max(etas_m)) + 0.05
        axes[2].plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.3, label='Perfect fit')
        for i, r in enumerate(valid):
            axes[2].annotate(r['model_name'], (pred[i], etas_m[i]),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)
        axes[2].set_xlabel('Predicted $\\eta = 1 - d_{model}/d_{FFN}$')
        axes[2].set_ylabel('Measured $\\eta$')
        axes[2].set_title('(c) Predicted vs Measured')
        axes[2].legend(fontsize=8)

    fig.suptitle('Phase 86: First Principles Derivation of $\\eta$',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase86_eta_derivation')
    plt.close()

    save_results('phase86_eta_derivation', {
        'experiment': 'First Principles eta Derivation',
        'results': all_results,
        'correlations': correlations if len(valid) >= 2 else {},
        'hypothesis_1_d_model_d_ffn': {
            'formula': 'eta = 1 - d_model/d_ffn',
            'predictions': [{'model': r['model_name'],
                            'predicted': float(1.0 - r['d_model']/r['d_ffn']),
                            'actual': r.get('eta_mean', None)}
                           for r in valid]
        } if len(valid) >= 2 else {},
    })

    print(f"\n{'='*70}")
    print("Phase 86 complete")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
