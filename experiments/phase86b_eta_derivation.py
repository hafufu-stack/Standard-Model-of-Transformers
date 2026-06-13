# -*- coding: utf-8 -*-
"""
Phase 86b: First Principles Derivation of eta
Fixed version: use T_cold from later half of layers (matching Phase 75).
Investigate whether Carnot efficiency depends on architecture hyperparameters.
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

# Use same prompts as Phase 75 for consistency
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
    "Protein folding determines biological function",
    "Cryptographic hash functions ensure data integrity",
    "The Turing test measures machine intelligence",
    "Semiconductors enable modern computing devices",
]


def get_architecture_params(model, model_name):
    """Extract key architecture hyperparameters."""
    config = model.config
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    d_ffn = getattr(config, 'intermediate_size', d_model * 4)
    vocab_size = config.vocab_size

    return {
        'model_name': model_name,
        'd_model': d_model,
        'd_ffn': d_ffn,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'n_layers': n_layers,
        'expansion_ratio': round(d_ffn / d_model, 3),
        'd_head': d_model // n_heads,
        'vocab_size': vocab_size,
        'gqa_ratio': round(n_heads / n_kv_heads, 2),
        'total_params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }


def measure_eta_fixed(model, tok, device):
    """Measure Carnot efficiency matching Phase 75 methodology."""
    etas = []
    T_hot_list = []
    T_cold_list = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li, hs in enumerate(out.hidden_states):
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if not np.isnan(T):
                T_vals.append(T)

        if len(T_vals) >= 2:
            T_hot = max(T_vals)
            # Phase 75 definition: T_cold = min of LATER HALF layers
            T_cold = min(T_vals[len(T_vals)//2:])
            if T_hot > 0.01:
                eta = 1.0 - T_cold / T_hot
                etas.append(eta)
                T_hot_list.append(T_hot)
                T_cold_list.append(T_cold)

    return etas, T_hot_list, T_cold_list


def main():
    print("=" * 70)
    print("Phase 86b: First Principles eta Derivation (Fixed)")
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
            print(f"  Failed: {str(e)[:80]}")
            continue

        params = get_architecture_params(model, model_name)
        etas, T_hots, T_colds = measure_eta_fixed(model, tok, device)

        if etas:
            params['eta_mean'] = float(np.mean(etas))
            params['eta_std'] = float(np.std(etas))
            params['eta_values'] = [float(e) for e in etas]
            params['T_hot_mean'] = float(np.mean(T_hots))
            params['T_cold_mean'] = float(np.mean(T_colds))
            print(f"  {model_name}: eta = {np.mean(etas):.4f} +/- {np.std(etas):.4f}")
            print(f"    T_hot={np.mean(T_hots):.3f}, T_cold={np.mean(T_colds):.3f}")
            print(f"    d_model={params['d_model']}, d_ffn={params['d_ffn']}, "
                  f"expansion={params['expansion_ratio']}, heads={params['n_heads']}, "
                  f"kv_heads={params['n_kv_heads']}, GQA={params['gqa_ratio']}")
        all_results.append(params)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    valid = [r for r in all_results if 'eta_mean' in r]
    if len(valid) < 2:
        print("Not enough models")
        save_results('phase86b_eta_derivation', {'experiment': 'eta derivation (fixed)', 'error': 'insufficient models'})
        return

    etas = [r['eta_mean'] for r in valid]
    names = [r['model_name'] for r in valid]

    # === Correlations ===
    correlations = {}
    for param_name in ['expansion_ratio', 'd_model', 'n_heads', 'n_layers',
                       'gqa_ratio', 'd_head', 'total_params_M']:
        vals = [r[param_name] for r in valid]
        if len(set(vals)) > 1:
            r, p = sp_stats.pearsonr(vals, etas)
            correlations[param_name] = {'r': float(r), 'p': float(p)}
            print(f"  Corr(eta, {param_name}): r={r:.3f}, p={p:.3f}")

    # === Hypotheses ===
    # H1: eta = 1 - 1/expansion_ratio
    h1_pred = [1.0 - 1.0/r['expansion_ratio'] for r in valid]
    # H2: eta = 1 - d_model/d_ffn
    h2_pred = [1.0 - r['d_model']/r['d_ffn'] for r in valid]
    # H3: eta = 1 - n_kv_heads/n_heads (GQA ratio)
    h3_pred = [1.0 - r['n_kv_heads']/r['n_heads'] for r in valid]
    # H4: eta = 1 - 1/sqrt(n_layers)
    h4_pred = [1.0 - 1.0/np.sqrt(r['n_layers']) for r in valid]

    hypotheses = {
        'H1: 1-1/r_exp': h1_pred,
        'H2: 1-d/d_ffn': h2_pred,
        'H3: 1-kv/heads': h3_pred,
        'H4: 1-1/sqrt(L)': h4_pred,
    }

    print("\n  Hypothesis comparison:")
    best_h = None
    best_mae = 999
    for hname, preds in hypotheses.items():
        mae = np.mean([abs(p-a) for p,a in zip(preds, etas)])
        print(f"    {hname}: MAE = {mae:.4f}")
        for r, p, a in zip(valid, preds, etas):
            print(f"      {r['model_name']}: pred={p:.4f}, actual={a:.4f}")
        if mae < best_mae:
            best_mae = mae
            best_h = hname

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = ['#c0392b', '#2980b9', '#27ae60']

    # (a) eta by model
    axes[0,0].bar(range(len(valid)), etas, yerr=[r['eta_std'] for r in valid],
                  color=colors[:len(valid)], alpha=0.8, edgecolor='black', capsize=5)
    axes[0,0].set_xticks(range(len(valid)))
    axes[0,0].set_xticklabels(names, fontsize=9)
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Carnot Efficiency (fixed)')
    mean_eta = np.mean(etas)
    axes[0,0].axhline(y=mean_eta, color='gray', linestyle='--', label=f'Mean={mean_eta:.3f}')
    axes[0,0].legend()

    # (b) eta vs expansion ratio
    exps = [r['expansion_ratio'] for r in valid]
    axes[0,1].scatter(exps, etas, s=120, c=colors[:len(valid)], edgecolors='black', zorder=5)
    for i, r in enumerate(valid):
        axes[0,1].annotate(r['model_name'], (exps[i], etas[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
    x_line = np.linspace(min(exps)-0.5, max(exps)+0.5, 50)
    axes[0,1].plot(x_line, [1-1/x for x in x_line], '--', color='gray', alpha=0.5, label='H1: $1-1/r$')
    axes[0,1].set_xlabel('FFN Expansion Ratio')
    axes[0,1].set_ylabel('$\\eta$')
    axes[0,1].set_title('(b) $\\eta$ vs Expansion')
    axes[0,1].legend(fontsize=8)

    # (c) Hypothesis comparison (MAE)
    h_names = list(hypotheses.keys())
    h_maes = [np.mean([abs(p-a) for p,a in zip(hypotheses[h], etas)]) for h in h_names]
    bar_colors = ['#27ae60' if h == best_h else '#bdc3c7' for h in h_names]
    axes[0,2].bar(range(len(h_names)), h_maes, color=bar_colors, edgecolor='black')
    axes[0,2].set_xticks(range(len(h_names)))
    axes[0,2].set_xticklabels(h_names, fontsize=8, rotation=15)
    axes[0,2].set_ylabel('MAE')
    axes[0,2].set_title(f'(c) Best Hypothesis: {best_h}')

    # (d) Predicted vs Actual for best hypothesis
    best_preds = hypotheses[best_h]
    axes[1,0].scatter(best_preds, etas, s=120, c=colors[:len(valid)], edgecolors='black', zorder=5)
    mn = min(min(best_preds), min(etas)) - 0.05
    mx = max(max(best_preds), max(etas)) + 0.05
    axes[1,0].plot([mn,mx], [mn,mx], 'k--', alpha=0.3, label='Perfect')
    for i, r in enumerate(valid):
        axes[1,0].annotate(r['model_name'], (best_preds[i], etas[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1,0].set_xlabel(f'Predicted ({best_h})')
    axes[1,0].set_ylabel('Measured $\\eta$')
    axes[1,0].set_title('(d) Best Prediction')
    axes[1,0].legend()

    # (e) T_hot and T_cold comparison
    T_hots = [r['T_hot_mean'] for r in valid]
    T_colds = [r['T_cold_mean'] for r in valid]
    x = np.arange(len(valid))
    w = 0.35
    axes[1,1].bar(x - w/2, T_hots, w, color='#c0392b', alpha=0.7, label='$T_{hot}$')
    axes[1,1].bar(x + w/2, T_colds, w, color='#2980b9', alpha=0.7, label='$T_{cold}$')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(names, fontsize=9)
    axes[1,1].set_ylabel('Temperature')
    axes[1,1].set_title('(e) $T_{hot}$ vs $T_{cold}$')
    axes[1,1].legend()

    # (f) Correlation heatmap
    param_names = [k for k in correlations.keys()]
    r_vals = [correlations[k]['r'] for k in param_names]
    p_vals = [correlations[k]['p'] for k in param_names]
    color_bars = ['#27ae60' if abs(r) > 0.9 else '#f39c12' if abs(r) > 0.5 else '#bdc3c7' for r in r_vals]
    axes[1,2].barh(range(len(param_names)), r_vals, color=color_bars, edgecolor='black')
    axes[1,2].set_yticks(range(len(param_names)))
    axes[1,2].set_yticklabels(param_names, fontsize=8)
    axes[1,2].set_xlabel('Pearson r')
    axes[1,2].set_title('(f) Correlations with $\\eta$')
    axes[1,2].axvline(x=0, color='black', linewidth=0.5)

    fig.suptitle(f'Phase 86b: First Principles $\\eta$ (mean={mean_eta:.3f}, best: {best_h})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase86b_eta_derivation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Mean eta = {mean_eta:.4f} (Phase 75 ref: 0.813)")
    print(f"Best hypothesis: {best_h} (MAE={best_mae:.4f})")
    print(f"{'='*70}")

    save_results('phase86b_eta_derivation', {
        'experiment': 'First Principles eta (Fixed)',
        'results': all_results,
        'correlations': correlations,
        'hypotheses': {h: {'predictions': [float(p) for p in preds],
                          'mae': float(np.mean([abs(p-a) for p,a in zip(preds, etas)]))}
                      for h, preds in hypotheses.items()},
        'summary': {
            'mean_eta': float(mean_eta),
            'best_hypothesis': best_h,
            'best_mae': float(best_mae),
        }
    })


if __name__ == '__main__':
    main()
