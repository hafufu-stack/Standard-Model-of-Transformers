# -*- coding: utf-8 -*-
"""
Phase 48: Boltzmann Universality
Test Phase 42's R2=0.984 result on other architectures (Llama, OPT).
If Boltzmann holds across architectures, it's a true universal law.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure


def boltzmann_pdf(E, A, kT):
    return A * np.exp(-E / (kT + 1e-10))


CANDIDATE_MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
    ("facebook/opt-1.3b", "OPT-1.3B"),
]

TEST_PROMPTS = [
    "The fundamental laws of physics describe how the universe operates at every scale from",
    "Machine learning algorithms can learn patterns from data without being explicitly",
    "The human brain processes information through billions of interconnected neurons that",
]


def analyze_boltzmann(model, tok, device, model_name):
    """Test Boltzmann distribution for a single model."""
    print(f"\n  Testing {model_name}...")

    # Find layers and MLP
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        get_mlp = lambda l: l.mlp
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        layers = model.model.decoder.layers
        get_mlp = lambda l: l.fc1 if hasattr(l, 'fc1') else l.mlp if hasattr(l, 'mlp') else None
    else:
        print(f"  Cannot identify layers for {model_name}")
        return None

    n_layers = len(layers)
    layer_r2s = []

    for prompt in TEST_PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Capture FFN activations
        ffn_acts = {}
        def make_hook(li):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                ffn_acts[li] = h[0, -1, :].detach().cpu().float().numpy()
            return hook

        hooks = []
        for li in range(n_layers):
            mlp = get_mlp(layers[li])
            if mlp is not None:
                h = mlp.register_forward_hook(make_hook(li))
                hooks.append(h)

        with torch.no_grad():
            model(**inp)

        for h in hooks:
            h.remove()

        for li in ffn_acts:
            act = ffn_acts[li]
            energies = act ** 2
            nonzero_E = energies[energies > 1e-8]
            if len(nonzero_E) < 50:
                continue

            hist, edges = np.histogram(nonzero_E, bins=50, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            mask = hist > 0
            bc = centers[mask]
            hv = hist[mask]
            if len(bc) < 10:
                continue

            try:
                popt, _ = curve_fit(boltzmann_pdf, bc, hv,
                                    p0=[hv[0], np.mean(nonzero_E)],
                                    maxfev=5000, bounds=([0, 1e-8], [np.inf, np.inf]))
                residuals = hv - boltzmann_pdf(bc, *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((hv - np.mean(hv)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                layer_r2s.append({'layer': li, 'r2': float(r2), 'kT': float(popt[1])})
            except Exception:
                pass

    if not layer_r2s:
        return None

    mean_r2 = np.mean([r['r2'] for r in layer_r2s])
    high_r2_pct = sum(1 for r in layer_r2s if r['r2'] > 0.8) / len(layer_r2s) * 100
    mean_kT = np.mean([r['kT'] for r in layer_r2s])

    print(f"  {model_name}: Mean R2={mean_r2:.4f}, {high_r2_pct:.0f}% above 0.8, mean_kT={mean_kT:.4f}")

    return {
        'model_name': model_name,
        'n_layers': n_layers,
        'mean_r2': float(mean_r2),
        'high_r2_pct': float(high_r2_pct),
        'mean_kT': float(mean_kT),
        'layer_details': layer_r2s,
    }


def main():
    print("=" * 70)
    print("Phase 48: Boltzmann Universality (Multi-Architecture)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    for model_id, model_name in CANDIDATE_MODELS:
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

        try:
            result = analyze_boltzmann(model, tok, device, model_name)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error: {str(e)[:80]}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    if len(all_results) < 2:
        print("\nWARNING: Need at least 2 models")
        save_results('phase48_boltzmann_universal', {
            'experiment': 'Boltzmann Universality',
            'summary': {'verdict': 'INSUFFICIENT MODELS'},
        })
        return

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    n = len(all_results)
    names = [r['model_name'] for r in all_results]
    x = np.arange(n)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:n]

    # (a) Mean R2
    r2s = [r['mean_r2'] for r in all_results]
    axes[0].bar(x, r2s, color=colors, alpha=0.8)
    axes[0].axhline(y=0.9, color='gray', linestyle='--', label='R2=0.9')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean R-squared')
    axes[0].set_title('(a) Boltzmann Fit Quality')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    # (b) % layers above R2>0.8
    pcts = [r['high_r2_pct'] for r in all_results]
    axes[1].bar(x, pcts, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('% Layers with R2 > 0.8')
    axes[1].set_title('(b) Consistency Across Layers')
    for i, v in enumerate(pcts):
        axes[1].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)

    # (c) Mean kT
    kTs = [r['mean_kT'] for r in all_results]
    axes[2].bar(x, kTs, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Mean Micro kT')
    axes[2].set_title('(c) Micro Temperature')

    r2_cv = np.std(r2s) / (np.mean(r2s) + 1e-10)
    fig.suptitle(f'Phase 48: Boltzmann Universality ({n} models, R2 CV={r2_cv:.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase48_boltzmann_universal')
    plt.close()

    # Verdict
    all_high = all(r['mean_r2'] > 0.8 for r in all_results)

    print(f"\n{'='*70}")
    print(f"VERDICT: R2 across {n} architectures: "
          f"{', '.join(f'{r['model_name']}={r['mean_r2']:.3f}' for r in all_results)}. "
          f"CV={r2_cv:.3f}. "
          f"Boltzmann distribution is {'UNIVERSAL' if all_high else 'NOT universal'}.")
    print(f"{'='*70}")

    save_results('phase48_boltzmann_universal', {
        'experiment': 'Boltzmann Universality',
        'results': all_results,
        'summary': {
            'n_models': n,
            'all_above_0.8': all_high,
            'r2_cv': r2_cv,
            'mean_r2': float(np.mean(r2s)),
        }
    })


if __name__ == '__main__':
    main()
