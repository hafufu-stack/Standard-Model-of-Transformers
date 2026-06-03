# -*- coding: utf-8 -*-
"""
Phase 12: Universal Migration Map (Dimension Scaling Law)
==========================================================
Test how Grammar Police position and PR*T conservation scale
across model sizes: Qwen2.5-0.5B vs 1.5B.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import save_results, save_figure
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


def load_model_by_name(model_name, device='cuda'):
    """Load a specific model by name."""
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=True, dtype=torch.float16,
        device_map=device if device == 'cpu' else None,
    )
    if device != 'cpu':
        model = model.to(device)
    model.eval()
    return model, tok


def main():
    print("=" * 70)
    print("Phase 12: Universal Migration Map (Dimension Scaling)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Models to compare (must be cached locally)
    model_specs = [
        ("Qwen/Qwen2.5-0.5B", "0.5B"),
        ("Qwen/Qwen2.5-1.5B", "1.5B"),
    ]

    prompts = [
        "The meaning of life is",
        "2 + 3 =",
        "The capital of France is",
        "Water freezes at",
        "In quantum mechanics,",
    ]

    noise_sigma = 0.3
    all_model_results = {}

    for model_name, label in model_specs:
        print(f"\n{'='*50}")
        print(f"Model: {label} ({model_name})")
        print(f"{'='*50}")

        try:
            model, tok = load_model_by_name(model_name, device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        print(f"  Layers={n_layers}, d={hidden_size}")

        # 1. Conservation PR*T per layer
        prt_per_layer = []
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            for hs in out.hidden_states[1:]:
                h = hs[0, -1, :].float()
                T = h.norm().item()
                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                PR = 1.0 / (h_prob ** 2).sum().item()
                prt_per_layer.append(PR * T)

        mean_prt = np.mean(prt_per_layer)
        cv_prt = np.std(prt_per_layer) / (mean_prt + 1e-10) * 100
        print(f"  PR*T = {mean_prt:.2f} +/- {np.std(prt_per_layer):.2f} (CV={cv_prt:.1f}%)")

        # 2. Surgical noise sweep (Grammar Police migration)
        noise_impact = []
        for target_layer in range(n_layers):
            impact_scores = []
            for prompt in prompts:
                inp = tok(prompt, return_tensors='pt').to(device)

                with torch.no_grad():
                    base_logits = model(**inp).logits[0, -1, :]

                def make_noise_hook(sigma):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                            noise = torch.randn(h.shape, dtype=torch.float32, device=h.device) * sigma
                            return (h + noise.to(h.dtype),) + output[1:]
                        return output
                    return hook

                handle = model.model.layers[target_layer].register_forward_hook(
                    make_noise_hook(noise_sigma))

                with torch.no_grad():
                    noisy_logits = model(**inp).logits[0, -1, :]

                handle.remove()

                base_p = torch.softmax(base_logits.float(), dim=-1)
                noisy_p = torch.softmax(noisy_logits.float(), dim=-1)
                kl = (base_p * (torch.log(base_p + 1e-10) - torch.log(noisy_p + 1e-10))).sum().item()
                impact_scores.append(max(0, kl))

            avg_impact = np.mean(impact_scores)
            noise_impact.append(avg_impact)

        # Find Grammar Police layers (highest impact)
        gp_threshold = np.percentile(noise_impact, 75)
        gp_layers = [i for i, v in enumerate(noise_impact) if v >= gp_threshold]
        gp_relative = [l / n_layers for l in gp_layers]
        print(f"  Grammar Police layers: {gp_layers}")
        print(f"  Relative positions: {[f'{p:.2f}' for p in gp_relative]}")

        all_model_results[label] = {
            'n_layers': n_layers, 'hidden_size': hidden_size,
            'mean_prt': mean_prt, 'cv_prt': cv_prt,
            'noise_impact': noise_impact,
            'gp_layers': gp_layers, 'gp_relative': gp_relative,
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Visualization
    n_models = len(all_model_results)
    if n_models == 0:
        print("No models loaded. Exiting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Noise impact profile for each model (normalized x-axis)
    ax = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for idx, (label, data) in enumerate(all_model_results.items()):
        x_norm = np.linspace(0, 1, len(data['noise_impact']))
        ax.plot(x_norm, data['noise_impact'], 'o-', ms=3, color=colors[idx % 3],
                label=f"{label} ({data['n_layers']}L)")
    ax.set_xlabel('Relative Layer Position')
    ax.set_ylabel('Noise Impact (KL div)')
    ax.set_title('(a) Grammar Police Migration')
    ax.legend()

    # (b) PR*T vs hidden_size
    ax = axes[1]
    sizes = [d['hidden_size'] for d in all_model_results.values()]
    prts = [d['mean_prt'] for d in all_model_results.values()]
    labels_list = list(all_model_results.keys())
    ax.scatter(sizes, prts, s=150, c=colors[:n_models], edgecolors='black', zorder=5)
    for i, lbl in enumerate(labels_list):
        ax.annotate(lbl, (sizes[i], prts[i]), fontsize=11, ha='center', va='bottom')
    ax.set_xlabel('Hidden Dimension d')
    ax.set_ylabel('Mean PR*T')
    ax.set_title('(b) Conservation vs Model Size')

    # (c) GP relative position vs model size
    ax = axes[2]
    for idx, (label, data) in enumerate(all_model_results.items()):
        gp_rel = data['gp_relative']
        ax.scatter([data['n_layers']] * len(gp_rel), gp_rel,
                   s=80, c=colors[idx % 3], alpha=0.7, label=label)
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Relative GP Position')
    ax.set_title('(c) Grammar Police Position Scaling')
    ax.legend()

    fig.suptitle(
        "Phase 12: Universal Migration Map\n"
        "How do constants and functions scale with model dimension?",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase12_migration_map")
    plt.close()

    # Verdict
    if n_models >= 2:
        prt_ratio = prts[-1] / (prts[0] + 1e-10)
        d_ratio = sizes[-1] / sizes[0]
        verdict = (f"SCALING LAW: PR*T scales {prt_ratio:.2f}x when d scales {d_ratio:.1f}x. "
                   f"GP layers migrate to relative positions "
                   f"{[f'{p:.2f}' for p in all_model_results[labels_list[-1]]['gp_relative']]}.")
    else:
        verdict = f"SINGLE MODEL: PR*T={prts[0]:.0f} for {labels_list[0]}. Need more models for scaling."

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 12: Universal Migration Map',
        'summary': {'verdict': verdict},
        'models': {k: {'n_layers': v['n_layers'], 'hidden_size': v['hidden_size'],
                       'mean_prt': v['mean_prt'], 'gp_layers': v['gp_layers']}
                   for k, v in all_model_results.items()},
    }
    save_results("phase12_migration_map", result)
    return result


if __name__ == '__main__':
    main()
