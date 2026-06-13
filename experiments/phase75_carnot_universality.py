# -*- coding: utf-8 -*-
"""
Phase 75: Carnot Engine Universality
Phase 46 found eta=0.924. Test if this efficiency is universal across
3 models and define the Transformer Carnot Constant.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def measure_carnot(model, tok, prompts, device):
    """Measure Carnot efficiency for a model."""
    efficiencies = []

    for prompt in prompts:
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
            T_cold = min(T_vals[len(T_vals)//2:])  # cold in later layers
            if T_hot > 0.01:
                eta = 1 - T_cold / T_hot
                efficiencies.append(eta)

    return efficiencies


def main():
    print("=" * 70)
    print("Phase 75: Carnot Engine Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
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

    model_etas = {}

    for model_size, model_name in [('1.5B', 'Qwen2.5-1.5B'), ('0.5B', 'Qwen2.5-0.5B')]:
        print(f"\n--- {model_name} ---")
        model, tok = load_model(device=device, size=model_size)
        etas = measure_carnot(model, tok, prompts, device)
        mean_eta = np.mean(etas)
        std_eta = np.std(etas)
        print(f"  eta = {mean_eta:.3f} +/- {std_eta:.3f}")
        model_etas[model_name] = {'etas': etas, 'mean': float(mean_eta), 'std': float(std_eta)}
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # TinyLlama
    try:
        _HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
        _SNAP_TL = os.path.join(_HF_CACHE, "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0", "snapshots")
        if os.path.exists(_SNAP_TL):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            snap_dir = os.path.join(_SNAP_TL, os.listdir(_SNAP_TL)[0])
            tok_tl = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
            model_tl = AutoModelForCausalLM.from_pretrained(
                snap_dir, torch_dtype=torch.float16, device_map=device, local_files_only=True)
            model_tl.eval()
            print(f"\n--- TinyLlama-1.1B ---")
            etas = measure_carnot(model_tl, tok_tl, prompts, device)
            mean_eta = np.mean(etas)
            std_eta = np.std(etas)
            print(f"  eta = {mean_eta:.3f} +/- {std_eta:.3f}")
            model_etas['TinyLlama-1.1B'] = {'etas': etas, 'mean': float(mean_eta), 'std': float(std_eta)}
            del model_tl
            import gc; gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TinyLlama error: {e}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'Qwen2.5-1.5B': '#e74c3c', 'Qwen2.5-0.5B': '#3498db', 'TinyLlama-1.1B': '#2ecc71'}
    mnames = list(model_etas.keys())

    # (a) Efficiency comparison
    means = [model_etas[m]['mean'] for m in mnames]
    stds = [model_etas[m]['std'] for m in mnames]
    axes[0, 0].bar(mnames, means, yerr=stds, capsize=5,
                   color=[colors.get(m, 'gray') for m in mnames], alpha=0.8)
    overall_mean = np.mean(means)
    axes[0, 0].axhline(y=overall_mean, color='black', linestyle='--',
                       label=f'Universal eta={overall_mean:.3f}')
    axes[0, 0].axhline(y=1.0, color='red', linestyle=':', label='Carnot limit (1.0)')
    axes[0, 0].set_ylabel('Carnot Efficiency eta')
    axes[0, 0].set_title('(a) Carnot Efficiency per Model')
    axes[0, 0].legend(fontsize=8)

    # (b) Distribution overlap
    for mname in mnames:
        axes[0, 1].hist(model_etas[mname]['etas'], bins=10, alpha=0.5,
                       color=colors.get(mname, 'gray'), label=mname, density=True)
    axes[0, 1].set_xlabel('eta')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('(b) Efficiency Distributions')
    axes[0, 1].legend(fontsize=8)

    # (c) Universality test: CV across models
    cv = np.std(means) / (np.mean(means) + 1e-10)
    axes[0, 2].bar(['Mean eta', 'Std across\nmodels', 'CV'],
                   [overall_mean, np.std(means), cv],
                   color=['#e74c3c', '#f39c12', '#3498db'], alpha=0.8)
    axes[0, 2].set_title(f'(c) Universality (CV={cv:.3f})')

    # (d) Individual prompt efficiencies (1.5B)
    if 'Qwen2.5-1.5B' in model_etas:
        etas_15 = model_etas['Qwen2.5-1.5B']['etas']
        axes[1, 0].bar(range(len(etas_15)), sorted(etas_15), color='#e74c3c', alpha=0.7)
        axes[1, 0].axhline(y=model_etas['Qwen2.5-1.5B']['mean'], color='black', linestyle='--')
        axes[1, 0].set_xlabel('Prompt (sorted)')
        axes[1, 0].set_ylabel('eta')
        axes[1, 0].set_title('(d) Per-Prompt Efficiency (1.5B)')

    # (e) Physical comparison
    physical_systems = {
        'Real heat engine': 0.30,
        'Car engine': 0.25,
        'Power plant': 0.40,
        'Ideal Carnot': 1.00,
        'LLM (ours)': overall_mean,
    }
    colors_phys = ['#95a5a6', '#95a5a6', '#95a5a6', '#f39c12', '#e74c3c']
    axes[1, 1].barh(list(physical_systems.keys()),
                     list(physical_systems.values()),
                     color=colors_phys, alpha=0.8)
    axes[1, 1].set_xlabel('Efficiency')
    axes[1, 1].set_title('(e) vs Physical Systems')

    # (f) Summary
    summary = (
        f"Carnot Efficiency:\n\n"
        + '\n'.join([f"  {m}: {model_etas[m]['mean']:.3f} +/- {model_etas[m]['std']:.3f}"
                     for m in mnames])
        + f"\n\nUniversal: eta = {overall_mean:.3f}\n"
        f"Cross-model CV = {cv:.3f}\n"
        f"{'UNIVERSAL' if cv < 0.15 else 'model-dependent'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=11, va='center', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    is_universal = cv < 0.15
    fig.suptitle(f'Phase 75: Carnot Universality (eta={overall_mean:.3f}, '
                 f'{"UNIVERSAL" if is_universal else "variable"})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase75_carnot_universality')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Universal eta={overall_mean:.3f}, cross-model CV={cv:.3f}. "
          f"Carnot efficiency {'IS' if is_universal else 'IS NOT'} a universal constant.")
    print(f"{'='*70}")

    save_results('phase75_carnot_universality', {
        'experiment': 'Carnot Universality',
        'per_model': {m: model_etas[m] for m in mnames},
        'summary': {
            'universal_eta': float(overall_mean),
            'cross_model_cv': float(cv),
            'is_universal': bool(is_universal),
        }
    })


if __name__ == '__main__':
    main()
