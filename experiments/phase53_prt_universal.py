# -*- coding: utf-8 -*-
"""
Phase 53: PRT Conservation Universality
Test PRT conservation across multiple architectures.
If PRT is conserved universally, it's a fundamental symmetry of Transformers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

CANDIDATE_MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
]

PROMPTS = [
    "The fundamental laws of physics describe how the universe operates at every scale from",
    "Machine learning algorithms can learn patterns from data without being explicitly",
    "The human brain processes information through billions of interconnected neurons that",
    "Evolution by natural selection operates on heritable variation within populations over",
    "The Turing test evaluates whether a machine can exhibit intelligent behavior",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using",
]


def measure_prt_conservation(model, tok, device, model_name):
    """Measure PRT conservation for one model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        norm_layer = model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        norm_layer = model.transformer.ln_f
    else:
        return None

    lm_head = model.lm_head
    all_prt_cvs = []
    all_prt_profiles = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        prt_vals = []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()

            try:
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                PR = 1.0 / (probs ** 2).sum().item()
                T = -(probs * torch.log(probs + 1e-10)).sum().item()
                PRT = PR * T
                if np.isnan(PRT):
                    PRT = 0
            except Exception:
                PRT = 0
            prt_vals.append(PRT)

        if len(prt_vals) > 2:
            cv = np.std(prt_vals) / (np.mean(prt_vals) + 1e-10)
            all_prt_cvs.append(cv)
            all_prt_profiles.append(prt_vals)

    mean_cv = np.mean(all_prt_cvs) if all_prt_cvs else 1.0
    print(f"  {model_name}: Mean PRT CV={mean_cv:.4f} ({len(all_prt_cvs)} prompts)")

    return {
        'model_name': model_name,
        'mean_cv': float(mean_cv),
        'all_cvs': [float(v) for v in all_prt_cvs],
        'prt_profiles': [[float(v) for v in p] for p in all_prt_profiles],
    }


def main():
    print("=" * 70)
    print("Phase 53: PRT Conservation Universality")
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
            result = measure_prt_conservation(model, tok, device, model_name)
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
        save_results('phase53_prt_universal', {'summary': {'verdict': 'INSUFFICIENT'}})
        return

    # === Visualization ===
    n = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = [r['model_name'] for r in all_results]
    x = np.arange(n)
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:n]

    # (a) Mean CV per model
    cvs = [r['mean_cv'] for r in all_results]
    axes[0].bar(x, cvs, color=colors, alpha=0.8)
    axes[0].axhline(y=0.05, color='gray', linestyle='--', label='CV=5%')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].set_ylabel('PRT CV (lower=better conservation)')
    axes[0].set_title('(a) PRT Conservation Quality')
    axes[0].legend()
    for i, v in enumerate(cvs):
        axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)

    # (b) PRT profiles per model
    for i, r in enumerate(all_results):
        for profile in r['prt_profiles']:
            axes[1].plot(profile, color=colors[i], alpha=0.15, linewidth=0.5)
        mean_p = np.mean(r['prt_profiles'], axis=0)
        axes[1].plot(mean_p, color=colors[i], linewidth=2, label=r['model_name'])
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('PRT')
    axes[1].set_title('(b) PRT Profiles')
    axes[1].legend(fontsize=8)

    # (c) CV distribution per model
    for i, r in enumerate(all_results):
        axes[2].hist(r['all_cvs'], bins=8, color=colors[i], alpha=0.5,
                    label=r['model_name'], edgecolor='black')
    axes[2].set_xlabel('PRT CV')
    axes[2].set_ylabel('Count')
    axes[2].set_title('(c) CV Distribution')
    axes[2].legend(fontsize=8)

    mean_cv_all = np.mean(cvs)
    all_conserved = all(cv < 0.15 for cv in cvs)
    fig.suptitle(f'Phase 53: PRT Conservation Universality '
                 f'(Mean CV={mean_cv_all:.3f}, All<15%: {all_conserved})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase53_prt_universal')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: PRT CV across {n} architectures: "
          + ", ".join(f"{r['model_name']}={r['mean_cv']:.3f}" for r in all_results)
          + f". Mean={mean_cv_all:.3f}. "
          f"PRT conservation is {'UNIVERSAL' if all_conserved else 'NOT universal'}.")
    print(f"{'='*70}")

    save_results('phase53_prt_universal', {
        'experiment': 'PRT Conservation Universality',
        'results': all_results,
        'summary': {
            'mean_cv': float(mean_cv_all),
            'all_conserved': all_conserved,
        }
    })


if __name__ == '__main__':
    main()
