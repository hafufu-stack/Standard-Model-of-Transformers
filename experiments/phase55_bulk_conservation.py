# -*- coding: utf-8 -*-
"""
Phase 55: Bulk Conservation Law
Re-test PRT conservation excluding Layer 0 (Big Bang singularity).
PRT should be conserved in the 'bulk space' (L1 to L_end).
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
    "The cosmic microwave background radiation provides evidence of the early universe",
    "Cryptographic hash functions transform arbitrary data into fixed-size output strings",
]


def measure_bulk_prt(model, tok, device, model_name):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        norm_layer = model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        norm_layer = model.transformer.ln_f
    else:
        return None

    lm_head = model.lm_head
    all_full_cvs = []
    all_bulk_cvs = []
    all_bulk_profiles = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        prt_vals = []
        for hs in out.hidden_states:
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

        if len(prt_vals) > 3:
            # Full (including Layer 0)
            full_cv = np.std(prt_vals) / (np.mean(prt_vals) + 1e-10)
            all_full_cvs.append(full_cv)

            # Bulk (excluding Layer 0 and optionally Layer 1)
            bulk = prt_vals[2:]  # Skip L0 and L1 (thermalization)
            if len(bulk) > 2 and np.mean(bulk) > 0:
                bulk_cv = np.std(bulk) / (np.mean(bulk) + 1e-10)
                all_bulk_cvs.append(bulk_cv)
                all_bulk_profiles.append(bulk)

    mean_full = np.mean(all_full_cvs) if all_full_cvs else 1.0
    mean_bulk = np.mean(all_bulk_cvs) if all_bulk_cvs else 1.0

    print(f"  {model_name}: Full CV={mean_full:.4f}, Bulk CV={mean_bulk:.4f} "
          f"(improvement: {(1-mean_bulk/mean_full)*100:.0f}%)")

    return {
        'model_name': model_name,
        'full_cv': float(mean_full),
        'bulk_cv': float(mean_bulk),
        'all_full_cvs': [float(v) for v in all_full_cvs],
        'all_bulk_cvs': [float(v) for v in all_bulk_cvs],
        'bulk_profiles': [[float(v) for v in p] for p in all_bulk_profiles],
    }


def main():
    print("=" * 70)
    print("Phase 55: Bulk Conservation Law (PRT in L2+)")
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
            result = measure_bulk_prt(model, tok, device, model_name)
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
        save_results('phase55_bulk_conservation', {'summary': {'verdict': 'INSUFFICIENT'}})
        return

    # === Visualization ===
    n = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = [r['model_name'] for r in all_results]
    x = np.arange(n)
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:n]

    # (a) Full vs Bulk CV
    full_cvs = [r['full_cv'] for r in all_results]
    bulk_cvs = [r['bulk_cv'] for r in all_results]
    w = 0.35
    axes[0].bar(x - w/2, full_cvs, w, color='#e74c3c', alpha=0.7, label='Full (L0+)')
    axes[0].bar(x + w/2, bulk_cvs, w, color='#2ecc71', alpha=0.7, label='Bulk (L2+)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].set_ylabel('PRT CV')
    axes[0].set_title('(a) Full vs Bulk Conservation')
    axes[0].legend()
    for i in range(n):
        axes[0].text(i + w/2, bulk_cvs[i] + 0.01, f'{bulk_cvs[i]:.3f}',
                    ha='center', fontsize=9, color='green')

    # (b) Bulk PRT profiles
    for i, r in enumerate(all_results):
        for p in r['bulk_profiles']:
            axes[1].plot(p, color=colors[i], alpha=0.15, linewidth=0.5)
        if r['bulk_profiles']:
            mean_p = np.mean(r['bulk_profiles'], axis=0)
            axes[1].plot(mean_p, color=colors[i], linewidth=2, label=r['model_name'])
    axes[1].set_xlabel('Bulk Layer (L2+)')
    axes[1].set_ylabel('PRT')
    axes[1].set_title('(b) Bulk PRT Profiles')
    axes[1].legend(fontsize=8)

    # (c) Improvement
    improvements = [(1 - b/f) * 100 if f > 0 else 0
                    for f, b in zip(full_cvs, bulk_cvs)]
    axes[2].bar(x, improvements, color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha='right')
    axes[2].set_ylabel('CV Improvement (%)')
    axes[2].set_title('(c) Improvement from Bulk Exclusion')
    for i, v in enumerate(improvements):
        axes[2].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=10)

    all_conserved = all(cv < 0.15 for cv in bulk_cvs)
    mean_bulk = np.mean(bulk_cvs)
    fig.suptitle(f'Phase 55: Bulk Conservation Law (Mean Bulk CV={mean_bulk:.3f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase55_bulk_conservation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: Bulk PRT CV: "
          + ", ".join(f"{r['model_name']}={r['bulk_cv']:.3f}" for r in all_results)
          + f". Mean={mean_bulk:.3f}. "
          f"Bulk conservation is {'UNIVERSAL' if all_conserved else 'improved but not yet universal'}.")
    print(f"{'='*70}")

    save_results('phase55_bulk_conservation', {
        'experiment': 'Bulk Conservation Law',
        'results': all_results,
        'summary': {
            'mean_bulk_cv': float(mean_bulk),
            'all_conserved': all_conserved,
        }
    })


if __name__ == '__main__':
    main()
