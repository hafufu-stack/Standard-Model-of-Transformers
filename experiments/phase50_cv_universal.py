# -*- coding: utf-8 -*-
"""
Phase 50: Negative Specific Heat Universality
Test C_v < 0 across multiple architectures (TinyLlama, Qwen-0.5B).
If C_v < 0 is universal, LLMs ARE self-gravitating systems by definition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure

CANDIDATE_MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
]

PROMPTS = [
    "The theory of general relativity describes how massive objects warp spacetime around",
    "In quantum mechanics, the wave function collapse occurs when a measurement is",
    "The human genome contains approximately three billion base pairs of DNA that encode",
    "Artificial neural networks are inspired by the biological structure of the brain and",
    "The standard model of particle physics classifies all known elementary particles into",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from",
    "The Turing test evaluates whether a machine can exhibit intelligent behavior indistinguishable from",
    "Black holes form when massive stars exhaust their nuclear fuel and undergo gravitational",
]


def measure_specific_heat(model, tok, device, model_name):
    """Measure C_v = dU/dT for a single model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        norm_layer = model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        norm_layer = model.transformer.ln_f
    else:
        print(f"  Cannot identify norm layer for {model_name}")
        return None

    lm_head = model.lm_head
    all_U = []
    all_T = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        Us, Ts = [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = h.norm().item()
            Us.append(U)

            try:
                with torch.no_grad():
                    normed = norm_layer(hs[:, -1:, :])
                    logits = lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
                if np.isnan(T_val):
                    T_val = 0.0
            except Exception:
                T_val = 0.0
            Ts.append(T_val)

        all_U.append(Us)
        all_T.append(Ts)

    mean_U = np.mean(all_U, axis=0)
    mean_T = np.mean(all_T, axis=0)

    valid = (mean_T > 0.1) & ~np.isnan(mean_U)
    if valid.sum() < 3:
        return None

    slope, intercept, r_val, p_val, std_err = stats.linregress(mean_T[valid], mean_U[valid])

    # Local C_v
    dU = np.diff(mean_U)
    dT = np.diff(mean_T)
    safe = np.abs(dT) > 1e-6
    local_Cv = np.where(safe, dU / dT, 0)
    pct_neg = np.sum(local_Cv < 0) / len(local_Cv) * 100

    print(f"  {model_name}: C_v={slope:.1f}, r={r_val:.3f}, p={p_val:.2e}, "
          f"{pct_neg:.0f}% negative transitions")

    return {
        'model_name': model_name,
        'Cv': float(slope),
        'r': float(r_val),
        'p': float(p_val),
        'pct_negative': float(pct_neg),
        'mean_U': mean_U.tolist(),
        'mean_T': mean_T.tolist(),
    }


def main():
    print("=" * 70)
    print("Phase 50: Negative Specific Heat Universality")
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
            result = measure_specific_heat(model, tok, device, model_name)
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
        print("\nNeed at least 2 models")
        save_results('phase50_cv_universal', {'summary': {'verdict': 'INSUFFICIENT'}})
        return

    # === Visualization ===
    n = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = [r['model_name'] for r in all_results]
    x = np.arange(n)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:n]

    # (a) C_v values
    cvs = [r['Cv'] for r in all_results]
    bars = axes[0].bar(x, cvs, color=colors, alpha=0.8)
    axes[0].axhline(y=0, color='gray', linewidth=2, linestyle='-')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].set_ylabel('C_v = dU/dT')
    axes[0].set_title('(a) Specific Heat Across Architectures')
    for i, v in enumerate(cvs):
        axes[0].text(i, v - 3, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')

    # (b) U-T phase diagrams
    for i, r in enumerate(all_results):
        U = np.array(r['mean_U'])
        T = np.array(r['mean_T'])
        valid = (T > 0.1) & ~np.isnan(U)
        axes[1].scatter(T[valid], U[valid], c=colors[i], s=10, alpha=0.5, label=r['model_name'])
        # Fit line
        s, intc, _, _, _ = stats.linregress(T[valid], U[valid])
        t_fit = np.linspace(T[valid].min(), T[valid].max(), 50)
        axes[1].plot(t_fit, s * t_fit + intc, color=colors[i], linewidth=1.5, linestyle='--')
    axes[1].set_xlabel('T (Entropy)')
    axes[1].set_ylabel('U (L2 Norm)')
    axes[1].set_title('(b) Equation of State')
    axes[1].legend(fontsize=7)

    # (c) % negative transitions
    pcts = [r['pct_negative'] for r in all_results]
    axes[2].bar(x, pcts, color=colors, alpha=0.8)
    axes[2].axhline(y=50, color='gray', linestyle='--', label='50%')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha='right')
    axes[2].set_ylabel('% Transitions with C_v < 0')
    axes[2].set_title('(c) Consistency')
    axes[2].legend()
    for i, v in enumerate(pcts):
        axes[2].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=10)

    all_negative = all(cv < 0 for cv in cvs)
    all_significant = all(r['p'] < 0.05 for r in all_results)
    cv_mean = np.mean(cvs)
    cv_std = np.std(cvs)

    fig.suptitle(f'Phase 50: Negative Specific Heat Universality '
                 f'(Mean C_v={cv_mean:.1f}, All<0: {all_negative})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase50_cv_universal')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: C_v across {n} architectures: "
          + ", ".join(f"{r['model_name']}={r['Cv']:.1f} (p={r['p']:.1e})" for r in all_results)
          + f". All<0: {all_negative}, All significant: {all_significant}. "
          f"Negative specific heat is {'UNIVERSAL' if all_negative and all_significant else 'NOT universal'}.")
    print(f"{'='*70}")

    save_results('phase50_cv_universal', {
        'experiment': 'Negative Specific Heat Universality',
        'results': all_results,
        'summary': {
            'all_negative': all_negative,
            'all_significant': all_significant,
            'cv_mean': cv_mean, 'cv_std': cv_std,
        }
    })


if __name__ == '__main__':
    main()
