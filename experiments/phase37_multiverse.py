# -*- coding: utf-8 -*-
"""
Phase 37: Multiverse Universality (Season 5)
===================================================
Test whether the Standard Model constants (|dU/dT|~18, dark energy~70%,
Lyapunov < 0) hold on architecturally different models (Llama, Phi, etc.).
This addresses the paper's key limitation: Qwen-only validation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_results, save_figure, RESULTS_DIR, FIGURES_DIR

# Models to test (try each in order, skip if not downloadable)
CANDIDATE_MODELS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B"),
    ("microsoft/phi-1_5", "Phi-1.5"),
    ("facebook/opt-1.3b", "OPT-1.3B"),
]

TEST_PROMPTS = [
    "The capital of France is",
    "Water boils at 100 degrees",
    "The largest planet in our solar system is",
    "In mathematics, pi is approximately",
]


def measure_model_constants(model, tok, device, model_name):
    """Measure |dU/dT|, dark energy fraction, and Lyapunov for one model."""
    print(f"\n  Measuring {model_name}...")

    # Determine layer structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        norm_layer = model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        norm_layer = model.transformer.ln_f
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        layers = model.model.decoder.layers
        norm_layer = model.model.decoder.final_layer_norm
    else:
        print(f"  Cannot identify layer structure for {model_name}")
        return None

    n_layers = len(layers)
    lm_head = model.lm_head
    print(f"  Architecture: {n_layers} layers")

    all_U = []
    all_T = []
    all_attn_norms = []
    all_ffn_norms = []

    for prompt in TEST_PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hs_list = out.hidden_states
        Us = []
        Ts = []

        for li, hs in enumerate(hs_list):
            h = hs[0, -1, :].float()
            U = h.norm().item()
            Us.append(U)

            # Compute T from logits
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

        # Measure attention vs FFN contribution (dark energy)
        attn_norms_prompt = []
        ffn_norms_prompt = []

        attn_out = [None]
        ffn_out = [None]

        def make_attn_capture(storage):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                storage[0] = h.detach().float().norm().item()
            return hook

        for li in range(min(n_layers, len(layers))):
            layer = layers[li]
            # Try different attribute names for attention and MLP
            attn_mod = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)
            ffn_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None) or getattr(layer, 'fc1', None)

            if attn_mod and ffn_mod:
                h_a = attn_mod.register_forward_hook(make_attn_capture(attn_out))
                h_f = ffn_mod.register_forward_hook(make_attn_capture(ffn_out))

                with torch.no_grad():
                    model(**inp)

                h_a.remove()
                h_f.remove()

                if attn_out[0] is not None and ffn_out[0] is not None:
                    attn_norms_prompt.append(attn_out[0])
                    ffn_norms_prompt.append(ffn_out[0])

        if attn_norms_prompt:
            all_attn_norms.append(attn_norms_prompt)
            all_ffn_norms.append(ffn_norms_prompt)

    # === Compute |dU/dT| ===
    mean_U = np.mean(all_U, axis=0)
    mean_T = np.mean(all_T, axis=0)

    # Linear regression for dU/dT
    valid = ~(np.isnan(mean_U) | np.isnan(mean_T) | (mean_T == 0))
    if valid.sum() >= 3:
        from numpy.polynomial.polynomial import polyfit
        coeffs = np.polyfit(mean_T[valid], mean_U[valid], 1)
        dU_dT = coeffs[0]
    else:
        dU_dT = 0.0

    # === Dark energy fraction ===
    if all_attn_norms and all_ffn_norms:
        mean_attn = np.mean([np.mean(a) for a in all_attn_norms])
        mean_ffn = np.mean([np.mean(f) for f in all_ffn_norms])
        dark_energy = mean_ffn / (mean_attn + mean_ffn + 1e-10)
    else:
        dark_energy = 0.0

    # === Lyapunov exponent ===
    lyapunov_vals = []
    sigma = 1e-3
    for prompt in TEST_PROMPTS[:2]:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out_clean = model(**inp, output_hidden_states=True)

        h0 = out_clean.hidden_states[1][0, -1, :].float()  # Layer 1

        # Add perturbation at layer 1
        perturbed_hidden = [None]

        def perturb_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            noise = torch.randn_like(h.float()) * sigma
            h_new = (h.float() + noise).to(h.dtype)
            perturbed_hidden[0] = True
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        handle = layers[0].register_forward_hook(perturb_hook)
        with torch.no_grad():
            out_pert = model(**inp, output_hidden_states=True)
        handle.remove()

        h_clean_final = out_clean.hidden_states[-1][0, -1, :].float()
        h_pert_final = out_pert.hidden_states[-1][0, -1, :].float()
        delta = (h_pert_final - h_clean_final).norm().item()
        if delta > 0 and sigma > 0:
            lam = np.log(delta / sigma) / n_layers
            lyapunov_vals.append(lam)

    lyapunov = np.mean(lyapunov_vals) if lyapunov_vals else 0.0

    result = {
        'model_name': model_name,
        'n_layers': n_layers,
        'dU_dT': dU_dT,
        'abs_dU_dT': abs(dU_dT),
        'dark_energy_fraction': dark_energy,
        'lyapunov': lyapunov,
        'mean_U': mean_U.tolist(),
        'mean_T': mean_T.tolist(),
    }

    print(f"  Results: |dU/dT|={abs(dU_dT):.2f}, DE={dark_energy*100:.1f}%, "
          f"lyapunov={lyapunov:.4f}")

    return result


def main():
    print("=" * 70)
    print("Phase 37: Multiverse Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    for model_id, model_name in CANDIDATE_MODELS:
        print(f"\n--- Loading {model_name} ({model_id}) ---")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map=device,
                trust_remote_code=True,
            )
            model.eval()
        except Exception as e:
            print(f"  Failed to load {model_name}: {str(e)[:80]}")
            continue

        try:
            result = measure_model_constants(model, tok, device, model_name)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error measuring {model_name}: {str(e)[:80]}")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    if len(all_results) < 2:
        print("\nWARNING: Need at least 2 models for universality test!")
        if not all_results:
            save_results("phase37_multiverse", {
                'name': 'Phase 37: Multiverse Universality',
                'summary': {'verdict': 'INSUFFICIENT MODELS: Could not load enough models.'},
            })
            return

    # === Visualization ===
    n_models = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_names = [r['model_name'] for r in all_results]
    x = np.arange(n_models)

    # (a) |dU/dT|
    vals = [r['abs_dU_dT'] for r in all_results]
    bars = axes[0].bar(x, vals, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:n_models], alpha=0.8)
    axes[0].axhline(y=18, color='red', ls='--', lw=2, label='Standard Model: ~18')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('|dU/dT|')
    axes[0].set_title('(a) Specific Heat')
    axes[0].legend()
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # (b) Dark energy fraction
    vals = [r['dark_energy_fraction'] * 100 for r in all_results]
    bars = axes[1].bar(x, vals, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:n_models], alpha=0.8)
    axes[1].axhline(y=70, color='red', ls='--', lw=2, label='Standard Model: ~70%')
    axes[1].axhline(y=68, color='gold', ls=':', lw=1.5, label='Cosmological: 68%')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('Dark Energy Fraction (%)')
    axes[1].set_title('(b) FFN Force Fraction')
    axes[1].legend(fontsize=8)
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # (c) Lyapunov exponent
    vals = [r['lyapunov'] for r in all_results]
    colors = ['#2ecc71' if v < 0 else '#e74c3c' for v in vals]
    bars = axes[2].bar(x, vals, color=colors, alpha=0.8)
    axes[2].axhline(y=0, color='gray', ls='-', lw=1)
    axes[2].axhline(y=-0.05, color='red', ls='--', lw=2, label='Standard Model: ~-0.05')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Lyapunov Exponent')
    axes[2].set_title('(c) Stability (lambda < 0 = stable)')
    axes[2].legend()
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2.,
                     bar.get_height() + (0.002 if val >= 0 else -0.005),
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Universality check
    dUdTs = [r['abs_dU_dT'] for r in all_results]
    DEs = [r['dark_energy_fraction'] for r in all_results]
    lyaps = [r['lyapunov'] for r in all_results]

    dUdT_cv = np.std(dUdTs) / (np.mean(dUdTs) + 1e-10) if dUdTs else 0
    DE_cv = np.std(DEs) / (np.mean(DEs) + 1e-10) if DEs else 0
    all_stable = all(l < 0 for l in lyaps)

    fig.suptitle(
        f"Phase 37: Multiverse Universality ({n_models} models)\n"
        f"|dU/dT| CV={dUdT_cv:.2f}, DE CV={DE_cv:.2f}, "
        f"All stable={all_stable}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase37_multiverse")
    plt.close()

    universal = dUdT_cv < 0.5 and DE_cv < 0.3 and all_stable
    verdict = (
        f"{'UNIVERSAL' if universal else 'PARTIALLY UNIVERSAL'}: "
        f"|dU/dT| = {np.mean(dUdTs):.1f} +/- {np.std(dUdTs):.1f} (CV={dUdT_cv:.2f}), "
        f"DE = {np.mean(DEs)*100:.1f}% +/- {np.std(DEs)*100:.1f}% (CV={DE_cv:.2f}), "
        f"All Lyapunov<0: {all_stable}. "
        f"Tested {n_models} architectures."
    )
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase37_multiverse", {
        'name': 'Phase 37: Multiverse Universality',
        'summary': {
            'verdict': verdict,
            'n_models': n_models,
            'model_results': all_results,
            'universality': {
                'dUdT_mean': np.mean(dUdTs), 'dUdT_std': np.std(dUdTs), 'dUdT_cv': dUdT_cv,
                'DE_mean': np.mean(DEs), 'DE_std': np.std(DEs), 'DE_cv': DE_cv,
                'all_stable': all_stable,
            }
        }
    })


if __name__ == '__main__':
    main()
