# -*- coding: utf-8 -*-
"""
Phase 262: Cross-Architecture Validation of P1 x T Conservation
=================================================================
Phase 257/261 discovered P1*T_sm is conserved (CV~0.14) in Qwen2.5.
This phase tests universality across different architectures:
- GPT-2 (117M, different tokenizer, different normalization)
- Llama-3.2-1B (GQA, RoPE, different scale)

If P1*T conservation holds across architectures, it's a UNIVERSAL law.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "General relativity describes gravity as spacetime curvature",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
    "Colorless green ideas sleep furiously in",
]

MODELS = {
    'GPT2': {
        'name': 'gpt2',
        'norm_path': lambda m: m.transformer.ln_f,
        'head_path': lambda m: m.lm_head,
        'layers_path': lambda m: m.transformer.h,
    },
    'Llama-3.2-1B': {
        'name': 'meta-llama/Llama-3.2-1B',
        'norm_path': lambda m: m.model.norm,
        'head_path': lambda m: m.lm_head,
        'layers_path': lambda m: m.model.layers,
    },
    'Qwen2.5-0.5B': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'norm_path': lambda m: m.model.norm,
        'head_path': lambda m: m.lm_head,
        'layers_path': lambda m: m.model.layers,
    },
}


def measure_P1T(model, tok, device, norm_layer, lm_head, arch_name):
    """Measure P1*T trajectory across layers for any architecture."""
    all_P1, all_T, all_PRT = [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        P1_l, T_l, PRT_l = [], [], []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            T_sm = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T_sm):
                T_sm = 0
            P1_l.append(P1)
            T_l.append(T_sm)
            PRT_l.append(P1 * T_sm)

        all_P1.append(P1_l)
        all_T.append(T_l)
        all_PRT.append(PRT_l)

    n = min(len(p) for p in all_P1)
    avg = lambda d: [float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)]

    mean_P1 = avg(all_P1)
    mean_T = avg(all_T)
    mean_PRT = avg(all_PRT)

    # CV of PRT (skip embedding layer)
    cv = float(np.std(mean_PRT[1:]) / (np.mean(mean_PRT[1:]) + 1e-10))

    # Arrow of time: T decreasing?
    rho_T, _ = stats.spearmanr(range(n), mean_T)

    return {
        'arch': arch_name,
        'n_layers': n,
        'mean_P1': mean_P1,
        'mean_T': mean_T,
        'mean_PRT': mean_PRT,
        'PRT_cv': round(cv, 4),
        'PRT_mean': round(float(np.mean(mean_PRT[1:])), 4),
        'T_arrow_rho': round(float(rho_T), 4),
    }


def main():
    print("=" * 70)
    print("Phase 262: Cross-Architecture P1*T Validation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for arch_name, cfg in MODELS.items():
        print(f"\n--- {arch_name} ---")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg['name'], torch_dtype=torch.float16,
                device_map=device, local_files_only=True)
            tok = AutoTokenizer.from_pretrained(cfg['name'], local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            norm_layer = cfg['norm_path'](model)
            lm_head = cfg['head_path'](model)

            r = measure_P1T(model, tok, device, norm_layer, lm_head, arch_name)
            results[arch_name] = r
            print(f"  P1*T CV = {r['PRT_cv']:.4f}, mean = {r['PRT_mean']:.2f}")
            print(f"  T arrow rho = {r['T_arrow_rho']:.4f}")
            print(f"  n_layers = {r['n_layers']}")

            del model, tok
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  SKIP ({e})")

    if not results:
        print("No models loaded successfully!")
        return

    # === Universality test ===
    cvs = [r['PRT_cv'] for r in results.values()]
    all_conserved = all(cv < 0.3 for cv in cvs)
    mean_cv = float(np.mean(cvs))

    print(f"\n  Cross-architecture CV: {[f'{k}={v['PRT_cv']:.3f}' for k, v in results.items()]}")
    print(f"  Mean CV = {mean_cv:.4f}")
    print(f"  Universal? {'YES' if all_conserved else 'NO'}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    # (a) P1*T profiles
    for i, (arch, r) in enumerate(results.items()):
        c = colors_list[i % len(colors_list)]
        axes[0, 0].plot(np.linspace(0, 1, len(r['mean_PRT'])), r['mean_PRT'],
                       '-o', color=c, markersize=3, lw=2,
                       label=f"{arch} (CV={r['PRT_cv']:.3f})")
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('P1 x T')
    axes[0, 0].set_title('(a) P1*T Conservation Across Architectures', fontweight='bold')
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    # (b) T profiles
    for i, (arch, r) in enumerate(results.items()):
        c = colors_list[i % len(colors_list)]
        axes[0, 1].plot(np.linspace(0, 1, len(r['mean_T'])), r['mean_T'],
                       '-', color=c, lw=2, label=f"{arch}")
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('T_sm (entropy)')
    axes[0, 1].set_title('(b) Temperature Profiles', fontweight='bold')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # (c) P1 profiles
    for i, (arch, r) in enumerate(results.items()):
        c = colors_list[i % len(colors_list)]
        axes[0, 2].plot(np.linspace(0, 1, len(r['mean_P1'])), r['mean_P1'],
                       '-', color=c, lw=2, label=arch)
    axes[0, 2].set_xlabel('Normalized Depth')
    axes[0, 2].set_ylabel('P1 (max prob)')
    axes[0, 2].set_title('(c) Max Probability Profiles', fontweight='bold')
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    # (d) P1 vs T scatter
    for i, (arch, r) in enumerate(results.items()):
        c = colors_list[i % len(colors_list)]
        axes[1, 0].plot(r['mean_T'], r['mean_P1'], 'o-', color=c,
                       markersize=4, lw=1.5, label=arch)
    axes[1, 0].set_xlabel('T_sm'); axes[1, 0].set_ylabel('P1')
    axes[1, 0].set_title('(d) State Space (T, P1)', fontweight='bold')
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)

    # (e) CV comparison bar
    archs = list(results.keys())
    cvs_plot = [results[a]['PRT_cv'] for a in archs]
    bars = axes[1, 1].bar(range(len(archs)), cvs_plot,
                          color=[colors_list[i % len(colors_list)] for i in range(len(archs))],
                          edgecolor='black', alpha=0.8)
    axes[1, 1].set_xticks(range(len(archs)))
    axes[1, 1].set_xticklabels(archs, fontsize=8, rotation=15)
    axes[1, 1].set_ylabel('CV(P1*T)')
    axes[1, 1].set_title('(e) Conservation Quality (lower = better)', fontweight='bold')
    axes[1, 1].axhline(0.3, color='red', ls='--', lw=1, label='Threshold')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3, axis='y')
    for bar, cv in zip(bars, cvs_plot):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{cv:.3f}', ha='center', fontsize=8, fontweight='bold')

    # (f) Summary
    summary = "CROSS-ARCHITECTURE VALIDATION\n\n"
    for arch, r in results.items():
        summary += f"{arch:20s}: CV={r['PRT_cv']:.4f}\n"
    summary += f"\nMean CV = {mean_cv:.4f}\n"
    summary += f"Universal: {'YES' if all_conserved else 'NO'}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 262: Is P1*T Conservation Universal?",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase262_cross_architecture')
    plt.close()

    save_results('phase262_cross_architecture', {
        'experiment': 'Cross-Architecture P1*T Validation',
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'mean_P1' and kk != 'mean_T' and kk != 'mean_PRT'}
                   for k, v in results.items()},
        'profiles': {k: {'P1': v['mean_P1'], 'T': v['mean_T'], 'PRT': v['mean_PRT']}
                    for k, v in results.items()},
        'mean_cv': round(mean_cv, 4),
        'universal': all_conserved,
    })


if __name__ == '__main__':
    main()
