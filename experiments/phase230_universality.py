# -*- coding: utf-8 -*-
"""
Phase 230: Cross-Architecture Universality
=============================================
THE critical test: Are the thermodynamic laws of transformers
UNIVERSAL across architectures, or Qwen-specific?

Test 7 architectures: Qwen, Llama, GPT2-XL, OPT, Phi-2, Falcon, Bloom
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_any_model, get_model_internals, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]

# 7 architectures spanning different design families
MODELS = [
    ('Qwen/Qwen2.5-0.5B',       'Qwen-0.5B'),
    ('Qwen/Qwen2.5-1.5B',       'Qwen-1.5B'),
    ('meta-llama/Llama-3.2-1B',  'Llama-1B'),
    ('gpt2-xl',                  'GPT2-XL'),
    ('facebook/opt-1.3b',        'OPT-1.3B'),
    ('microsoft/phi-2',          'Phi-2'),
    ('bigscience/bloom-1b1',     'Bloom-1.1B'),
]


def profile_model(model, tok, device, model_name):
    """Comprehensive thermodynamic profile for any architecture."""
    internals = get_model_internals(model)
    norm_layer = internals['norm']
    lm_head = internals['lm_head']
    n_layers = internals['n_layers']

    all_T, all_U, all_P1, all_PR = [], [], [], []
    all_cos = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_l, U_l, P1_l, PR_l, cos_l = [], [], [], [], []
        prev_h = None
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR_l.append(1.0 / ((h_prob ** 2).sum().item() + 1e-10))

            if prev_h is not None:
                cos_val = torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h.unsqueeze(0)).item()
                cos_l.append(cos_val)
            else:
                cos_l.append(0)
            prev_h = h.clone()

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)

        all_T.append(T_l); all_U.append(U_l)
        all_P1.append(P1_l); all_PR.append(PR_l)
        all_cos.append(cos_l)

    n = min(len(t) for t in all_T)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_T, mean_U, mean_P1, mean_PR, mean_cos = (
        avg(all_T), avg(all_U), avg(all_P1), avg(all_PR), avg(all_cos))

    dT = [mean_T[i+1] - mean_T[i] for i in range(n-1)]

    # Key metrics
    abs_dT = [abs(x) for x in dT]
    L_ignition = int(np.argmax(abs_dT))
    rho_S, p_S = stats.spearmanr(range(n), mean_T)
    rho_P1, p_P1 = stats.spearmanr(range(n), mean_P1)

    # Onsager symmetry (quick version)
    all_dT_per_prompt = [[all_T[p][i+1] - all_T[p][i] for i in range(n-1)] for p in range(len(PROMPTS))]
    all_dU_per_prompt = [[all_U[p][i+1] - all_U[p][i] for i in range(n-1)] for p in range(len(PROMPTS))]
    L_TU, L_UT = 0, 0
    for l in range(min(len(all_dT_per_prompt[0]), len(all_dU_per_prompt[0]))):
        jt = [all_dT_per_prompt[p][l] for p in range(len(PROMPTS))]
        ju = [all_dU_per_prompt[p][l] for p in range(len(PROMPTS))]
        L_TU += float(np.mean(np.array(jt) * np.array(ju)))
        L_UT += float(np.mean(np.array(ju) * np.array(jt)))
    onsager_asym = abs(L_TU - L_UT) / (abs(L_TU) + abs(L_UT) + 1e-10)

    # Normalized T profile for RG comparison
    T_range = max(mean_T) - min(mean_T) if max(mean_T) > min(mean_T) else 1
    T_norm = [(t - min(mean_T)) / T_range for t in mean_T]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'n_states': n,
        'mean_T': mean_T,
        'mean_U': mean_U,
        'mean_P1': mean_P1,
        'mean_cos': mean_cos,
        'T_norm': T_norm,
        'dT': [float(x) for x in dT],
        'L_ignition': L_ignition,
        'rho_S': float(rho_S),
        'p_S': float(p_S),
        'rho_P1': float(rho_P1),
        'p_P1': float(p_P1),
        'onsager_asym': onsager_asym,
        'T_initial': mean_T[0],
        'T_final': mean_T[-1],
        'P1_initial': mean_P1[0],
        'P1_final': mean_P1[-1],
    }


def main():
    print("=" * 70)
    print("Phase 230: Cross-Architecture Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for model_id, short_name in MODELS:
        print(f"\n--- {short_name} ({model_id}) ---")
        try:
            model, tok = load_any_model(model_id, device=device)
            r = profile_model(model, tok, device, short_name)
            results[short_name] = r
            print(f"  {r['n_layers']}L, Ignition=L{r['L_ignition']}")
            print(f"  Arrow: rho(S)={r['rho_S']:.4f}, rho(P1)={r['rho_P1']:.4f}")
            print(f"  Onsager asym={r['onsager_asym']:.6f}")
            print(f"  T: {r['T_initial']:.2f} -> {r['T_final']:.2f}")
            print(f"  P1: {r['P1_initial']:.4f} -> {r['P1_final']:.4f}")
            del model, tok
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cmap = plt.cm.Set1
    model_colors = {name: cmap(i / len(results)) for i, name in enumerate(results.keys())}

    # (a) T profiles (raw)
    for name, r in results.items():
        x = np.linspace(0, 1, len(r['mean_T']))
        axes[0, 0].plot(x, r['mean_T'], '-', color=model_colors[name], lw=1.5,
                       label=f"{name} ({r['n_layers']}L)")
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('Temperature T')
    axes[0, 0].set_title('(a) Temperature Profiles')
    axes[0, 0].legend(fontsize=6, ncol=2)

    # (b) Normalized T profiles (universality test)
    for name, r in results.items():
        x = np.linspace(0, 1, len(r['T_norm']))
        axes[0, 1].plot(x, r['T_norm'], '-', color=model_colors[name], lw=1.5,
                       label=name)
    axes[0, 1].set_xlabel('Normalized Depth')
    axes[0, 1].set_ylabel('Normalized T')
    axes[0, 1].set_title('(b) Universal Profile Test')
    axes[0, 1].legend(fontsize=6, ncol=2)

    # (c) P1 profiles
    for name, r in results.items():
        x = np.linspace(0, 1, len(r['mean_P1']))
        axes[0, 2].plot(x, r['mean_P1'], '-', color=model_colors[name], lw=1.5,
                       label=name)
    axes[0, 2].set_xlabel('Normalized Depth')
    axes[0, 2].set_ylabel('P1')
    axes[0, 2].set_title('(c) Order Parameter P1')
    axes[0, 2].legend(fontsize=6, ncol=2)

    # (d) Arrow strength comparison
    names = list(results.keys())
    rho_S_vals = [results[n]['rho_S'] for n in names]
    rho_P1_vals = [results[n]['rho_P1'] for n in names]
    x = np.arange(len(names))
    axes[1, 0].bar(x - 0.15, rho_S_vals, 0.3, label='rho(S)', color='steelblue')
    axes[1, 0].bar(x + 0.15, rho_P1_vals, 0.3, label='rho(P1)', color='coral')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, fontsize=7, ha='right')
    axes[1, 0].set_ylabel('Spearman rho')
    axes[1, 0].set_title('(d) Arrow Strength')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].axhline(y=0, color='gray', ls='-', alpha=0.3)

    # (e) Onsager asymmetry
    onsager_vals = [results[n]['onsager_asym'] for n in names]
    axes[1, 1].bar(range(len(names)), onsager_vals, color=[model_colors[n] for n in names])
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, fontsize=7, ha='right')
    axes[1, 1].set_ylabel('|L_TU - L_UT| / (|L_TU| + |L_UT|)')
    axes[1, 1].set_title('(e) Onsager Asymmetry')
    axes[1, 1].axhline(y=0.1, color='red', ls='--', alpha=0.5, label='Threshold')
    axes[1, 1].legend(fontsize=7)

    # (f) Summary
    summary = "UNIVERSALITY TEST\n\n"
    all_negative_rho = all(results[n]['rho_S'] < 0 for n in names)
    all_positive_p1 = all(results[n]['rho_P1'] > 0 for n in names)
    all_symmetric = all(results[n]['onsager_asym'] < 0.01 for n in names)
    summary += f"Arrow (S decreasing): {'UNIVERSAL' if all_negative_rho else 'NOT universal'}\n"
    summary += f"Arrow (P1 increasing): {'UNIVERSAL' if all_positive_p1 else 'NOT universal'}\n"
    summary += f"Onsager symmetric: {'UNIVERSAL' if all_symmetric else 'NOT universal'}\n\n"
    for n in names:
        r = results[n]
        summary += f"{n}: rho_S={r['rho_S']:.3f}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 230: Cross-Architecture Universality (7 Models)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase230_universality')
    plt.close()
    save_results('phase230_universality', {'experiment': 'Cross-Architecture Universality',
                                           'results': results})


if __name__ == '__main__':
    main()
