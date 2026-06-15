# -*- coding: utf-8 -*-
"""
Phase 238: Instruct vs Base — The Thermodynamic Effect of RLHF
================================================================
Compare Qwen2.5-0.5B (base) vs Qwen2.5-0.5B-Instruct (fine-tuned).
Question: Does instruction tuning lower temperature, strengthen the arrow,
          or change the phase transition pattern?
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
    "The speed of light is constant in all reference frames",
    "DNA encodes genetic information using four bases",
    "Entropy always increases in closed systems",
    "The Higgs boson gives particles their mass",
    "Plate tectonics shapes the surface of the Earth",
]

MODELS = [
    ('Qwen/Qwen2.5-0.5B',          'Base-0.5B'),
    ('Qwen/Qwen2.5-0.5B-Instruct', 'Instruct-0.5B'),
    ('Qwen/Qwen2.5-1.5B',          'Base-1.5B'),
    ('Qwen/Qwen2.5-1.5B-Instruct', 'Instruct-1.5B'),
]


def profile_model(model, tok, device, name):
    """Full thermodynamic profile."""
    internals = get_model_internals(model)
    norm_layer = internals['norm']
    lm_head = internals['lm_head']
    n_layers = internals['n_layers']

    all_T, all_P1, all_U = [], [], []
    all_kl_from_uniform = []

    vocab_size = lm_head.out_features if hasattr(lm_head, 'out_features') else lm_head.weight.shape[0]
    uniform = torch.ones(vocab_size, device=device) / vocab_size

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, P1_l, U_l, kl_l = [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_l.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)
            # KL from uniform
            kl = torch.nn.functional.kl_div(
                torch.log(uniform + 1e-10), probs, reduction='sum').item()
            kl_l.append(float(kl) if not np.isnan(kl) else 0)
        all_T.append(T_l); all_P1.append(P1_l)
        all_U.append(U_l); all_kl_from_uniform.append(kl_l)

    n = min(len(t) for t in all_T)
    avg = lambda data: [float(np.mean([d[i] for d in data if i < len(d)])) for i in range(n)]
    mean_T = avg(all_T)
    mean_P1 = avg(all_P1)
    mean_U = avg(all_U)
    mean_kl = avg(all_kl_from_uniform)

    rho_S, _ = stats.spearmanr(range(n), mean_T)
    rho_P1, _ = stats.spearmanr(range(n), mean_P1)

    # Onsager matrix
    dT = [mean_T[i+1] - mean_T[i] for i in range(n-1)]
    dU = [mean_U[i+1] - mean_U[i] for i in range(n-1)]
    if len(dT) >= 4:
        L_TU = float(np.corrcoef(dT, dU)[0, 1])
        L_UT = float(np.corrcoef(dU, dT)[0, 1])
    else:
        L_TU, L_UT = 0, 0

    return {
        'model': name,
        'n_layers': n_layers,
        'mean_T': mean_T,
        'mean_P1': mean_P1,
        'mean_U': mean_U,
        'mean_kl': mean_kl,
        'T_final': mean_T[-1],
        'T_initial': mean_T[0],
        'P1_final': mean_P1[-1],
        'rho_S': float(rho_S),
        'rho_P1': float(rho_P1),
        'onsager_asym': abs(L_TU - L_UT),
    }


def main():
    print("=" * 70)
    print("Phase 238: Instruct vs Base Model Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for model_id, name in MODELS:
        print(f"\n--- {name} ({model_id}) ---")
        try:
            model, tok = load_any_model(model_id, device=device)
            r = profile_model(model, tok, device, name)
            results[name] = r
            print(f"  {r['n_layers']}L, T: {r['T_initial']:.2f} -> {r['T_final']:.2f}")
            print(f"  rho_S={r['rho_S']:.3f}, rho_P1={r['rho_P1']:.3f}")
            print(f"  P1_final={r['P1_final']:.3f}")
            del model, tok
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {
        'Base-0.5B': '#3498db', 'Instruct-0.5B': '#e74c3c',
        'Base-1.5B': '#2ecc71', 'Instruct-1.5B': '#e67e22',
    }
    ls = {'Base-0.5B': '-', 'Instruct-0.5B': '--',
          'Base-1.5B': '-', 'Instruct-1.5B': '--'}

    # (a) T profiles
    for name, r in results.items():
        x = np.linspace(0, 1, len(r['mean_T']))
        axes[0, 0].plot(x, r['mean_T'], ls.get(name, '-'), color=colors.get(name, 'gray'),
                       lw=2, label=name)
    axes[0, 0].set_xlabel('Normalized Depth'); axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].set_title('(a) Temperature Profile')
    axes[0, 0].legend(fontsize=7)

    # (b) P1 profiles
    for name, r in results.items():
        x = np.linspace(0, 1, len(r['mean_P1']))
        axes[0, 1].plot(x, r['mean_P1'], ls.get(name, '-'), color=colors.get(name, 'gray'),
                       lw=2, label=name)
    axes[0, 1].set_xlabel('Normalized Depth'); axes[0, 1].set_ylabel('P1')
    axes[0, 1].set_title('(b) Order Parameter')
    axes[0, 1].legend(fontsize=7)

    # (c) KL from uniform
    for name, r in results.items():
        x = np.linspace(0, 1, len(r['mean_kl']))
        axes[0, 2].plot(x, r['mean_kl'], ls.get(name, '-'), color=colors.get(name, 'gray'),
                       lw=2, label=name)
    axes[0, 2].set_xlabel('Normalized Depth'); axes[0, 2].set_ylabel('KL(p || uniform)')
    axes[0, 2].set_title('(c) KL from Uniform')
    axes[0, 2].legend(fontsize=7)

    # (d) T_final comparison
    names = list(results.keys())
    T_finals = [results[n]['T_final'] for n in names]
    bars = axes[1, 0].bar(range(len(names)), T_finals,
                          color=[colors.get(n, 'gray') for n in names], alpha=0.8)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, fontsize=7, rotation=30)
    axes[1, 0].set_ylabel('T_final')
    axes[1, 0].set_title('(d) Final Temperature')

    # (e) Arrow strength comparison
    rho_S_vals = [results[n]['rho_S'] for n in names]
    rho_P1_vals = [results[n]['rho_P1'] for n in names]
    x = np.arange(len(names))
    width = 0.35
    axes[1, 1].bar(x - width/2, rho_S_vals, width, label='rho(S)',
                   color='steelblue', alpha=0.7)
    axes[1, 1].bar(x + width/2, rho_P1_vals, width, label='rho(P1)',
                   color='coral', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, fontsize=7, rotation=30)
    axes[1, 1].set_ylabel('Spearman rho')
    axes[1, 1].set_title('(e) Arrow Strength')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)

    # (f) Summary
    summary = "INSTRUCT vs BASE\n\n"
    for name, r in results.items():
        summary += f"{name:>16}:\n"
        summary += f"  T: {r['T_initial']:.1f} -> {r['T_final']:.1f}\n"
        summary += f"  P1={r['P1_final']:.3f}, rho_S={r['rho_S']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 238: Instruct vs Base Model Thermodynamics",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase238_instruct_vs_base')
    plt.close()
    save_results('phase238_instruct_vs_base', {
        'experiment': 'Instruct vs Base',
        'results': results,
    })


if __name__ == '__main__':
    main()
