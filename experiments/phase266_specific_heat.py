# -*- coding: utf-8 -*-
"""
Phase 266: Specific Heat and Phase Transitions
=================================================
In stat-mech, specific heat C = dU/dT identifies phase transitions (C diverges).
For transformers, we define an analogous quantity:
  C_layer = dT_sm / d_layer  (rate of temperature change per layer)

If there's a "phase transition" at some depth, C_layer should show a sharp peak
or sign change. This would identify the critical layer where the transformer
shifts from "exploration" (high T) to "exploitation" (low T).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, signal
from utils import load_model, save_results, save_figure

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
    "Machine learning discovers hidden patterns",
    "Purple elephants calculated the square root of",
]


def measure_specific_heat(model, tok, device, model_name):
    """Compute layer-resolved specific heat C = dT/dl."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    all_T, all_P1, all_U = [], [], []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_l, P1_l, U_l = [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U = float(h.norm().item())

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1 = float(probs.max().item())
            T_sm = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T_sm): T_sm = 0

            T_l.append(T_sm)
            P1_l.append(P1)
            U_l.append(U)

        all_T.append(T_l)
        all_P1.append(P1_l)
        all_U.append(U_l)

    n = min(len(t) for t in all_T)
    avg = lambda d: np.array([float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)])

    mean_T = avg(all_T)
    mean_P1 = avg(all_P1)
    mean_U = avg(all_U)
    mean_PRT = mean_P1 * mean_T

    # Specific heat: C = dT/dl (finite difference)
    C_T = np.gradient(mean_T)
    C_U = np.gradient(mean_U)
    C_PRT = np.gradient(mean_PRT)

    # Absolute specific heat (magnitude of change)
    abs_C = np.abs(C_T)

    # Find critical layer (max |dT/dl|)
    critical_idx = int(np.argmax(abs_C[1:]) + 1)  # Skip layer 0
    critical_depth = round(critical_idx / (n - 1), 3)

    # Sign changes in dT/dl (phase transition markers)
    sign_changes = []
    for i in range(1, len(C_T) - 1):
        if C_T[i] * C_T[i+1] < 0:
            sign_changes.append(i)

    # Susceptibility: variance of T across prompts at each layer
    chi = np.array([float(np.var([all_T[p][i] for p in range(len(all_T))])) for i in range(n)])

    return {
        'model': model_name,
        'n_layers': n,
        'mean_T': mean_T.tolist(),
        'mean_P1': mean_P1.tolist(),
        'mean_U': mean_U.tolist(),
        'mean_PRT': mean_PRT.tolist(),
        'C_T': C_T.tolist(),
        'C_U': C_U.tolist(),
        'C_PRT': C_PRT.tolist(),
        'chi': chi.tolist(),
        'critical_idx': critical_idx,
        'critical_depth': critical_depth,
        'sign_changes': sign_changes,
        'max_C': round(float(abs_C[1:].max()), 4),
    }


def main():
    print("=" * 70)
    print("Phase 266: Specific Heat and Phase Transitions")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = measure_specific_heat(model, tok, device, size)
        results[size] = r
        print(f"  Critical layer: {r['critical_idx']}/{r['n_layers']-1} (depth={r['critical_depth']})")
        print(f"  Max |C_T| = {r['max_C']:.4f}")
        print(f"  Sign changes at layers: {r['sign_changes']}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, r in results.items():
        c = colors[size]
        n = r['n_layers']
        x = np.arange(n)

        # (a) Temperature profile with critical layer
        axes[0, 0].plot(x, r['mean_T'], '-', color=c, lw=2, label=f'{size}')
        axes[0, 0].axvline(r['critical_idx'], color=c, ls='--', alpha=0.5,
                          label=f'Critical ({size}): L{r["critical_idx"]}')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('T_sm')
    axes[0, 0].set_title('(a) Temperature Profile', fontweight='bold')
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    # (b) Specific heat C_T = dT/dl
    for size, r in results.items():
        c = colors[size]
        axes[0, 1].plot(range(len(r['C_T'])), r['C_T'], '-', color=c, lw=2, label=size)
        axes[0, 1].axhline(0, color='gray', ls='--', lw=0.5)
        for sc in r['sign_changes']:
            axes[0, 1].axvline(sc, color=c, ls=':', alpha=0.3)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('C_T = dT/dl')
    axes[0, 1].set_title('(b) Specific Heat (Temperature)', fontweight='bold')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # (c) Energy profile + C_U
    for size, r in results.items():
        c = colors[size]
        axes[0, 2].plot(range(len(r['mean_U'])), r['mean_U'], '-', color=c, lw=2, label=f'U ({size})')
        ax2 = axes[0, 2].twinx()
        ax2.plot(range(len(r['C_U'])), r['C_U'], '--', color=c, lw=1, alpha=0.5)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('U (L2 norm, solid)')
    ax2.set_ylabel('dU/dl (dashed)')
    axes[0, 2].set_title('(c) Internal Energy', fontweight='bold')
    axes[0, 2].legend(fontsize=7); axes[0, 2].grid(alpha=0.3)

    # (d) P1*T + its specific heat
    for size, r in results.items():
        c = colors[size]
        axes[1, 0].plot(range(len(r['mean_PRT'])), r['mean_PRT'], '-', color=c, lw=2, label=f'P1*T ({size})')
        ax3 = axes[1, 0].twinx()
        ax3.plot(range(len(r['C_PRT'])), r['C_PRT'], '--', color=c, lw=1, alpha=0.5)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('P1*T (solid)')
    ax3.set_ylabel('d(P1*T)/dl (dashed)')
    axes[1, 0].set_title('(d) P1*T Specific Heat', fontweight='bold')
    axes[1, 0].legend(fontsize=7); axes[1, 0].grid(alpha=0.3)

    # (e) Susceptibility chi
    for size, r in results.items():
        c = colors[size]
        axes[1, 1].plot(range(len(r['chi'])), r['chi'], '-', color=c, lw=2, label=size)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('chi = Var(T across prompts)')
    axes[1, 1].set_title('(e) Susceptibility', fontweight='bold')
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "PHASE TRANSITIONS\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  Critical layer: L{r['critical_idx']}/{r['n_layers']-1}\n"
        summary += f"  Depth: {r['critical_depth']:.1%}\n"
        summary += f"  Max |C_T| = {r['max_C']:.4f}\n"
        summary += f"  Sign changes: {r['sign_changes']}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 266: Specific Heat and Phase Transitions in Transformers",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase266_specific_heat')
    plt.close()

    save_results('phase266_specific_heat', {
        'experiment': 'Specific Heat and Phase Transitions',
        'results': {k: {kk: vv for kk, vv in v.items()
                       if kk not in ('mean_T', 'mean_P1', 'mean_U', 'mean_PRT',
                                     'C_T', 'C_U', 'C_PRT', 'chi')}
                   for k, v in results.items()},
        'profiles': {k: {'T': v['mean_T'], 'C_T': v['C_T'], 'chi': v['chi']}
                    for k, v in results.items()},
    })


if __name__ == '__main__':
    main()
