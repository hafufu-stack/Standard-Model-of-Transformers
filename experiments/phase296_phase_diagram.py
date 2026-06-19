# -*- coding: utf-8 -*-
"""
Phase 296: Complete Thermodynamic Phase Diagram
=================================================
Map the full T-PR phase space across layers, prompts, and model sizes.
Identify phase boundaries, critical points, and universality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "The most effective approach to solving climate change is",
    "Once upon a time in a kingdom far away",
    "Machine learning models can classify data by",
    "The chemical composition of water molecules is",
    "The speed of light is constant in all reference frames",
    "Artificial intelligence will transform how we live and work",
    "Evolution explains the diversity of life on Earth through",
    "The structure of the atom includes a nucleus surrounded by",
    "The human brain contains approximately one hundred billion neurons",
    "The periodic table organizes chemical elements by their properties",
    "Democracy is a system of government in which power is held by",
    "The theory of evolution was proposed by Charles Darwin in his",
    "Quantum computing uses qubits that can exist in multiple states",
    "The laws of thermodynamics govern all energy transformations",
]


def main():
    print("=" * 70)
    print("Phase 296: Complete Thermodynamic Phase Diagram")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        n_layers = len(model.model.layers)

        all_T = []
        all_PR = []
        all_P1T = []
        all_layers = []
        all_prompt_idx = []

        for pi, prompt in enumerate(PROMPTS):
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            for li in range(n_layers + 1):
                h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
                T = float(np.std(h))
                h_sq = h ** 2
                h_p = h_sq / (np.sum(h_sq) + 1e-15)
                PR = float(1.0 / (np.sum(h_p ** 2) + 1e-15))
                P1 = float(np.max(h_p))
                P1T = P1 * T

                all_T.append(T)
                all_PR.append(PR)
                all_P1T.append(P1T)
                all_layers.append(li)
                all_prompt_idx.append(pi)

        all_T = np.array(all_T)
        all_PR = np.array(all_PR)
        all_P1T = np.array(all_P1T)
        all_layers = np.array(all_layers)

        # Phase regions
        # High-T, Low-PR = "gas" (early layers)
        # Low-T, High-PR = "condensate" (late layers)
        # Transition region
        median_T = np.median(all_T)
        median_PR = np.median(all_PR)

        gas_mask = (all_T > median_T) & (all_PR < median_PR)
        cond_mask = (all_T < median_T) & (all_PR > median_PR)
        trans_mask = ~gas_mask & ~cond_mask

        # Correlation: T vs PR
        r_t_pr, p_t_pr = stats.pearsonr(all_T, all_PR)

        # Equation of state: PR = f(T)
        slope_eos, int_eos, r_eos, _, _ = stats.linregress(np.log(all_T + 1e-10), np.log(all_PR + 1e-10))

        all_results[size] = {
            'n_layers': n_layers,
            'n_points': len(all_T),
            'T_range': [round(float(all_T.min()), 4), round(float(all_T.max()), 4)],
            'PR_range': [round(float(all_PR.min()), 1), round(float(all_PR.max()), 1)],
            'r_T_PR': round(float(r_t_pr), 4),
            'eos_exponent': round(float(slope_eos), 4),
            'eos_R2': round(float(r_eos**2), 4),
            'n_gas': int(gas_mask.sum()),
            'n_condensate': int(cond_mask.sum()),
            'n_transition': int(trans_mask.sum()),
            'P1T_mean': round(float(np.mean(all_P1T)), 4),
            'P1T_std': round(float(np.std(all_P1T)), 4),
            # Store raw data for plotting
            '_T': all_T.tolist(),
            '_PR': all_PR.tolist(),
            '_layers': all_layers.tolist(),
            '_P1T': all_P1T.tolist(),
        }
        print(f"  T range: [{all_T.min():.3f}, {all_T.max():.3f}]")
        print(f"  PR range: [{all_PR.min():.0f}, {all_PR.max():.0f}]")
        print(f"  T-PR correlation: r={r_t_pr:.4f}")
        print(f"  EoS: PR ~ T^{slope_eos:.2f} (R2={r_eos**2:.4f})")
        print(f"  Phases: Gas={gas_mask.sum()}, Condensate={cond_mask.sum()}, Transition={trans_mask.sum()}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_size = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    # (a) Phase diagram: T vs PR colored by layer
    for si, (size, data) in enumerate(all_results.items()):
        ax = axes[0, si]
        T = np.array(data['_T'])
        PR = np.array(data['_PR'])
        layers = np.array(data['_layers'])
        sc = ax.scatter(T, PR, c=layers, cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel('Temperature T')
        ax.set_ylabel('Participation Ratio PR')
        ax.set_title(f'({"a" if si==0 else "b"}) Phase Diagram: {size}', fontweight='bold')
        plt.colorbar(sc, ax=ax, label='Layer')
        ax.grid(alpha=0.3)

    # (c) T vs PR overlay
    for size, data in all_results.items():
        T = np.array(data['_T'])
        PR = np.array(data['_PR'])
        axes[0, 2].scatter(T, PR, s=5, alpha=0.3, color=colors_size[size], label=size)
    axes[0, 2].set_xlabel('Temperature T')
    axes[0, 2].set_ylabel('Participation Ratio PR')
    axes[0, 2].set_title('(c) Phase Diagram Overlay', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) P1*T along layers
    for size, data in all_results.items():
        P1T = np.array(data['_P1T'])
        layers = np.array(data['_layers'])
        # Average per layer
        unique_layers = sorted(set(layers))
        avg_p1t = [float(np.mean(P1T[layers == l])) for l in unique_layers]
        axes[1, 0].plot(unique_layers, avg_p1t, '-', color=colors_size[size], lw=2, label=size)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('P1 * T')
    axes[1, 0].set_title('(d) P1*T Layer Profile', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Equation of state: log-log T vs PR
    for size, data in all_results.items():
        T = np.array(data['_T'])
        PR = np.array(data['_PR'])
        valid = (T > 0) & (PR > 0)
        axes[1, 1].loglog(T[valid], PR[valid], '.', color=colors_size[size],
                         alpha=0.3, markersize=3, label=f"{size} (a={data['eos_exponent']:.2f})")
    axes[1, 1].set_xlabel('Temperature T (log)')
    axes[1, 1].set_ylabel('PR (log)')
    axes[1, 1].set_title('(e) Equation of State', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "THERMODYNAMIC PHASE DIAGRAM\n\n"
    for size, data in all_results.items():
        txt += f"{size}:\n"
        txt += f"  T-PR corr: r={data['r_T_PR']:.3f}\n"
        txt += f"  EoS: PR ~ T^{data['eos_exponent']:.2f}\n"
        txt += f"  P1T = {data['P1T_mean']:.3f}\n\n"
    txt += "Phase transitions:\n"
    txt += "  Gas (hot, low PR)\n"
    txt += "  -> Condensate (cold, high PR)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 296: Complete Thermodynamic Phase Diagram",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase296_phase_diagram')
    plt.close()

    # Save without raw data (too large)
    save_data = {
        'experiment': 'Complete Thermodynamic Phase Diagram',
        'results': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                   for k, v in all_results.items()},
    }
    save_results('phase296_phase_diagram', save_data)

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
