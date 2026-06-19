# -*- coding: utf-8 -*-
"""
Phase 316: Vacuum Energy -- Zero-Point Fluctuations
=====================================================
Vacuum energy = ground state energy of quantum fields.
In transformers: what is the "zero-point" energy of hidden states?
Measure the energy floor that exists even in the absence of
meaningful input (random/null tokens).
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

REAL_PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
]

VACUUM_PROMPTS = [
    "                                              ",
    "a a a a a a a a a a a a a a a a a a a a a a a",
    "the the the the the the the the the the the the",
]


def measure_vacuum_energy(model, tok, prompts_real, prompts_vacuum, device):
    """Compare energy states of real vs vacuum inputs."""
    results = {'real': [], 'vacuum': []}

    for category, prompts in [('real', prompts_real), ('vacuum', prompts_vacuum)]:
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)

            n_layers = len(out.hidden_states) - 1
            layer_energies = []
            layer_temps = []

            for li in range(n_layers + 1):
                h = out.hidden_states[li][0, -1, :].float()
                E = float((h ** 2).sum().item())
                T = float(h.std().item())
                layer_energies.append(E)
                layer_temps.append(T)

            results[category].append({
                'energies': layer_energies,
                'temps': layer_temps,
                'total_E': float(np.sum(layer_energies)),
                'mean_E': float(np.mean(layer_energies)),
            })

    # Vacuum energy = mean energy of vacuum states
    vac_E = float(np.mean([r['mean_E'] for r in results['vacuum']]))
    real_E = float(np.mean([r['mean_E'] for r in results['real']]))
    vacuum_fraction = vac_E / (real_E + 1e-10)

    # Layer-by-layer vacuum vs real
    n = len(results['real'][0]['energies'])
    vac_profile = [float(np.mean([r['energies'][i] for r in results['vacuum']])) for i in range(n)]
    real_profile = [float(np.mean([r['energies'][i] for r in results['real']])) for i in range(n)]
    ratio_profile = [v / (r + 1e-10) for v, r in zip(vac_profile, real_profile)]

    return {
        'vacuum_E': round(vac_E, 4),
        'real_E': round(real_E, 4),
        'vacuum_fraction': round(vacuum_fraction, 4),
        'vac_profile': [round(v, 4) for v in vac_profile],
        'real_profile': [round(r, 4) for r in real_profile],
        'ratio_profile': [round(r, 4) for r in ratio_profile],
    }


def main():
    print("=" * 70)
    print("Phase 316: Vacuum Energy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        data = measure_vacuum_energy(model, tok, REAL_PROMPTS, VACUUM_PROMPTS, device)
        all_results[size] = data
        print(f"  Vacuum E: {data['vacuum_E']:.2f}")
        print(f"  Real E: {data['real_E']:.2f}")
        print(f"  Vacuum fraction: {data['vacuum_fraction']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['real_profile'], '-', color=colors[size], lw=2, label=f'{size} real')
        axes[0, 0].plot(data['vac_profile'], '--', color=colors[size], lw=2, label=f'{size} vacuum', alpha=0.7)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_title('(a) Real vs Vacuum Energy', fontweight='bold')
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['ratio_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(1.0, color='gold', ls='--', lw=1)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Vacuum / Real')
    axes[0, 1].set_title('(b) Energy Ratio', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['vacuum_fraction'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Vacuum Fraction'); axes[0, 2].set_title('(c) Vacuum/Real Ratio', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "VACUUM ENERGY\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  E_vac = {d['vacuum_E']:.1f}\n"
        txt += f"  E_real = {d['real_E']:.1f}\n"
        txt += f"  ratio = {d['vacuum_fraction']:.3f}\n\n"
    txt += "High ratio -> large vacuum energy\n"
    txt += "(cosmological constant problem)"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 316: Vacuum Energy -- Zero-Point Fluctuations",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase316_vacuum')
    plt.close()
    save_results('phase316_vacuum', {'experiment': 'Vacuum Energy', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
