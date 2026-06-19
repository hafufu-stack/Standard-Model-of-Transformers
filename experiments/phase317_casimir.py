# -*- coding: utf-8 -*-
"""
Phase 317: Casimir Effect -- Inter-Layer Force
=================================================
The Casimir effect: two conducting plates feel a force due to
vacuum fluctuations. In transformers:
- "Plates" = adjacent layers
- "Force" = change in energy when layers are brought together/apart
Measure the effective force between layers.
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
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_casimir(model, tok, prompt, device):
    """Measure Casimir-like forces between layers."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Energy at each layer
    layer_E = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        E = float((h ** 2).sum().item())
        layer_E.append(E)

    # Casimir force: F(l) = -dE/dl (negative gradient of energy)
    casimir_force = []
    for i in range(len(layer_E) - 1):
        F = -(layer_E[i + 1] - layer_E[i])
        casimir_force.append(float(F))

    # Casimir energy between layer pairs (interaction energy)
    # E_int(i,j) = cos(h_i, h_j) * sqrt(E_i * E_j) - E_i - E_j
    interaction_energy = []
    for i in range(n_layers):
        h1 = out.hidden_states[i][0, -1, :].float()
        h2 = out.hidden_states[i + 1][0, -1, :].float()
        cos = torch.nn.functional.cosine_similarity(
            h1.unsqueeze(0), h2.unsqueeze(0)).item()
        E_int = cos * np.sqrt(layer_E[i] * layer_E[i + 1]) - layer_E[i] - layer_E[i + 1]
        interaction_energy.append(float(E_int))

    # Force power law: F ~ 1/d^alpha (test with distance=1)
    # Check if force scales like 1/d^4 (Casimir) or other
    # Use force at different "effective distances" (cosine distance)
    distances = []
    forces = []
    for i in range(n_layers):
        h1 = out.hidden_states[i][0, -1, :].float()
        h2 = out.hidden_states[i + 1][0, -1, :].float()
        d = float(1.0 - torch.nn.functional.cosine_similarity(
            h1.unsqueeze(0), h2.unsqueeze(0)).item())
        distances.append(d + 1e-10)
        forces.append(abs(casimir_force[i]) + 1e-10)

    # Fit power law
    log_d = np.log(distances)
    log_f = np.log(forces)
    slope, intercept, r, _, _ = stats.linregress(log_d, log_f)

    return {
        'casimir_force': [round(f, 4) for f in casimir_force],
        'interaction_energy': [round(e, 4) for e in interaction_energy],
        'force_power_law_alpha': round(float(-slope), 4),
        'force_power_law_r2': round(float(r**2), 4),
        'mean_force': round(float(np.mean(np.abs(casimir_force))), 4),
        'mean_interaction': round(float(np.mean(interaction_energy)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 317: Casimir Effect")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        cas_data = []
        for prompt in PROMPTS:
            c = measure_casimir(model, tok, prompt, device)
            cas_data.append(c)

        n = len(cas_data[0]['casimir_force'])
        avg_force = [float(np.mean([c['casimir_force'][i] for c in cas_data])) for i in range(n)]
        avg_int = [float(np.mean([c['interaction_energy'][i] for c in cas_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n,
            'avg_casimir_force': [round(f, 4) for f in avg_force],
            'avg_interaction_energy': [round(e, 4) for e in avg_int],
            'force_power_law_alpha': round(float(np.mean([c['force_power_law_alpha'] for c in cas_data])), 4),
            'mean_force': round(float(np.mean([c['mean_force'] for c in cas_data])), 4),
            'mean_interaction': round(float(np.mean([c['mean_interaction'] for c in cas_data])), 4),
        }
        print(f"  Mean |F|: {all_results[size]['mean_force']:.4f}")
        print(f"  Power law alpha: {all_results[size]['force_power_law_alpha']:.4f}")
        print(f"  Mean interaction: {all_results[size]['mean_interaction']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_casimir_force'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Casimir Force')
    axes[0, 0].set_title('(a) Casimir Force Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_interaction_energy'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Interaction Energy')
    axes[0, 1].set_title('(b) Inter-Layer Interaction', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[0, 2].bar(sizes, [all_results[s]['force_power_law_alpha'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[0, 2].axhline(4.0, color='gold', ls='--', lw=2, label='Casimir alpha=4')
    axes[0, 2].set_ylabel('alpha'); axes[0, 2].set_title('(c) Force Power Law', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "CASIMIR EFFECT\n\n"
    txt += "F ~ 1/d^alpha\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  alpha = {d['force_power_law_alpha']:.2f}\n"
        txt += f"  <|F|> = {d['mean_force']:.1f}\n\n"
    txt += "alpha=4: true Casimir\n"
    txt += "alpha=2: Coulomb"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 317: Casimir Effect -- Inter-Layer Force",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase317_casimir')
    plt.close()
    save_results('phase317_casimir', {'experiment': 'Casimir Effect', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
