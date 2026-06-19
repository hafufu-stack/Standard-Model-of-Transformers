# -*- coding: utf-8 -*-
"""
Phase 298: Reynolds Number -- Turbulence in Information Flow
==============================================================
Reynolds number Re = (inertial forces) / (viscous forces).
In fluid dynamics, Re > ~2300 => turbulent flow.
Define transformer Re as:
  Re = (information velocity * characteristic length) / (dissipation rate)
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


def measure_reynolds(model, tok, prompt, device):
    """Measure Reynolds number at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    D = out.hidden_states[0].shape[-1]

    layer_Re = []
    layer_velocity = []
    layer_viscosity = []

    for li in range(n_layers):
        h_before = out.hidden_states[li][0].float()
        h_after = out.hidden_states[li + 1][0].float()

        # Velocity = norm of change (inertial)
        delta_h = h_after - h_before
        velocity = delta_h.norm(dim=-1).mean().item()

        # Characteristic length = effective dimension
        _, s, _ = torch.linalg.svd(h_after, full_matrices=False)
        s_norm = s / (s.sum() + 1e-10)
        eff_dim = torch.exp(-(s_norm * torch.log(s_norm + 1e-15)).sum()).item()

        # Viscosity = smoothness of flow (inverse of local turbulence)
        # Measure via variance of per-position changes
        delta_norms = delta_h.norm(dim=-1)  # (seq,)
        viscosity = 1.0 / (delta_norms.std().item() + 1e-10)

        Re = velocity * eff_dim / (1.0 / viscosity + 1e-10)

        layer_Re.append(float(Re))
        layer_velocity.append(float(velocity))
        layer_viscosity.append(float(1.0 / viscosity))

    # Turbulence detection: Re variance across layers
    Re_arr = np.array(layer_Re)
    turbulent_layers = (Re_arr > np.median(Re_arr) * 2).sum()

    return {
        'layer_Re': [round(r, 2) for r in layer_Re],
        'layer_velocity': [round(v, 4) for v in layer_velocity],
        'layer_viscosity': [round(v, 4) for v in layer_viscosity],
        'mean_Re': round(float(np.mean(layer_Re)), 2),
        'max_Re': round(float(np.max(layer_Re)), 2),
        'max_Re_layer': int(np.argmax(layer_Re)),
        'turbulent_layers': int(turbulent_layers),
        'Re_std': round(float(np.std(layer_Re)), 2),
    }


def main():
    print("=" * 70)
    print("Phase 298: Reynolds Number -- Turbulence in Info Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B', '7B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        re_data = []
        for prompt in PROMPTS:
            r = measure_reynolds(model, tok, prompt, device)
            re_data.append(r)

        n = len(re_data[0]['layer_Re'])
        avg_Re = [float(np.mean([r['layer_Re'][i] for r in re_data])) for i in range(n)]
        avg_vel = [float(np.mean([r['layer_velocity'][i] for r in re_data])) for i in range(n)]
        avg_visc = [float(np.mean([r['layer_viscosity'][i] for r in re_data])) for i in range(n)]

        all_results[size] = {
            'n_layers': n,
            'avg_Re': [round(r, 2) for r in avg_Re],
            'avg_velocity': [round(v, 4) for v in avg_vel],
            'avg_viscosity': [round(v, 4) for v in avg_visc],
            'mean_Re': round(float(np.mean(avg_Re)), 2),
            'max_Re': round(float(np.max(avg_Re)), 2),
            'max_Re_layer': int(np.argmax(avg_Re)),
            'Re_std': round(float(np.std(avg_Re)), 2),
        }
        print(f"  Mean Re = {all_results[size]['mean_Re']:.1f}")
        print(f"  Max Re = {all_results[size]['max_Re']:.1f} at L{all_results[size]['max_Re_layer']}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) Reynolds profiles
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_Re'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Reynolds Number')
    axes[0, 0].set_title('(a) Reynolds Number Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Velocity profiles
    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_velocity'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Information Velocity')
    axes[0, 1].set_title('(b) Velocity Profile', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Viscosity profiles
    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_viscosity'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer')
    axes[0, 2].set_ylabel('Viscosity (turbulence)')
    axes[0, 2].set_title('(c) Viscosity Profile', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Mean Re vs model
    sizes = list(all_results.keys())
    mean_res = [all_results[s]['mean_Re'] for s in sizes]
    axes[1, 0].bar(sizes, mean_res, color=[colors[s] for s in sizes])
    axes[1, 0].set_ylabel('Mean Reynolds Number')
    axes[1, 0].set_title('(d) Mean Re vs Scale', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) Re normalized
    for size, data in all_results.items():
        n = len(data['avg_Re'])
        x = np.linspace(0, 1, n)
        axes[1, 1].plot(x, data['avg_Re'], '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Normalized Depth')
    axes[1, 1].set_ylabel('Reynolds Number')
    axes[1, 1].set_title('(e) Re vs Normalized Depth', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "REYNOLDS NUMBER\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Mean Re = {d['mean_Re']:.0f}\n"
        txt += f"  Max Re = {d['max_Re']:.0f} at L{d['max_Re_layer']}\n\n"
    txt += "Re > 2300: turbulent\n"
    txt += "Re < 2300: laminar"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 298: Reynolds Number -- Turbulence in Information Flow",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase298_reynolds')
    plt.close()

    save_results('phase298_reynolds', {
        'experiment': 'Reynolds Number - Turbulence Detection',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
