# -*- coding: utf-8 -*-
"""
Phase 299: Navier-Stokes Analogy -- Complete Fluid Description
================================================================
Combine Mach, Reynolds, drag into a complete Navier-Stokes picture.
Compute: pressure, density, velocity, viscosity fields across layers.
Test if the Euler equation (inviscid) or full N-S applies.
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


def compute_fluid_fields(model, tok, prompt, device):
    """Compute all fluid dynamics fields for a single forward pass."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n = len(out.hidden_states) - 1
    D = out.hidden_states[0].shape[-1]

    # Fields per layer
    density = []     # rho = norm^2 / D
    pressure = []    # P = T * rho (ideal gas)
    velocity = []    # v = |delta h| / |h|
    temperature = [] # T = std(h)
    vorticity = []   # omega = curl-like: variance of per-position velocities
    energy = []      # E = 0.5 * rho * v^2 + P (Bernoulli)

    for li in range(n):
        h = out.hidden_states[li + 1][0].float()  # (seq, D)
        h_prev = out.hidden_states[li][0].float()

        # Density: norm^2 / D (energy density)
        rho = (h.norm() ** 2 / D).item()
        density.append(rho)

        # Temperature
        T = h[:, :].std().item()
        temperature.append(T)

        # Pressure (ideal gas law: P = rho * T)
        P = rho * T
        pressure.append(P)

        # Velocity: relative change
        delta = h - h_prev
        v = delta.norm().item() / (h.norm().item() + 1e-10)
        velocity.append(v)

        # Vorticity: variance of per-position velocities
        delta_norms = delta.norm(dim=-1)  # (seq,)
        omega = delta_norms.std().item() / (delta_norms.mean().item() + 1e-10)
        vorticity.append(omega)

        # Bernoulli energy: 0.5*rho*v^2 + P
        E = 0.5 * rho * v**2 + P
        energy.append(E)

    # Test conservation laws
    # Continuity: d(rho)/dl + d(rho*v)/dl = 0
    rho_arr = np.array(density)
    v_arr = np.array(velocity)
    flux = rho_arr * v_arr
    continuity_residual = np.diff(rho_arr) + np.diff(flux)

    # Bernoulli: E should be constant along streamlines
    E_arr = np.array(energy)
    bernoulli_cv = float(np.std(E_arr) / (np.mean(E_arr) + 1e-10))

    # Euler equation test: dv/dl = -(1/rho) * dP/dl
    # Left side
    dv = np.diff(v_arr)
    # Right side
    dP = np.diff(np.array(pressure))
    rho_mid = (rho_arr[:-1] + rho_arr[1:]) / 2
    euler_rhs = -dP / (rho_mid + 1e-10)

    # Correlation between left and right
    if len(dv) > 2:
        r_euler, p_euler = stats.pearsonr(dv, euler_rhs)
    else:
        r_euler, p_euler = 0, 1

    return {
        'density': [round(d, 4) for d in density],
        'pressure': [round(p, 4) for p in pressure],
        'velocity': [round(v, 6) for v in velocity],
        'temperature': [round(t, 4) for t in temperature],
        'vorticity': [round(o, 4) for o in vorticity],
        'energy': [round(e, 4) for e in energy],
        'continuity_residual_std': round(float(np.std(continuity_residual)), 4),
        'bernoulli_cv': round(bernoulli_cv, 4),
        'euler_r': round(float(r_euler), 4),
        'euler_p': round(float(p_euler), 6),
    }


def main():
    print("=" * 70)
    print("Phase 299: Navier-Stokes Analogy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        fields = []
        for prompt in PROMPTS:
            f = compute_fluid_fields(model, tok, prompt, device)
            fields.append(f)

        n = len(fields[0]['density'])
        avg_fields = {}
        for key in ['density', 'pressure', 'velocity', 'temperature', 'vorticity', 'energy']:
            avg_fields[key] = [float(np.mean([f[key][i] for f in fields])) for i in range(n)]

        all_results[size] = {
            'n_layers': n,
            'fields': {k: [round(v, 4) for v in vals] for k, vals in avg_fields.items()},
            'continuity_residual': round(float(np.mean([f['continuity_residual_std'] for f in fields])), 4),
            'bernoulli_cv': round(float(np.mean([f['bernoulli_cv'] for f in fields])), 4),
            'euler_r': round(float(np.mean([f['euler_r'] for f in fields])), 4),
            'euler_p': round(float(np.mean([f['euler_p'] for f in fields])), 6),
        }
        print(f"  Continuity residual: {all_results[size]['continuity_residual']:.4f}")
        print(f"  Bernoulli CV: {all_results[size]['bernoulli_cv']:.4f}")
        print(f"  Euler equation r: {all_results[size]['euler_r']:.4f} (p={all_results[size]['euler_p']:.4f})")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    field_names = ['density', 'pressure', 'velocity', 'temperature', 'vorticity', 'energy']
    field_labels = ['Density rho', 'Pressure P', 'Velocity v', 'Temperature T', 'Vorticity omega', 'Energy E']

    for idx, (fname, flabel) in enumerate(zip(field_names, field_labels)):
        row, col = divmod(idx, 3)
        for size, data in all_results.items():
            axes[row, col].plot(data['fields'][fname], '-', color=colors[size], lw=2, label=size)
        axes[row, col].set_xlabel('Layer')
        axes[row, col].set_ylabel(flabel)
        axes[row, col].set_title(f'({"abcdef"[idx]}) {flabel}', fontweight='bold')
        axes[row, col].legend(); axes[row, col].grid(alpha=0.3)

    fig.suptitle("Phase 299: Navier-Stokes Fluid Fields in Transformer",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase299_navier_stokes')
    plt.close()

    save_results('phase299_navier_stokes', {
        'experiment': 'Navier-Stokes Analogy',
        'results': {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
