# -*- coding: utf-8 -*-
"""
Phase 320: Unruh Effect -- Acceleration-Temperature Duality
=============================================================
The Unruh effect: an accelerating observer perceives thermal
radiation where an inertial observer sees vacuum.
T_Unruh = hbar * a / (2*pi*c*k_B)
In transformers: does "acceleration" (rapid change of hidden states)
create an effective temperature?
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


def measure_unruh(model, tok, prompt, device):
    """Measure Unruh effect: acceleration-temperature duality."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # Velocity: change of hidden state between layers
    velocities = []
    for li in range(n_layers):
        h1 = out.hidden_states[li][0, -1, :].float()
        h2 = out.hidden_states[li + 1][0, -1, :].float()
        v = float((h2 - h1).norm().item())
        velocities.append(v)

    # Acceleration: change of velocity
    accelerations = []
    for i in range(len(velocities) - 1):
        a = abs(velocities[i + 1] - velocities[i])
        accelerations.append(a)

    # Temperature at each layer
    temperatures = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        T = float(h.std().item())
        temperatures.append(T)

    # Unruh relation: T ~ a * constant
    # Test correlation between acceleration and temperature
    # Use midpoint temperature for each acceleration
    if len(accelerations) > 3:
        mid_temps = [(temperatures[i+1] + temperatures[i+2]) / 2 for i in range(len(accelerations))]
        slope, intercept, r, _, _ = stats.linregress(accelerations, mid_temps)
        unruh_r2 = r**2
        unruh_constant = slope  # T = slope * a + intercept
    else:
        unruh_r2 = 0
        unruh_constant = 0

    # Jerk (rate of change of acceleration)
    jerks = []
    for i in range(len(accelerations) - 1):
        j = abs(accelerations[i + 1] - accelerations[i])
        jerks.append(j)

    return {
        'velocities': [round(v, 4) for v in velocities],
        'accelerations': [round(a, 4) for a in accelerations],
        'temperatures': [round(t, 4) for t in temperatures],
        'unruh_r2': round(float(unruh_r2), 4),
        'unruh_constant': round(float(unruh_constant), 6),
        'mean_velocity': round(float(np.mean(velocities)), 4),
        'mean_acceleration': round(float(np.mean(accelerations)), 4),
    }


def main():
    print("=" * 70)
    print("Phase 320: Unruh Effect")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        ur_data = []
        for prompt in PROMPTS:
            u = measure_unruh(model, tok, prompt, device)
            ur_data.append(u)

        nv = len(ur_data[0]['velocities'])
        na = len(ur_data[0]['accelerations'])
        nt = len(ur_data[0]['temperatures'])

        avg_vel = [float(np.mean([u['velocities'][i] for u in ur_data])) for i in range(nv)]
        avg_acc = [float(np.mean([u['accelerations'][i] for u in ur_data])) for i in range(na)]
        avg_tmp = [float(np.mean([u['temperatures'][i] for u in ur_data])) for i in range(nt)]

        all_results[size] = {
            'n_layers': nv,
            'avg_velocity': [round(v, 4) for v in avg_vel],
            'avg_acceleration': [round(a, 4) for a in avg_acc],
            'avg_temperature': [round(t, 4) for t in avg_tmp],
            'unruh_r2': round(float(np.mean([u['unruh_r2'] for u in ur_data])), 4),
            'unruh_constant': round(float(np.mean([u['unruh_constant'] for u in ur_data])), 6),
            'mean_velocity': round(float(np.mean([u['mean_velocity'] for u in ur_data])), 4),
            'mean_acceleration': round(float(np.mean([u['mean_acceleration'] for u in ur_data])), 4),
        }
        print(f"  Unruh R2: {all_results[size]['unruh_r2']:.4f}")
        print(f"  Unruh constant: {all_results[size]['unruh_constant']:.6f}")
        print(f"  Mean velocity: {all_results[size]['mean_velocity']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_velocity'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Velocity ||dh/dl||')
    axes[0, 0].set_title('(a) Layer Velocity', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_acceleration'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Acceleration |dv/dl|')
    axes[0, 1].set_title('(b) Layer Acceleration', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['avg_temperature'], '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('Temperature')
    axes[0, 2].set_title('(c) Temperature Profile', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Acceleration vs Temperature scatter
    for size, data in all_results.items():
        acc = data['avg_acceleration']
        mid_t = [(data['avg_temperature'][i+1] + data['avg_temperature'][i+2]) / 2
                for i in range(len(acc))]
        axes[1, 0].scatter(acc, mid_t, color=colors[size], s=20, alpha=0.7, label=size)
    axes[1, 0].set_xlabel('Acceleration'); axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title('(d) Unruh: T vs a', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    sizes = list(all_results.keys())
    axes[1, 1].bar(sizes, [all_results[s]['unruh_r2'] for s in sizes],
                  color=[colors[s] for s in sizes])
    axes[1, 1].set_ylabel('R2'); axes[1, 1].set_title('(e) Unruh Relation R2', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    txt = "UNRUH EFFECT\n\n"
    txt += "T = c_U * a\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  R2 = {d['unruh_r2']:.3f}\n"
        txt += f"  c_U = {d['unruh_constant']:.5f}\n\n"
    txt += "High R2 -> T correlates\n"
    txt += "with acceleration"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 320: Unruh Effect -- Acceleration-Temperature Duality",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase320_unruh')
    plt.close()
    save_results('phase320_unruh', {'experiment': 'Unruh Effect', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
