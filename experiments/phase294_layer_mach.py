# -*- coding: utf-8 -*-
"""
Phase 294: Layer-Resolved Mach Profile
========================================
Phase 290 measured global Mach number. But does Mach vary within a model?
Measure Mach number at each layer to find:
- Where does the model go supersonic?
- Is there a "throat" (minimum area) like a Laval nozzle?
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
]


def measure_layer_mach(model, tok, prompt, device):
    """Measure Mach number at each layer."""
    inp = tok(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    seq_len = out.hidden_states[0].shape[1]

    # Speed of sound per layer: perturbation spread rate
    # Use inter-position correlation decay as proxy
    layer_c_sound = []
    layer_c_light = []

    for li in range(n_layers):
        h = out.hidden_states[li + 1][0].float()  # (seq, D)
        if h.shape[0] < 3:
            layer_c_sound.append(0)
            layer_c_light.append(1)
            continue

        # c_sound: how quickly info spreads between positions at this layer
        # Measure via auto-correlation length
        last = h[-1]
        sims = torch.nn.functional.cosine_similarity(h[:-1], last.unsqueeze(0), dim=-1)
        # Correlation length = how many positions have sim > 0.5
        corr_length = (sims > 0.5).sum().item()
        layer_c_light.append(corr_length)

        # c_sound: local perturbation speed
        # Use norm gradient as proxy
        norms = h.norm(dim=-1)  # (seq,)
        if norms.shape[0] > 1:
            norm_grad = torch.diff(norms).abs().mean().item()
            layer_c_sound.append(norm_grad)
        else:
            layer_c_sound.append(0)

    # Layer Mach = c_sound / c_light (with normalization)
    max_cs = max(layer_c_sound) if max(layer_c_sound) > 0 else 1
    max_cl = max(layer_c_light) if max(layer_c_light) > 0 else 1
    # Normalize both to [0, 1] for fair comparison
    layer_mach = []
    for cs, cl in zip(layer_c_sound, layer_c_light):
        cs_norm = cs / max_cs
        cl_norm = cl / max_cl if cl > 0 else 1
        mach = cs_norm / (cl_norm + 1e-10)
        layer_mach.append(float(mach))

    # Find sonic transition layer
    sonic_layer = None
    for i in range(len(layer_mach) - 1):
        if layer_mach[i] < 1.0 and layer_mach[i+1] >= 1.0:
            sonic_layer = i
            break
        elif layer_mach[i] >= 1.0 and layer_mach[i+1] < 1.0:
            sonic_layer = i
            break

    return {
        'layer_c_sound': [round(c, 4) for c in layer_c_sound],
        'layer_c_light': [round(c, 4) for c in layer_c_light],
        'layer_mach': [round(m, 4) for m in layer_mach],
        'sonic_layer': sonic_layer,
        'max_mach': round(float(max(layer_mach)), 4),
        'max_mach_layer': int(np.argmax(layer_mach)),
        'min_mach': round(float(min(layer_mach)), 4),
        'min_mach_layer': int(np.argmin(layer_mach)),
    }


def main():
    print("=" * 70)
    print("Phase 294: Layer-Resolved Mach Profile")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B', '7B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        profiles = []
        for prompt in PROMPTS:
            p = measure_layer_mach(model, tok, prompt, device)
            profiles.append(p)

        # Average across prompts
        n = len(profiles[0]['layer_mach'])
        avg_mach = [float(np.mean([p['layer_mach'][i] for p in profiles if i < len(p['layer_mach'])]))
                   for i in range(n)]

        # Find supersonic regions
        supersonic_layers = [i for i, m in enumerate(avg_mach) if m > 1.0]
        subsonic_layers = [i for i, m in enumerate(avg_mach) if m <= 1.0]

        all_results[size] = {
            'n_layers': n,
            'avg_mach_profile': [round(m, 4) for m in avg_mach],
            'max_mach': round(float(max(avg_mach)), 4),
            'max_mach_layer': int(np.argmax(avg_mach)),
            'min_mach': round(float(min(avg_mach)), 4),
            'min_mach_layer': int(np.argmin(avg_mach)),
            'n_supersonic_layers': len(supersonic_layers),
            'supersonic_fraction': round(len(supersonic_layers) / n, 4),
            'supersonic_layers': supersonic_layers[:10],
        }
        print(f"  Max Mach: {all_results[size]['max_mach']:.3f} at L{all_results[size]['max_mach_layer']}")
        print(f"  Supersonic layers: {len(supersonic_layers)}/{n} ({all_results[size]['supersonic_fraction']*100:.0f}%)")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c', '7B': '#2ecc71'}

    # (a) Mach profiles
    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_mach_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(1.0, color='gold', ls='--', lw=2, label='Mach = 1')
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Local Mach Number')
    axes[0, 0].set_title('(a) Layer-Resolved Mach Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Normalized depth
    for size, data in all_results.items():
        n = len(data['avg_mach_profile'])
        x = np.linspace(0, 1, n)
        axes[0, 1].plot(x, data['avg_mach_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(1.0, color='gold', ls='--', lw=2)
    axes[0, 1].set_xlabel('Normalized Depth'); axes[0, 1].set_ylabel('Mach')
    axes[0, 1].set_title('(b) Normalized Mach Profile', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Supersonic fraction
    sizes = list(all_results.keys())
    fracs = [all_results[s]['supersonic_fraction'] for s in sizes]
    axes[0, 2].bar(sizes, fracs, color=[colors[s] for s in sizes])
    axes[0, 2].set_ylabel('Supersonic Fraction')
    axes[0, 2].set_title('(c) Fraction of Supersonic Layers', fontweight='bold')
    axes[0, 2].grid(alpha=0.3)

    # (d) Max Mach vs model
    max_machs = [all_results[s]['max_mach'] for s in sizes]
    axes[1, 0].bar(sizes, max_machs, color=[colors[s] for s in sizes])
    axes[1, 0].axhline(1.0, color='gold', ls='--', lw=1)
    axes[1, 0].set_ylabel('Max Local Mach')
    axes[1, 0].set_title('(d) Peak Mach per Model', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) Mach at key depths
    depths = [0.25, 0.5, 0.75]
    for size, data in all_results.items():
        n = len(data['avg_mach_profile'])
        vals = [data['avg_mach_profile'][int(d * (n-1))] for d in depths]
        axes[1, 1].plot(depths, vals, 'o-', color=colors[size], lw=2, markersize=8, label=size)
    axes[1, 1].axhline(1.0, color='gold', ls='--', lw=1)
    axes[1, 1].set_xlabel('Depth Fraction')
    axes[1, 1].set_ylabel('Mach')
    axes[1, 1].set_title('(e) Mach at Key Depths', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "LAYER-RESOLVED MACH\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  Max M = {d['max_mach']:.2f} at L{d['max_mach_layer']}\n"
        txt += f"  Supersonic: {d['n_supersonic_layers']}/{d['n_layers']}\n\n"
    txt += "Laval nozzle analogy:\n"
    txt += "Subsonic -> throat -> supersonic"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 294: Layer-Resolved Mach Profile (Laval Nozzle?)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase294_layer_mach')
    plt.close()

    save_results('phase294_layer_mach', {
        'experiment': 'Layer-Resolved Mach Profile',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
