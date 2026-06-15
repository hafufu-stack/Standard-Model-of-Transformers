# -*- coding: utf-8 -*-
"""
Phase 179: Brownian Ratchet (Waste Heat Recycling)
Collect "waste" from LayerNorm residuals and reinject into next layer.
Test if recycled thermal noise acts as stochastic resonance fuel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

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
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def run_with_ratchet(model, tok, prompt, device, ratchet_gain=0.01):
    """Run forward pass with Brownian ratchet: reinject LayerNorm waste."""
    inp = tok(prompt, return_tensors='pt').to(device)
    n_layers = len(model.model.layers)

    # Storage for waste heat and measurements
    waste_heat = {}
    layer_U = []
    layer_T = []

    def make_ratchet_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            h_float = h.float()

            # Collect waste: difference between pre-norm and post-norm magnitude
            norm_before = h_float.norm(dim=-1, keepdim=True)
            waste = h_float - h_float / (norm_before + 1e-8) * norm_before.mean()
            waste_energy = waste.norm().item()
            waste_heat[layer_idx] = waste_energy

            # Reinject waste from PREVIOUS layer (if available)
            if layer_idx > 0 and (layer_idx - 1) in waste_heat:
                prev_waste = waste_heat[layer_idx - 1]
                # Ratchet: only add noise in the "forward" direction (positive gradient)
                noise = torch.randn_like(h_float) * ratchet_gain * prev_waste
                # Gate: only keep components that align with existing direction
                cos_sim = torch.nn.functional.cosine_similarity(noise, h_float, dim=-1, eps=1e-8)
                gate = (cos_sim > 0).float().unsqueeze(-1)
                h_mod = h_float + noise * gate
                h_mod = torch.nan_to_num(h_mod, nan=0.0)
                result = h_mod.to(h.dtype)
                if isinstance(output, tuple):
                    return (result,) + output[1:]
                return result

        return hook

    # Register hooks
    handles = []
    for i in range(n_layers):
        h = model.model.layers[i].register_forward_hook(make_ratchet_hook(i))
        handles.append(h)

    # Forward pass with ratchet
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Remove hooks
    for h in handles:
        h.remove()

    # Measure thermodynamics from hidden states
    results = []
    for li in range(len(out.hidden_states)):
        hs = out.hidden_states[li]
        h = hs[0, -1, :].float()
        U = h.norm().item()
        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        results.append({'U': U, 'T': T if not np.isnan(T) else 0.0})

    return results, waste_heat, out


def main():
    print("=" * 70)
    print("Phase 179: Brownian Ratchet")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    gains = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    all_results = {}

    for gain in gains:
        print(f"\n--- Ratchet gain = {gain} ---")
        U_all, T_all, waste_all = [], [], []

        for prompt in PROMPTS:
            if gain == 0.0:
                # Baseline: no ratchet
                thermo, out = measure_full_thermodynamics(model, tok, prompt, device)
                U_vals = [r['U'] for r in thermo]
                T_vals = [r['T'] for r in thermo]
                waste = {}
            else:
                res, waste, out = run_with_ratchet(model, tok, prompt, device, ratchet_gain=gain)
                U_vals = [r['U'] for r in res]
                T_vals = [r['T'] for r in res]

            U_all.append(U_vals)
            T_all.append(T_vals)
            if waste:
                waste_all.append(list(waste.values()))

        U_mean = np.mean(U_all, axis=0)
        T_mean = np.mean(T_all, axis=0)

        # Carnot efficiency
        T_hot = np.mean(T_mean[:3])
        T_cold = np.mean(T_mean[-3:])
        eta = 1 - T_cold / (T_hot + 1e-10)

        # Confidence from final layer
        confs = []
        for prompt in PROMPTS[:4]:
            inp = tok(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                out2 = model(**inp)
            probs = torch.softmax(out2.logits[0, -1, :].float(), dim=-1)
            confs.append(probs.max().item())

        all_results[gain] = {
            'U_mean': [float(x) for x in U_mean],
            'T_mean': [float(x) for x in T_mean],
            'eta': float(eta),
            'mean_conf': float(np.mean(confs)),
            'waste_mean': [float(x) for x in np.mean(waste_all, axis=0)] if waste_all else [],
        }
        print(f"  eta={eta:.4f}, conf={np.mean(confs):.4f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) U profile comparison
    for gain in [0.0, 0.01, 0.05, 0.1]:
        if gain in all_results:
            r = all_results[gain]
            label = 'Baseline' if gain == 0 else f'gain={gain}'
            axes[0, 0].plot(r['U_mean'], 'o-', markersize=3, label=label, linewidth=2)
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('$U$ (Energy)')
    axes[0, 0].set_title('(a) Energy Profile with Ratchet')
    axes[0, 0].legend(fontsize=7)

    # (b) T profile comparison
    for gain in [0.0, 0.01, 0.05, 0.1]:
        if gain in all_results:
            r = all_results[gain]
            label = 'Baseline' if gain == 0 else f'gain={gain}'
            axes[0, 1].plot(r['T_mean'], 'o-', markersize=3, label=label, linewidth=2)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('$T$ (Temperature)')
    axes[0, 1].set_title('(b) Temperature with Ratchet')
    axes[0, 1].legend(fontsize=7)

    # (c) eta vs gain
    gains_plot = sorted(all_results.keys())
    etas_plot = [all_results[g]['eta'] for g in gains_plot]
    axes[0, 2].plot(gains_plot, etas_plot, 'o-', color='#e74c3c', markersize=8, linewidth=2)
    axes[0, 2].axhline(y=0.813, color='black', linestyle='--', alpha=0.3, label='$\\eta = 0.813$')
    axes[0, 2].set_xlabel('Ratchet Gain')
    axes[0, 2].set_ylabel('$\\eta$')
    axes[0, 2].set_title('(c) Efficiency vs Ratchet Gain')
    axes[0, 2].legend(fontsize=8)

    # (d) Waste heat profile
    for gain in [0.01, 0.05, 0.1]:
        if gain in all_results and all_results[gain]['waste_mean']:
            axes[1, 0].plot(all_results[gain]['waste_mean'], 'o-', markersize=3,
                            label=f'gain={gain}', linewidth=2)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Waste Heat Energy')
    axes[1, 0].set_title('(d) Waste Heat per Layer')
    axes[1, 0].legend(fontsize=8)

    # (e) Confidence vs gain
    confs_plot = [all_results[g]['mean_conf'] for g in gains_plot]
    axes[1, 1].plot(gains_plot, confs_plot, 's-', color='#2ecc71', markersize=8, linewidth=2)
    axes[1, 1].set_xlabel('Ratchet Gain')
    axes[1, 1].set_ylabel('Mean Confidence')
    axes[1, 1].set_title('(e) Confidence vs Ratchet Gain')

    # (f) Summary
    baseline = all_results[0.0]
    best_gain = max(gains_plot, key=lambda g: all_results[g]['mean_conf'] if g > 0 else 0)
    best = all_results[best_gain]
    summary = (
        f"Brownian Ratchet\n\n"
        f"BASELINE (no ratchet):\n"
        f"  eta = {baseline['eta']:.4f}\n"
        f"  conf = {baseline['mean_conf']:.4f}\n\n"
        f"BEST RATCHET (gain={best_gain}):\n"
        f"  eta = {best['eta']:.4f}\n"
        f"  conf = {best['mean_conf']:.4f}\n\n"
        f"Waste heat {'RECYCLED' if best['eta'] > baseline['eta'] else 'not effective'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 179: Brownian Ratchet (Waste Heat Recycling)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase179_brownian_ratchet')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Baseline: eta={baseline['eta']:.4f}, conf={baseline['mean_conf']:.4f}")
    print(f"Best ratchet (gain={best_gain}): eta={best['eta']:.4f}, conf={best['mean_conf']:.4f}")
    print(f"{'=' * 70}")

    save_results('phase179_brownian_ratchet', {
        'experiment': 'Brownian Ratchet',
        'results': {str(k): v for k, v in all_results.items()},
    })


if __name__ == '__main__':
    main()
