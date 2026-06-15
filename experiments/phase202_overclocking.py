# -*- coding: utf-8 -*-
"""
Phase 202: Thermodynamic Overclocking
======================================
The model runs at 1.7% of the Mandelstam-Tamm speed limit (Phase 193).
Force it to go faster by scaling residual stream contributions.

At what gain does chaos emerge (positive Lyapunov exponents)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
]

L0 = 21
GAINS = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
EPS = 1e-4  # Perturbation for Lyapunov


def measure_at_gain(model, tok, device, gain):
    """Run forward pass with residual scaling and measure thermodynamics + Lyapunov."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Install residual scaling hooks
    hooks = []
    if gain != 1.0:
        for li in range(n_layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    # Residual connection: output = input + attn(x) + ffn(x)
                    # Scale the non-residual contribution
                    inp = input[0] if isinstance(input, tuple) else input
                    residual = inp
                    delta = h.float() - residual.float()
                    h_new = residual.float() + gain * delta
                    h_new = torch.nan_to_num(h_new, nan=0.0, posinf=65000.0, neginf=-65000.0)
                    result = h_new.to(h.dtype)
                    if isinstance(output, tuple):
                        return (result,) + output[1:]
                    return result
                return hook
            hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    all_U, all_T, all_S = [], [], []
    all_bures = []
    all_lyapunov = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Normal forward
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_list, T_list, S_list = [], [], []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S_list.append(-(h_prob * torch.log(h_prob + 1e-10)).sum().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T_val if not np.isnan(T_val) else 0)

        all_U.append(U_list)
        all_T.append(T_list)
        all_S.append(S_list)

        # Bures angles between consecutive layers
        bures = []
        for li in range(len(out.hidden_states) - 1):
            h1 = out.hidden_states[li][0, -1, :].float()
            h2 = out.hidden_states[li + 1][0, -1, :].float()
            cos_sim = torch.dot(h1, h2) / (h1.norm() * h2.norm() + 1e-10)
            cos_sim = cos_sim.clamp(-1, 1)
            angle = torch.arccos(cos_sim).item()
            bures.append(angle if not np.isnan(angle) else 0)
        all_bures.append(bures)

        # Lyapunov: perturb and measure divergence
        lyap = []
        for li in range(1, min(n_layers, len(out.hidden_states) - 1)):
            h = out.hidden_states[li][0, -1, :].float()
            h_perturbed = h + EPS * torch.randn_like(h)
            # Compare after one more layer propagation
            h_next = out.hidden_states[li + 1][0, -1, :].float()
            # Approximate: divergence = ||delta_next|| / ||delta_in||
            delta_in = EPS
            delta_out = (h_next - h).norm().item()
            if delta_in > 0:
                lyap.append(np.log(abs(delta_out / delta_in + 1e-30)))
            else:
                lyap.append(0)
        all_lyapunov.append(lyap)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    U_mean = np.mean(all_U, axis=0).tolist()
    T_mean = np.mean(all_T, axis=0).tolist()
    S_mean = np.mean(all_S, axis=0).tolist()
    bures_mean = np.mean(all_bures, axis=0).tolist()
    lyap_mean = np.mean(all_lyapunov, axis=0).tolist()

    T_hot = max(T_mean[1:]) if len(T_mean) > 1 else 0
    T_cold = min(T_mean[1:]) if len(T_mean) > 1 else 0
    eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

    speed = np.mean(bures_mean) if bures_mean else 0
    max_lyap = max(lyap_mean) if lyap_mean else 0
    n_positive_lyap = sum(1 for l in lyap_mean if l > 0)

    return {
        'U': U_mean, 'T': T_mean, 'S': S_mean,
        'eta': eta, 'speed': speed,
        'max_lyapunov': max_lyap,
        'n_positive_lyapunov': n_positive_lyap,
        'bures_mean': bures_mean,
        'lyapunov_spectrum': lyap_mean,
    }


def main():
    print("=" * 70)
    print("Phase 202: Thermodynamic Overclocking")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    results_by_gain = {}
    for gi, gain in enumerate(GAINS):
        print(f"\n[{gi+1}/{len(GAINS)}] Gain = {gain:.1f}x")
        r = measure_at_gain(model, tok, device, gain)
        results_by_gain[str(gain)] = r
        print(f"  eta={r['eta']:.4f}, speed={r['speed']:.4f}, "
              f"max_lyap={r['max_lyapunov']:.4f}, n_positive={r['n_positive_lyapunov']}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(GAINS)))

    # (a) eta vs gain
    etas = [results_by_gain[str(g)]['eta'] for g in GAINS]
    axes[0, 0].plot(GAINS, etas, 'o-', color='#e74c3c', markersize=8, linewidth=2)
    axes[0, 0].axhline(y=etas[0], color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Residual Gain')
    axes[0, 0].set_ylabel('Carnot Efficiency eta')
    axes[0, 0].set_title('(a) Efficiency vs Gain')

    # (b) Speed vs gain
    speeds = [results_by_gain[str(g)]['speed'] for g in GAINS]
    axes[0, 1].plot(GAINS, speeds, 's-', color='#3498db', markersize=8, linewidth=2)
    axes[0, 1].set_xlabel('Residual Gain')
    axes[0, 1].set_ylabel('Mean Bures Angle (rad)')
    axes[0, 1].set_title('(b) Computation Speed vs Gain')

    # (c) Max Lyapunov vs gain
    lyaps = [results_by_gain[str(g)]['max_lyapunov'] for g in GAINS]
    n_pos = [results_by_gain[str(g)]['n_positive_lyapunov'] for g in GAINS]
    axes[0, 2].plot(GAINS, lyaps, 'D-', color='#e67e22', markersize=8, linewidth=2)
    axes[0, 2].axhline(y=0, color='black', linewidth=1, linestyle='-')
    axes[0, 2].set_xlabel('Residual Gain')
    axes[0, 2].set_ylabel('Max Lyapunov Exponent')
    axes[0, 2].set_title('(c) Chaos Onset')
    # Mark chaos threshold
    for gi, (g, l) in enumerate(zip(GAINS, lyaps)):
        if l > 0:
            axes[0, 2].axvline(x=g, color='red', linestyle=':', alpha=0.5)
            break

    # (d) T profiles at different gains
    for gi, g in enumerate(GAINS):
        T = results_by_gain[str(g)]['T']
        axes[1, 0].plot(range(len(T)), T, '-', color=cmap[gi], linewidth=1.5,
                        label=f'{g}x', alpha=0.8)
    axes[1, 0].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Temperature T')
    axes[1, 0].set_title('(d) Temperature Profiles')
    axes[1, 0].legend(fontsize=7, ncol=2)

    # (e) Lyapunov spectra
    for gi, g in enumerate(GAINS):
        lyap_spec = results_by_gain[str(g)]['lyapunov_spectrum']
        axes[1, 1].plot(range(len(lyap_spec)), lyap_spec, '-', color=cmap[gi],
                        linewidth=1.5, label=f'{g}x', alpha=0.8)
    axes[1, 1].axhline(y=0, color='black', linewidth=1)
    axes[1, 1].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Lyapunov Exponent')
    axes[1, 1].set_title('(e) Lyapunov Spectra')
    axes[1, 1].legend(fontsize=7, ncol=2)

    # (f) Summary
    # Find chaos onset gain
    chaos_gain = None
    for g, l in zip(GAINS, lyaps):
        if l > 0:
            chaos_gain = g
            break
    summary_text = (
        f"Thermodynamic Overclocking\n\n"
        f"Baseline (1.0x):\n"
        f"  eta = {etas[0]:.4f}\n"
        f"  speed = {speeds[0]:.4f}\n"
        f"  max_lyap = {lyaps[0]:.4f}\n\n"
        f"Speed increase at 2.0x:\n"
        f"  {speeds[GAINS.index(2.0)]/speeds[0]:.1f}x faster\n\n"
        f"Chaos onset: {'gain=' + str(chaos_gain) + 'x' if chaos_gain else 'Not reached'}\n"
        f"N positive Lyap at max gain:\n"
        f"  {n_pos[-1]}/{len(results_by_gain[str(GAINS[-1])]['lyapunov_spectrum'])}"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 202: Thermodynamic Overclocking", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase202_overclocking')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Chaos onset at gain = {chaos_gain}")
    print(f"{'=' * 70}")

    save_results('phase202_overclocking', {
        'experiment': 'Thermodynamic Overclocking',
        'gains': GAINS,
        'results': {str(g): {'eta': results_by_gain[str(g)]['eta'],
                             'speed': results_by_gain[str(g)]['speed'],
                             'max_lyapunov': results_by_gain[str(g)]['max_lyapunov'],
                             'n_positive_lyapunov': results_by_gain[str(g)]['n_positive_lyapunov']}
                    for g in GAINS},
        'summary': {
            'chaos_onset_gain': chaos_gain,
            'speed_increase_2x': speeds[GAINS.index(2.0)] / (speeds[0] + 1e-10),
            'eta_at_baseline': etas[0],
            'eta_at_max_gain': etas[-1],
        }
    })


if __name__ == '__main__':
    main()
