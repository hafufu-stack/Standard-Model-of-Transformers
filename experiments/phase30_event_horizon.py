# -*- coding: utf-8 -*-
"""
Phase 30: The Event Horizon (Opus Original)
=============================================
Find the exact layer where the "phase transition" occurs:
- Free particles (high kinetic, low potential) -> Bound state (low kinetic, high potential)
- This is the "event horizon" of the Transformer black hole
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, measure_full_thermodynamics, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 30: The Event Horizon")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The fundamental laws of physics govern all matter",
        "Neural networks learn representations from data",
        "The second law of thermodynamics states that entropy",
        "Stars form from collapsing clouds of gas and dust",
        "Quantum mechanics describes the behavior of particles",
        "The gradient descent algorithm minimizes loss functions",
        "Information theory quantifies uncertainty in signals",
        "Black holes have an event horizon beyond which nothing",
    ]

    all_entropy_rate = []
    all_cos_velocity = []
    all_norm_accel = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hs = [out.hidden_states[l][0, -1, :].float().cpu() for l in range(len(out.hidden_states))]

        # Entropy of direction change (cosine between consecutive velocity vectors)
        velocities = [hs[l+1] - hs[l] for l in range(len(hs)-1)]
        cos_changes = []
        for l in range(len(velocities)-1):
            v1 = velocities[l]
            v2 = velocities[l+1]
            cos = torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-10)
            cos_changes.append(cos.item())
        all_cos_velocity.append(cos_changes)

        # Norm acceleration (second derivative of norm)
        norms = [h.norm().item() for h in hs]
        accel = np.diff(np.diff(norms))
        all_norm_accel.append(accel)

        # Information entropy rate (how fast entropy changes)
        thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
        entropies = [t['T'] for t in thermo]
        entropy_rate = np.diff(entropies)
        all_entropy_rate.append(entropy_rate)

    # Average across prompts
    min_cos = min(len(c) for c in all_cos_velocity)
    avg_cos = np.mean([c[:min_cos] for c in all_cos_velocity], axis=0)

    min_accel = min(len(a) for a in all_norm_accel)
    avg_accel = np.mean([a[:min_accel] for a in all_norm_accel], axis=0)

    min_erate = min(len(e) for e in all_entropy_rate)
    avg_erate = np.mean([e[:min_erate] for e in all_entropy_rate], axis=0)

    # Find event horizon: the layer with maximum cos velocity change
    # (maximum directional shift = point of no return)
    horizon_cos = np.argmin(avg_cos) + 1  # +1 because cos is between consecutive
    horizon_accel = np.argmax(np.abs(avg_accel)) + 1

    # Also find the layer where entropy rate changes sign (cooling onset)
    sign_changes = np.where(np.diff(np.sign(avg_erate)))[0]
    horizon_entropy = sign_changes[0] + 1 if len(sign_changes) > 0 else -1

    print(f"\n--- Event Horizon Detection ---")
    print(f"  Directional shift maximum at layer: {horizon_cos}")
    print(f"  Norm acceleration maximum at layer: {horizon_accel}")
    print(f"  Entropy rate sign change at layer: {horizon_entropy}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    layers = np.arange(len(avg_cos)) + 1
    ax.plot(layers, avg_cos, 'o-', color='#e74c3c', ms=4)
    ax.axvline(x=horizon_cos, color='gold', ls='--', lw=2, label=f'Event horizon L{horizon_cos}')
    ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cos(velocity[l], velocity[l+1])')
    ax.set_title('(a) Directional Coherence')
    ax.legend()

    ax = axes[1]
    layers2 = np.arange(len(avg_accel)) + 1
    ax.plot(layers2, avg_accel, 'o-', color='#3498db', ms=4)
    ax.axvline(x=horizon_accel, color='gold', ls='--', lw=2, label=f'Max accel L{horizon_accel}')
    ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Norm Acceleration (d2||h||/dl2)')
    ax.set_title('(b) Norm Acceleration')
    ax.legend()

    ax = axes[2]
    layers3 = np.arange(len(avg_erate)) + 1
    ax.plot(layers3, avg_erate, 'o-', color='#2ecc71', ms=4)
    if horizon_entropy > 0:
        ax.axvline(x=horizon_entropy, color='gold', ls='--', lw=2, label=f'Entropy flip L{horizon_entropy}')
    ax.axhline(y=0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('dT/dl (entropy rate)')
    ax.set_title('(c) Entropy Rate')
    ax.legend()

    fig.suptitle(
        f"Phase 30: The Event Horizon\n"
        f"Direction shift: L{horizon_cos} | Norm accel: L{horizon_accel} | "
        f"Entropy flip: L{horizon_entropy}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase30_event_horizon")
    plt.close()

    verdict = (f"EVENT HORIZON at L{horizon_cos} (direction) / L{horizon_accel} (acceleration) / "
               f"L{horizon_entropy} (entropy). Before this layer: free particles. "
               f"After: gravitationally bound state.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase30_event_horizon", {
        'name': 'Phase 30: The Event Horizon',
        'summary': {'verdict': verdict, 'horizon_direction': int(horizon_cos),
                    'horizon_acceleration': int(horizon_accel),
                    'horizon_entropy': int(horizon_entropy)},
    })


if __name__ == '__main__':
    main()
