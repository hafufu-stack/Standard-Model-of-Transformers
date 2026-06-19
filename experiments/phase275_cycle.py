# -*- coding: utf-8 -*-
"""
Phase 275: Thermodynamic Cycle Efficiency
===========================================
Carnot efficiency eta=0.813 was measured for a single forward pass.
What happens during multi-step autoregressive generation?

Track U, T, S, P1 at each generation step to construct a thermodynamic
cycle on the PV (P1 vs T) diagram and compute the work W = integral P dV.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPT = "The theory of thermodynamics tells us that"
N_STEPS = 50


def generate_with_thermo_tracking(model, tok, prompt, device, n_steps=N_STEPS):
    """Generate tokens while tracking thermodynamic quantities at each step."""
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    generated = inp['input_ids'].clone()
    input_len = generated.shape[1]

    trajectory = []

    for step in range(n_steps):
        with torch.no_grad():
            out = model(generated, output_hidden_states=True)

        # Hidden state at last position, last layer
        hs = out.hidden_states[-1][0, -1, :].float()
        U = hs.norm().item()

        # PR from hidden state
        h_sq = hs ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        PR = 1.0 / (h_prob ** 2).sum().item()

        # T, P1 from logits
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs.max().item()
        t_val = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Free energy F = U - T*S where S ~ log(PR)
        S = np.log(PR + 1) if PR > 0 else 0
        F = U - t_val * S

        trajectory.append({
            'step': step,
            'U': round(U, 4),
            'T': round(t_val, 4),
            'P1': round(p1, 4),
            'PR': round(PR, 2),
            'S': round(S, 4),
            'F': round(F, 4),
            'P1T': round(p1 * t_val, 4),
        })

        # Greedy next token
        next_id = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break

    # Compute cycle work: W = integral P1 dT (on P1-T diagram)
    p1_arr = np.array([t['P1'] for t in trajectory])
    t_arr = np.array([t['T'] for t in trajectory])
    # Trapezoidal integration
    work = float(np.trapz(p1_arr, t_arr))

    # Efficiency: eta = W / Q_in where Q_in ~ sum of |dU| for heating steps
    u_arr = np.array([t['U'] for t in trajectory])
    du = np.diff(u_arr)
    q_in = float(np.sum(np.abs(du[du > 0]))) if np.any(du > 0) else 1.0
    eta = abs(work) / q_in if q_in > 0 else 0

    # Carnot comparison
    t_hot = max(t_arr)
    t_cold = min(t_arr)
    eta_carnot = 1 - t_cold / (t_hot + 1e-10)

    text = tok.decode(generated[0, input_len:], skip_special_tokens=True)

    return {
        'trajectory': trajectory,
        'work': round(work, 6),
        'q_in': round(q_in, 4),
        'eta': round(eta, 4),
        'eta_carnot': round(eta_carnot, 4),
        't_hot': round(float(t_hot), 4),
        't_cold': round(float(t_cold), 4),
        'text': text[:200],
    }


def main():
    print("=" * 70)
    print("Phase 275: Thermodynamic Cycle Efficiency")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)
        r = generate_with_thermo_tracking(model, tok, PROMPT, device)
        all_results[size] = r
        print(f"  Work W = {r['work']:.4f}")
        print(f"  Efficiency eta = {r['eta']:.4f}")
        print(f"  Carnot eta = {r['eta_carnot']:.4f}")
        print(f"  T range: {r['t_cold']:.2f} -> {r['t_hot']:.2f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        c = colors[size]
        traj = data['trajectory']
        steps = [t['step'] for t in traj]

        # (a) P1 vs T diagram (thermodynamic cycle)
        axes[0, 0].plot([t['T'] for t in traj], [t['P1'] for t in traj],
                       '-', color=c, lw=2, label=size)
        axes[0, 0].scatter([traj[0]['T']], [traj[0]['P1']], marker='o',
                          c=c, s=100, zorder=5, edgecolors='black')
        axes[0, 0].scatter([traj[-1]['T']], [traj[-1]['P1']], marker='s',
                          c=c, s=100, zorder=5, edgecolors='black')

    axes[0, 0].set_xlabel('Temperature T')
    axes[0, 0].set_ylabel('P1 (Pressure)')
    axes[0, 0].set_title('(a) Thermodynamic Cycle (P1 vs T)', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) U trajectory
    for size, data in all_results.items():
        traj = data['trajectory']
        axes[0, 1].plot([t['step'] for t in traj], [t['U'] for t in traj],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('Internal Energy U')
    axes[0, 1].set_title('(b) Internal Energy Trajectory', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) T trajectory
    for size, data in all_results.items():
        traj = data['trajectory']
        axes[0, 2].plot([t['step'] for t in traj], [t['T'] for t in traj],
                       '-', color=colors[size], lw=2, label=size)
    axes[0, 2].set_xlabel('Generation Step')
    axes[0, 2].set_ylabel('Temperature T')
    axes[0, 2].set_title('(c) Temperature During Generation', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) P1*T conservation during generation
    for size, data in all_results.items():
        traj = data['trajectory']
        axes[1, 0].plot([t['step'] for t in traj], [t['P1T'] for t in traj],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 0].axhline(0.84, color='red', ls='--', label='P1T=0.84')
    axes[1, 0].set_xlabel('Generation Step')
    axes[1, 0].set_ylabel('P1 * T')
    axes[1, 0].set_title('(d) P1*T During Generation', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # (e) Free energy trajectory
    for size, data in all_results.items():
        traj = data['trajectory']
        axes[1, 1].plot([t['step'] for t in traj], [t['F'] for t in traj],
                       '-', color=colors[size], lw=2, label=size)
    axes[1, 1].set_xlabel('Generation Step')
    axes[1, 1].set_ylabel('Free Energy F')
    axes[1, 1].set_title('(e) Free Energy During Generation', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "THERMODYNAMIC CYCLE\n\n"
    for size, data in all_results.items():
        summary += f"{size}:\n"
        summary += f"  Work W = {data['work']:.4f}\n"
        summary += f"  eta = {data['eta']:.4f}\n"
        summary += f"  eta_Carnot = {data['eta_carnot']:.4f}\n"
        summary += f"  T: [{data['t_cold']:.1f}, {data['t_hot']:.1f}]\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 275: Thermodynamic Cycle Efficiency",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase275_cycle')
    plt.close()

    save_results('phase275_cycle', {
        'experiment': 'Thermodynamic Cycle Efficiency',
        'prompt': PROMPT,
        'n_steps': N_STEPS,
        'results': all_results,
    })


if __name__ == '__main__':
    main()
