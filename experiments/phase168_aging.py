# -*- coding: utf-8 -*-
"""
Phase 168: Thermodynamic Aging
Does the model's thermodynamic profile change when it has seen
many tokens (long context)? Compare early vs late generation.
Simulates the "aging" of a thermodynamic system.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def safe_str(s):
    return s.encode('ascii', errors='replace').decode('ascii')


def main():
    print("=" * 70)
    print("Phase 168: Thermodynamic Aging")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Generate a long sequence and measure thermodynamics at each step
    prompt = "The history of science is a story of"
    n_gen = 50
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)

    S_history = []
    kT_history = []
    eta_history = []
    conf_history = []
    tokens_generated = []

    for step in range(n_gen):
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        # Measure at final layer
        hs_final = out.hidden_states[-1]
        with torch.no_grad():
            normed = model.model.norm(hs_final[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        S = -(probs * torch.log(probs + 1e-10)).sum().item()
        conf = probs.max().item()

        # kT
        top_k = 50
        top_probs = torch.topk(probs, top_k).values
        log_probs = torch.log(top_probs + 1e-10).cpu().numpy()
        ranks = np.arange(1, top_k + 1, dtype=np.float64)
        if np.std(log_probs) > 0.01:
            slope = np.polyfit(ranks, log_probs, 1)[0]
            kT = -1.0 / (slope + 1e-10)
        else:
            kT = 0.1
        kT = max(0.01, min(kT, 50))

        # Eta
        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed_l = model.model.norm(hs[:, -1:, :])
                log_l = model.lm_head(normed_l).squeeze().float()
            p_l = torch.softmax(log_l, dim=-1)
            s = -(p_l * torch.log(p_l + 1e-10)).sum().item()
            T_vals.append(s if not np.isnan(s) else 0)
        T_hot = max(T_vals)
        T_cold = min(T_vals[len(T_vals)//2:])
        eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0

        S_history.append(S if not np.isnan(S) else 0)
        kT_history.append(float(kT))
        eta_history.append(float(eta))
        conf_history.append(float(conf))

        # Generate next token (greedy)
        next_id = torch.argmax(out.logits[0, -1, :])
        tokens_generated.append(tok.decode([next_id]))
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)

        if step % 10 == 0:
            print(f"  Step {step}: S={S_history[-1]:.2f}, kT={kT_history[-1]:.1f}, "
                  f"eta={eta_history[-1]:.3f}")

    steps = np.arange(n_gen)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Entropy over generation
    axes[0,0].plot(steps, S_history, 'o-', color='#c0392b', markersize=4, linewidth=2)
    # Moving average
    window = 5
    if len(S_history) >= window:
        ma = np.convolve(S_history, np.ones(window)/window, mode='valid')
        axes[0,0].plot(np.arange(window-1, len(S_history)), ma, '-',
                      color='black', linewidth=2, label=f'MA({window})')
    axes[0,0].set_xlabel('Generation Step')
    axes[0,0].set_ylabel('$S$')
    axes[0,0].set_title('(a) Entropy Aging')
    axes[0,0].legend()

    # (b) kT over generation
    axes[0,1].plot(steps, kT_history, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,1].set_xlabel('Generation Step')
    axes[0,1].set_ylabel('$kT$')
    axes[0,1].set_title('(b) Temperature Aging')

    # (c) Eta over generation
    axes[0,2].plot(steps, eta_history, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0,2].set_xlabel('Generation Step')
    axes[0,2].set_ylabel('$\\eta$')
    axes[0,2].set_title('(c) Efficiency Aging')

    # (d) Confidence
    axes[1,0].plot(steps, conf_history, 'o-', color='#f39c12', markersize=4, linewidth=2)
    axes[1,0].set_xlabel('Generation Step')
    axes[1,0].set_ylabel('Top-1 Confidence')
    axes[1,0].set_title('(d) Confidence Aging')

    # (e) Phase space: S vs kT trajectory
    scatter = axes[1,1].scatter(kT_history, S_history, c=steps, cmap='viridis',
                                s=50, edgecolors='black', zorder=5)
    axes[1,1].plot(kT_history, S_history, '-', color='gray', alpha=0.3)
    plt.colorbar(scatter, ax=axes[1,1], label='Step')
    axes[1,1].set_xlabel('$kT$')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) Phase Space Trajectory')

    # (f) Summary
    early_S = np.mean(S_history[:10])
    late_S = np.mean(S_history[-10:])
    early_eta = np.mean(eta_history[:10])
    late_eta = np.mean(eta_history[-10:])

    # Check for trend
    slope_S = np.polyfit(steps, S_history, 1)[0]
    slope_eta = np.polyfit(steps, eta_history, 1)[0]

    generated = safe_str("".join(tokens_generated[:30]))
    summary = (
        f"Thermodynamic Aging\n\n"
        f"Early (0-9): S={early_S:.2f}, eta={early_eta:.3f}\n"
        f"Late ({n_gen-10}-{n_gen-1}): S={late_S:.2f}, eta={late_eta:.3f}\n\n"
        f"S trend: {slope_S:+.4f}/step\n"
        f"eta trend: {slope_eta:+.4f}/step\n\n"
        f"System {'COOLS' if slope_S < 0 else 'HEATS'}\n"
        f"during generation"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 168: Thermodynamic Aging',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase168_aging')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Early S: {early_S:.2f}, Late S: {late_S:.2f}")
    print(f"S trend: {slope_S:+.4f}/step")
    print(f"Generated: {generated}...")
    print(f"{'='*70}")

    save_results('phase168_aging', {
        'experiment': 'Thermodynamic Aging',
        'summary': {
            'early_S': float(early_S),
            'late_S': float(late_S),
            'slope_S': float(slope_S),
            'slope_eta': float(slope_eta),
        }
    })


if __name__ == '__main__':
    main()
