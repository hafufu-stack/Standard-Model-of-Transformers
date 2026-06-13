# -*- coding: utf-8 -*-
"""
Phase 130: Active Matter Classification
FDT is violated (P126), Maxwell relations fail (P128).
This points to a non-equilibrium, active-matter-like system.
Active matter is characterized by:
  1. Persistent currents (non-zero probability flux)
  2. Time-reversal symmetry breaking
  3. Entropy production > 0 (irreversibility)
  4. Detailed balance violation

Measure these properties across layers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
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
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def main():
    print("=" * 70)
    print("Phase 130: Active Matter Classification")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1
    L0 = 21.7

    # 1. Persistent currents: measure directional bias of hidden state updates
    # If delta_h has consistent direction across prompts, there's a "current"
    current_magnitude = []
    current_consistency = []  # cosine sim of delta_h across prompts

    # 2. Time-reversal: compare forward (L->L+1) and backward (L+1->L) statistics
    time_asymmetry = []

    # 3. Entropy production: already measured in P113, but compute here too
    entropy_production = []

    # 4. Detailed balance: P(state_A -> state_B) vs P(state_B -> state_A)
    detailed_balance_violation = []

    all_hidden = []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        hs = [out.hidden_states[li][0, -1, :].float().cpu().numpy() for li in range(n_layers)]
        all_hidden.append(hs)

    # Compute for each layer transition (L -> L+1)
    for li in range(n_layers - 1):
        # Collect delta_h vectors
        deltas = []
        norms_fwd = []
        for pi in range(len(PROMPTS)):
            delta = all_hidden[pi][li + 1] - all_hidden[pi][li]
            deltas.append(delta)
            norms_fwd.append(np.linalg.norm(delta))

        deltas = np.array(deltas)
        mean_delta = np.mean(deltas, axis=0)

        # 1. Current: mean delta norm / individual delta norms
        current_mag = np.linalg.norm(mean_delta) / (np.mean(norms_fwd) + 1e-10)
        current_magnitude.append(float(current_mag))

        # Consistency: avg pairwise cosine between deltas
        cos_sims = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                c = np.dot(deltas[i], deltas[j]) / (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j]) + 1e-10)
                cos_sims.append(c)
        current_consistency.append(float(np.mean(cos_sims)) if cos_sims else 0)

        # 2. Time-reversal asymmetry: compare |h_{l+1} - h_l| with transition statistics
        # Asymmetry = |mean(delta)| / std(delta) -- higher = more asymmetric
        std_delta = np.std(norms_fwd)
        ta = np.linalg.norm(mean_delta) / (std_delta + 1e-10)
        time_asymmetry.append(float(ta))

        # 3. Entropy production: sigma = d(output_entropy)/dL
        S_vals = []
        for pi in range(len(PROMPTS)):
            h_tensor = torch.tensor(all_hidden[pi][li + 1], device=device).to(model.dtype)
            with torch.no_grad():
                normed = model.model.norm(h_tensor.unsqueeze(0).unsqueeze(0))
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_vals.append(S if not np.isnan(S) else 0)

        S_prev = []
        for pi in range(len(PROMPTS)):
            h_tensor = torch.tensor(all_hidden[pi][li], device=device).to(model.dtype)
            with torch.no_grad():
                normed = model.model.norm(h_tensor.unsqueeze(0).unsqueeze(0))
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_prev.append(S if not np.isnan(S) else 0)

        sigma = float(np.mean(S_vals) - np.mean(S_prev))
        entropy_production.append(sigma)

        # 4. Detailed balance: measure asymmetry of transition probabilities
        # Proxy: difference in cosine similarity forward vs "reverse"
        fwd_cos = []
        for pi in range(len(PROMPTS)):
            c = np.dot(all_hidden[pi][li], all_hidden[pi][li+1]) / (
                np.linalg.norm(all_hidden[pi][li]) * np.linalg.norm(all_hidden[pi][li+1]) + 1e-10)
            fwd_cos.append(c)
        detailed_balance_violation.append(float(1 - np.mean(fwd_cos)))

    layers_t = np.arange(n_layers - 1) + 0.5  # transition midpoints

    # Active matter score: combine all indicators
    active_score = np.array(current_magnitude) * np.array(time_asymmetry)

    pre_active = np.mean(active_score[:int(L0)])
    post_active = np.mean(active_score[int(L0):])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Persistent currents
    axes[0,0].plot(layers_t, current_magnitude, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--', label=f'$L_0$')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Current Magnitude')
    axes[0,0].set_title('(a) Persistent Currents')
    axes[0,0].legend()

    # (b) Current consistency
    axes[0,1].plot(layers_t, current_consistency, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Update Consistency')
    axes[0,1].set_title('(b) Direction Agreement')

    # (c) Time-reversal asymmetry
    axes[0,2].plot(layers_t, time_asymmetry, 'o-', color='#27ae60', markersize=4, linewidth=2)
    axes[0,2].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Time Asymmetry')
    axes[0,2].set_title('(c) Time-Reversal Breaking')

    # (d) Entropy production
    colors_d = ['#c0392b' if s > 0 else '#2980b9' for s in entropy_production]
    axes[1,0].bar(layers_t, entropy_production, color=colors_d, alpha=0.7, edgecolor='black', width=0.8)
    axes[1,0].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].axhline(y=0, color='black', linewidth=0.5)
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$\\sigma$ (entropy production)')
    axes[1,0].set_title('(d) Irreversibility')

    # (e) Active matter score
    axes[1,1].plot(layers_t, active_score, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[1,1].axvline(x=L0, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].fill_between(layers_t, active_score, alpha=0.2, color='#8e44ad')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Active Score')
    axes[1,1].set_title(f'(e) Active Matter Score (pre={pre_active:.2f}, post={post_active:.2f})')

    # (f) Classification summary
    sigma_total = np.sum(entropy_production)
    mean_current = np.mean(current_magnitude)
    mean_ta = np.mean(time_asymmetry)

    classification = "ACTIVE MATTER" if (sigma_total > 0 or mean_ta > 1) else "NEAR-EQUILIBRIUM"

    summary = (
        f"Active Matter Classification\n\n"
        f"Persistent current: {mean_current:.3f}\n"
        f"Time asymmetry: {mean_ta:.2f}\n"
        f"Total sigma: {sigma_total:.3f}\n"
        f"DB violation: {np.mean(detailed_balance_violation):.3f}\n\n"
        f"Pre-transition: {pre_active:.2f}\n"
        f"Post-transition: {post_active:.2f}\n\n"
        f"Classification: {classification}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Classification')

    fig.suptitle('Phase 130: Active Matter Classification',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase130_active')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Classification: {classification}")
    print(f"Current: {mean_current:.3f}, Time asym: {mean_ta:.2f}")
    print(f"Active score: pre={pre_active:.2f}, post={post_active:.2f}")
    print(f"{'='*70}")

    save_results('phase130_active', {
        'experiment': 'Active Matter Classification',
        'current_magnitude': [float(v) for v in current_magnitude],
        'time_asymmetry': [float(v) for v in time_asymmetry],
        'entropy_production': [float(v) for v in entropy_production],
        'active_score': [float(v) for v in active_score],
        'summary': {
            'classification': classification,
            'mean_current': float(mean_current),
            'mean_time_asym': float(mean_ta),
            'sigma_total': float(sigma_total),
            'pre_active': float(pre_active),
            'post_active': float(post_active),
        }
    })


if __name__ == '__main__':
    main()
