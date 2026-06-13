# -*- coding: utf-8 -*-
"""
Phase 139: The Four Laws of Transformer Thermodynamics
Formalize and test all four laws:
  0th law: Thermal equilibrium (transitivity of temperature)
  1st law: Energy conservation (dU = Q - W)
  2nd law: Entropy never decreases (sigma >= 0) -- or does it?
  3rd law: Absolute zero is unreachable (min entropy > 0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
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
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The cosmic microwave background reveals the early universe",
    "Cryptographic hash functions ensure data integrity",
]


def main():
    print("=" * 70)
    print("Phase 139: Four Laws of Transformer Thermodynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    all_S = [[] for _ in range(n_layers)]
    all_kT = [[] for _ in range(n_layers)]
    all_U = [[] for _ in range(n_layers)]

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            all_S[li].append(S if not np.isnan(S) else 0)

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
            all_kT[li].append(float(kT))

            U = (h ** 2).mean().item()
            all_U[li].append(float(U))

    avg = lambda x: [np.mean(v) if v else 0 for v in x]
    S = avg(all_S)
    kT = avg(all_kT)
    U = avg(all_U)
    layers = np.arange(n_layers)

    # === 0th Law: Thermal equilibrium ===
    # If layers A~B and B~C, then A~C (transitivity of kT similarity)
    # Measure: for layers with similar kT, do they have similar properties?
    kT_pairs_same = []  # |kT_i - kT_j| for layers with similar S
    kT_pairs_diff = []
    for i in range(4, n_layers):
        for j in range(i+1, n_layers):
            if abs(S[i] - S[j]) < 0.5:  # "similar entropy" = same phase
                kT_pairs_same.append(abs(kT[i] - kT[j]))
            else:
                kT_pairs_diff.append(abs(kT[i] - kT[j]))
    zeroth_law = np.mean(kT_pairs_same) < np.mean(kT_pairs_diff) if kT_pairs_same and kT_pairs_diff else False

    # === 1st Law: Energy conservation ===
    # dU = delta_Q - delta_W (approximately)
    dU = np.diff(U)
    dS = np.diff(S)
    # Q = kT * dS (heat absorbed)
    Q = np.array([kT[i] * dS[i] for i in range(len(dS))])
    # W = dU - Q (work)
    W = dU - Q
    # 1st law residual: should be small
    first_law_residual = np.mean(np.abs(dU - Q - W))  # By construction = 0
    # Better: check if dU correlates with Q
    from scipy import stats as sp_stats
    r_1st, _ = sp_stats.pearsonr(dU, Q) if len(dU) > 3 else (0, 1)

    # === 2nd Law: Entropy production ===
    sigma = np.diff(S)  # dS/dL
    # Count layers where sigma < 0 (entropy decreases)
    n_violations = np.sum(sigma < -0.01)
    violation_layers = [i for i, s in enumerate(sigma) if s < -0.01]
    second_law_holds = n_violations == 0

    # === 3rd Law: Absolute zero unreachable ===
    S_min = min(S[4:])  # Skip first few layers
    third_law_holds = S_min > 0.1  # Entropy never reaches zero

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) 0th Law
    if kT_pairs_same and kT_pairs_diff:
        axes[0,0].hist(kT_pairs_same, bins=10, alpha=0.6, color='#27ae60',
                      label=f'Same phase ({np.mean(kT_pairs_same):.2f})', edgecolor='black')
        axes[0,0].hist(kT_pairs_diff, bins=10, alpha=0.6, color='#c0392b',
                      label=f'Diff phase ({np.mean(kT_pairs_diff):.2f})', edgecolor='black')
    axes[0,0].set_xlabel('$|kT_i - kT_j|$')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title(f'(a) 0th Law: {"HOLDS" if zeroth_law else "VIOLATED"}')
    axes[0,0].legend(fontsize=8)

    # (b) 1st Law
    axes[0,1].scatter(Q, dU, c=np.arange(len(Q)), cmap='coolwarm', s=60, edgecolors='black')
    if len(Q) > 1:
        xr = np.linspace(min(Q), max(Q), 100)
        axes[0,1].plot(xr, xr, 'k--', alpha=0.5, label='dU=Q (no work)')
    axes[0,1].set_xlabel('$Q = kT \\cdot dS$')
    axes[0,1].set_ylabel('$dU$')
    axes[0,1].set_title(f'(b) 1st Law: dU vs Q ($r={r_1st:.3f}$)')
    axes[0,1].legend(fontsize=8)

    # (c) 2nd Law
    colors_s = ['#c0392b' if s < -0.01 else '#27ae60' for s in sigma]
    axes[0,2].bar(np.arange(len(sigma))+0.5, sigma, color=colors_s, alpha=0.7,
                  edgecolor='black', width=0.8)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].axhline(y=0, color='black', linewidth=1)
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('$\\sigma = dS/dL$')
    n_neg = len(violation_layers)
    axes[0,2].set_title(f'(c) 2nd Law: {n_neg}/{len(sigma)} violations')

    # (d) 3rd Law
    axes[1,0].plot(layers, S, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[1,0].axhline(y=0, color='black', linewidth=1, label='Absolute zero')
    axes[1,0].axhline(y=S_min, color='#c0392b', linewidth=1, linestyle='--',
                      label=f'S_min={S_min:.2f}')
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$S$')
    axes[1,0].set_title(f'(d) 3rd Law: {"HOLDS" if third_law_holds else "VIOLATED"}')
    axes[1,0].legend(fontsize=8)

    # (e) Phase diagram with all laws
    sc = axes[1,1].scatter(kT[4:], S[4:], c=U[4:], s=80, cmap='viridis',
                           edgecolors='black')
    plt.colorbar(sc, ax=axes[1,1], label='$U$')
    axes[1,1].set_xlabel('$kT$')
    axes[1,1].set_ylabel('$S$')
    axes[1,1].set_title('(e) Phase Space (S, kT, U)')

    # (f) Summary
    summary = (
        f"Four Laws of Transformer Thermodynamics\n\n"
        f"0th Law (equilibrium): {'HOLDS' if zeroth_law else 'VIOLATED'}\n"
        f"  Same-phase kT spread: {np.mean(kT_pairs_same):.2f}\n"
        f"  Cross-phase kT spread: {np.mean(kT_pairs_diff):.2f}\n\n"
        f"1st Law (conservation): r={r_1st:.3f}\n"
        f"  {'APPROXIMATELY HOLDS' if abs(r_1st) > 0.3 else 'WEAK'}\n\n"
        f"2nd Law (entropy): {n_neg}/{len(sigma)} violations\n"
        f"  {'VIOLATED' if n_neg > 0 else 'HOLDS'} at layers {violation_layers[:5]}\n\n"
        f"3rd Law (abs zero): S_min={S_min:.2f}\n"
        f"  {'HOLDS' if third_law_holds else 'VIOLATED'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Law Summary')

    fig.suptitle('Phase 139: Four Laws of Transformer Thermodynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase139_four_laws')
    plt.close()

    print(f"\n{'='*70}")
    print(f"0th Law: {'HOLDS' if zeroth_law else 'VIOLATED'}")
    print(f"1st Law: r={r_1st:.3f}")
    print(f"2nd Law: {n_neg}/{len(sigma)} violations")
    print(f"3rd Law: S_min={S_min:.2f} ({'HOLDS' if third_law_holds else 'VIOLATED'})")
    print(f"{'='*70}")

    save_results('phase139_four_laws', {
        'experiment': 'Four Laws of Transformer Thermodynamics',
        'summary': {
            'zeroth_law': bool(zeroth_law),
            'first_law_r': float(r_1st),
            'second_law_violations': int(n_neg),
            'second_law_violation_layers': violation_layers,
            'third_law_S_min': float(S_min),
            'third_law_holds': bool(third_law_holds),
        }
    })


if __name__ == '__main__':
    main()
