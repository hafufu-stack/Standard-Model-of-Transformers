# -*- coding: utf-8 -*-
"""
Phase 66: SVD Entropy Free Energy (from Deep Think 3)
Use SVD effective rank (intrinsic dimensionality) as true entropy S.
F = U - T * S_svd should decrease across layers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def svd_entropy(hidden_states_matrix):
    """Compute SVD entropy = effective rank from singular values."""
    # hidden_states_matrix: (seq, hidden)
    try:
        U, S, Vh = torch.linalg.svd(hidden_states_matrix.float(), full_matrices=False)
        S = S[S > 1e-10]
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
        eff_rank = np.exp(entropy)
        return entropy, eff_rank
    except Exception:
        return 0.0, 1.0


def main():
    print("=" * 70)
    print("Phase 66: SVD Entropy Free Energy")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration through",
        "In quantum mechanics the wave function collapse occurs when a measurement is",
        "The human genome contains approximately three billion base pairs encoding",
        "Artificial neural networks process information through layers of interconnected",
        "Black holes form when massive stars exhaust their nuclear fuel and",
        "The periodic table organizes elements by their atomic number and electron",
        "Evolution by natural selection operates on heritable variation within",
        "Climate models simulate atmospheric dynamics using differential equations that",
        "Cryptographic hash functions produce fixed size output from arbitrary input",
        "The cosmic microwave background provides a snapshot of the early universe",
        "Photosynthesis converts light energy into chemical energy stored in glucose",
        "Machine learning algorithms discover patterns in data without explicit programming",
    ]

    all_profiles = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        F_svd_list = []
        F_logPR_list = []
        F_old_list = []
        U_list = []
        T_list = []
        S_svd_list = []

        for li, hs in enumerate(out.hidden_states):
            h = hs[0].float()  # (seq, hidden)
            last_h = h[-1, :]

            # U = L2 norm of last token
            U = last_h.norm().item()

            # T = logit entropy
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0

            # S_svd = SVD entropy of all token hidden states
            S_svd_val, eff_rank = svd_entropy(h)

            # PR from hidden activations
            h_sq = last_h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_prob ** 2).sum().item()

            # Three definitions of F
            F_svd = U - T * S_svd_val
            F_logPR = U - T * np.log(PR + 1e-10)
            F_old = U - T * T  # original wrong definition

            F_svd_list.append(F_svd)
            F_logPR_list.append(F_logPR)
            F_old_list.append(F_old)
            U_list.append(U)
            T_list.append(T)
            S_svd_list.append(S_svd_val)

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' F_svd: {F_svd_list[0]:.0f} -> {F_svd_list[-1]:.0f}")

        all_profiles.append({
            'F_svd': F_svd_list, 'F_logPR': F_logPR_list, 'F_old': F_old_list,
            'U': U_list, 'T': T_list, 'S_svd': S_svd_list,
        })

    n_layers = len(all_profiles[0]['F_svd'])
    layers_x = np.arange(n_layers)

    mean_F_svd = np.mean([p['F_svd'] for p in all_profiles], axis=0)
    mean_F_logPR = np.mean([p['F_logPR'] for p in all_profiles], axis=0)
    mean_F_old = np.mean([p['F_old'] for p in all_profiles], axis=0)
    mean_S_svd = np.mean([p['S_svd'] for p in all_profiles], axis=0)

    # Trend analysis
    slope_svd, _, r_svd, p_svd, _ = stats.linregress(layers_x, mean_F_svd)
    slope_logPR, _, r_logPR, p_logPR, _ = stats.linregress(layers_x, mean_F_logPR)
    slope_old, _, r_old, p_old, _ = stats.linregress(layers_x, mean_F_old)

    dF = np.diff(mean_F_svd)
    pct_dec_svd = np.sum(dF < 0) / len(dF) * 100

    print(f"\n=== Free Energy Comparison ===")
    print(f"  F_old (U-T^2):    slope={slope_old:.2f}")
    print(f"  F_logPR (U-TlnPR): slope={slope_logPR:.2f}")
    print(f"  F_svd (U-T*S_svd): slope={slope_svd:.2f}, {pct_dec_svd:.0f}% decreasing")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Three F definitions compared
    axes[0, 0].plot(layers_x, mean_F_old, 'gray', linewidth=1, linestyle='--',
                    label=f'F_old=U-T^2 (slope={slope_old:.1f})')
    axes[0, 0].plot(layers_x, mean_F_logPR, '#f39c12', linewidth=1.5,
                    label=f'F_logPR=U-TlnPR (slope={slope_logPR:.1f})')
    axes[0, 0].plot(layers_x, mean_F_svd, '#e74c3c', linewidth=2,
                    label=f'F_svd=U-T*S_svd (slope={slope_svd:.1f})')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Free Energy')
    axes[0, 0].set_title('(a) Three F Definitions')
    axes[0, 0].legend(fontsize=7)

    # (b) SVD entropy profile
    axes[0, 1].plot(layers_x, mean_S_svd, 'o-', color='#2ecc71', linewidth=2, markersize=3)
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('SVD Entropy')
    axes[0, 1].set_title('(b) SVD Entropy S_svd per Layer')

    # (c) dF/dl for SVD
    colors_df = ['#2ecc71' if d < 0 else '#e74c3c' for d in dF]
    axes[0, 2].bar(np.arange(len(dF)), dF, color=colors_df, alpha=0.7)
    axes[0, 2].axhline(y=0, color='black', linewidth=1)
    axes[0, 2].set_xlabel('Layer Transition')
    axes[0, 2].set_ylabel('dF_svd/dl')
    axes[0, 2].set_title(f'(c) F_svd Gradient ({pct_dec_svd:.0f}% decreasing)')

    # (d) Individual F_svd trajectories
    for p in all_profiles:
        axes[1, 0].plot(p['F_svd'], alpha=0.2, color='#e74c3c', linewidth=0.8)
    axes[1, 0].plot(mean_F_svd, 'k-', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('F_svd')
    axes[1, 0].set_title('(d) Individual F_svd Trajectories')
    axes[1, 0].legend()

    # (e) F_svd minimum location
    min_layers = [np.argmin(p['F_svd']) for p in all_profiles]
    axes[1, 1].hist(min_layers, bins=range(n_layers + 1), color='#9b59b6',
                    alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'(e) F_svd Min Layer (mean={np.mean(min_layers):.1f})')

    # (f) Summary comparison
    slopes = [slope_old, slope_logPR, slope_svd]
    names_short = ['U-T^2', 'U-TlnPR', 'U-T*S_svd']
    colors_bar = ['#95a5a6', '#f39c12', '#e74c3c']
    axes[1, 2].bar(names_short, slopes, color=colors_bar, alpha=0.8)
    axes[1, 2].axhline(y=0, color='black', linewidth=1)
    axes[1, 2].set_ylabel('Slope (negative = FEP)')
    axes[1, 2].set_title('(f) Slope Comparison')
    for i, v in enumerate(slopes):
        axes[1, 2].text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=9)

    is_fep = slope_svd < 0 and p_svd < 0.05
    fig.suptitle(f'Phase 66: SVD Entropy Free Energy '
                 f'({"FEP CONFIRMED" if is_fep else "FEP not confirmed"})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase66_svd_free_energy')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: F_svd slope={slope_svd:.2f} (p={p_svd:.2e}), {pct_dec_svd:.0f}% decreasing. "
          f"FEP {'CONFIRMED' if is_fep else 'STILL NOT confirmed'} with SVD entropy.")
    print(f"{'='*70}")

    save_results('phase66_svd_free_energy', {
        'experiment': 'SVD Entropy Free Energy',
        'summary': {
            'slope_svd': float(slope_svd), 'slope_logPR': float(slope_logPR),
            'slope_old': float(slope_old), 'pct_decreasing': float(pct_dec_svd),
            'is_fep': bool(is_fep),
        }
    })


if __name__ == '__main__':
    main()
