# -*- coding: utf-8 -*-
"""
Phase 49: Negative Specific Heat (Formal Verification)
Formally measure C_v = dU/dT across many prompts to prove LLMs are
self-gravitating systems (negative specific heat, like stars and black holes).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 49: Negative Specific Heat (Formal Verification)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Diverse prompts for statistical power
    prompts = [
        "The theory of general relativity describes how massive objects warp spacetime around",
        "In quantum mechanics, the wave function collapse occurs when a measurement is",
        "The human genome contains approximately three billion base pairs of DNA that encode",
        "Artificial neural networks are inspired by the biological structure of the brain and",
        "The standard model of particle physics classifies all known elementary particles into",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from",
        "The Turing test evaluates whether a machine can exhibit intelligent behavior indistinguishable from",
        "Black holes form when massive stars exhaust their nuclear fuel and undergo gravitational",
        "The central dogma of molecular biology describes the flow of genetic information from DNA to",
        "Climate models use differential equations to simulate atmospheric dynamics and predict future",
        "The Fibonacci sequence appears throughout nature in the arrangement of leaves and the spirals of",
        "Cryptographic hash functions transform arbitrary input data into fixed-size output that is computationally",
        "Evolution by natural selection operates on heritable variation within populations over many",
        "The Navier-Stokes equations describe the motion of viscous fluid substances and remain one of",
        "Reinforcement learning agents maximize cumulative reward by learning optimal policies through trial and",
        "The cosmic microwave background radiation provides a snapshot of the universe approximately 380000 years after",
    ]

    n_layers = len(model.model.layers)
    # Collect U(l) and T(l) for every (prompt, layer) pair
    all_U_profiles = []  # (n_prompts, n_layers+1)
    all_T_profiles = []

    for pi, prompt in enumerate(prompts):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        Us = []
        Ts = []

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()
            Us.append(U)

            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T_val):
                T_val = 0.0
            Ts.append(T_val)

        all_U_profiles.append(Us)
        all_T_profiles.append(Ts)

    all_U = np.array(all_U_profiles)  # (n_prompts, n_layers+1)
    all_T = np.array(all_T_profiles)

    # Compute mean profiles
    mean_U = np.mean(all_U, axis=0)
    mean_T = np.mean(all_T, axis=0)
    std_U = np.std(all_U, axis=0)
    std_T = np.std(all_T, axis=0)

    # === Key measurement: dU/dT via linear regression ===
    # Method 1: Layer-wise regression (U vs T across layers)
    valid = (mean_T > 0.1) & ~np.isnan(mean_U)
    slope_layer, intercept_layer, r_val, p_val, std_err = stats.linregress(
        mean_T[valid], mean_U[valid])
    Cv_layer = slope_layer  # This is dU/dT

    # Method 2: Prompt-wise regression at each layer
    Cv_per_layer = []
    for li in range(len(mean_U)):
        U_col = all_U[:, li]
        T_col = all_T[:, li]
        mask = (T_col > 0.1) & ~np.isnan(U_col)
        if mask.sum() >= 5:
            s, _, r, p, _ = stats.linregress(T_col[mask], U_col[mask])
            Cv_per_layer.append({'layer': li, 'Cv': float(s), 'r': float(r), 'p': float(p)})
        else:
            Cv_per_layer.append({'layer': li, 'Cv': 0, 'r': 0, 'p': 1})

    # Method 3: Delta-based (consecutive layers)
    dU = np.diff(mean_U)
    dT = np.diff(mean_T)
    # Avoid division by zero
    safe = np.abs(dT) > 1e-6
    local_Cv = np.where(safe, dU / dT, 0)

    # Count negative Cv layers
    n_negative = np.sum(local_Cv < 0)
    n_total = len(local_Cv)
    pct_negative = n_negative / n_total * 100

    print(f"\n=== Specific Heat Analysis ===")
    print(f"  Layer-wise regression: C_v = dU/dT = {Cv_layer:.2f} (r={r_val:.3f}, p={p_val:.2e})")
    print(f"  Local dU/dT: {pct_negative:.0f}% of layer transitions have C_v < 0")
    print(f"  Mean local C_v: {np.mean(local_Cv):.2f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) U vs T (state diagram)
    axes[0, 0].scatter(mean_T[valid], mean_U[valid], c=np.arange(valid.sum()),
                       cmap='viridis', s=30, edgecolors='black', linewidth=0.5)
    T_fit = np.linspace(mean_T[valid].min(), mean_T[valid].max(), 50)
    axes[0, 0].plot(T_fit, slope_layer * T_fit + intercept_layer, 'r--',
                    linewidth=2, label=f'C_v = {Cv_layer:.1f}')
    axes[0, 0].set_xlabel('Temperature T (Logit Entropy)')
    axes[0, 0].set_ylabel('Internal Energy U (L2 Norm)')
    axes[0, 0].set_title(f'(a) Equation of State (r={r_val:.3f})')
    axes[0, 0].legend()
    cb = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cb.set_label('Layer Index')

    # (b) U and T profiles
    layers_x = np.arange(len(mean_U))
    axes[0, 1].plot(layers_x, mean_U, 'b-', linewidth=1.5, label='U (Internal Energy)')
    axes[0, 1].fill_between(layers_x, mean_U - std_U, mean_U + std_U, alpha=0.2, color='blue')
    ax_t = axes[0, 1].twinx()
    ax_t.plot(layers_x, mean_T, 'r-', linewidth=1.5, label='T (Temperature)')
    ax_t.fill_between(layers_x, mean_T - std_T, mean_T + std_T, alpha=0.2, color='red')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('U (L2 Norm)', color='blue')
    ax_t.set_ylabel('T (Entropy)', color='red')
    axes[0, 1].set_title('(b) U increases, T decreases')

    # (c) Local C_v per layer transition
    trans_x = np.arange(len(local_Cv))
    colors_cv = ['#e74c3c' if cv < 0 else '#2ecc71' for cv in local_Cv]
    axes[0, 2].bar(trans_x, local_Cv, color=colors_cv, alpha=0.7, width=0.8)
    axes[0, 2].axhline(y=0, color='black', linewidth=1)
    axes[0, 2].set_xlabel('Layer Transition (l -> l+1)')
    axes[0, 2].set_ylabel('Local C_v = dU/dT')
    axes[0, 2].set_title(f'(c) {pct_negative:.0f}% Negative C_v')

    # (d) Per-prompt U-T scatter at key layers
    key_layers = [0, len(mean_U)//4, len(mean_U)//2, 3*len(mean_U)//4, len(mean_U)-1]
    for li in key_layers:
        axes[1, 0].scatter(all_T[:, li], all_U[:, li], s=15, alpha=0.6,
                          label=f'L{li}')
    axes[1, 0].set_xlabel('T')
    axes[1, 0].set_ylabel('U')
    axes[1, 0].set_title('(d) Prompt-wise U vs T at Key Layers')
    axes[1, 0].legend(fontsize=7)

    # (e) Per-layer C_v (prompt-wise regression)
    Cv_vals = [r['Cv'] for r in Cv_per_layer]
    Cv_layers = [r['layer'] for r in Cv_per_layer]
    colors_cv2 = ['#e74c3c' if cv < 0 else '#2ecc71' for cv in Cv_vals]
    axes[1, 1].bar(Cv_layers, Cv_vals, color=colors_cv2, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linewidth=1)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('C_v (prompt-wise regression)')
    axes[1, 1].set_title('(e) Prompt-wise Specific Heat')

    # (f) Summary: histogram of all local C_v
    axes[1, 2].hist(local_Cv[np.isfinite(local_Cv)], bins=20, color='#3498db',
                    alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(x=0, color='red', linewidth=2, linestyle='--', label='C_v = 0')
    axes[1, 2].axvline(x=np.mean(local_Cv), color='orange', linewidth=2,
                       label=f'Mean = {np.mean(local_Cv):.1f}')
    axes[1, 2].set_xlabel('C_v = dU/dT')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('(f) C_v Distribution')
    axes[1, 2].legend()

    fig.suptitle('Phase 49: Negative Specific Heat Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase49_negative_specific_heat')
    plt.close()

    # === Verdict ===
    is_negative = Cv_layer < 0
    is_significant = p_val < 0.05
    is_consistent = pct_negative > 60

    print(f"\n{'='*70}")
    print(f"VERDICT: C_v = {Cv_layer:.2f} (p={p_val:.2e}), "
          f"{pct_negative:.0f}% layer transitions have C_v < 0. "
          f"{'CONFIRMED' if is_negative and is_significant else 'NOT confirmed'}: "
          f"LLM {'exhibits' if is_negative else 'does not exhibit'} negative specific heat "
          f"(self-gravitating system).")
    print(f"{'='*70}")

    save_results('phase49_negative_specific_heat', {
        'experiment': 'Negative Specific Heat',
        'Cv_global': float(Cv_layer),
        'r_value': float(r_val),
        'p_value': float(p_val),
        'pct_negative_transitions': float(pct_negative),
        'mean_local_Cv': float(np.mean(local_Cv)),
        'Cv_per_layer': Cv_per_layer,
        'summary': {
            'is_negative': bool(is_negative),
            'is_significant': bool(is_significant),
            'pct_negative': float(pct_negative),
        }
    })


if __name__ == '__main__':
    main()
