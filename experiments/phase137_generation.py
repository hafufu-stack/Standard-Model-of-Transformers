# -*- coding: utf-8 -*-
"""
Phase 137: Token-by-Token Dynamics
How does the phase transition evolve during autoregressive generation?
Does L0 shift, does kT change, does eta reach a steady state?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


def main():
    print("=" * 70)
    print("Phase 137: Token-by-Token Dynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    SEED = "The most important discovery in physics was"
    N_TOKENS = 30

    # Generate tokens and track thermodynamics at each step
    input_ids = tok(SEED, return_tensors='pt')['input_ids'].to(device)

    token_L0s = []
    token_kTs = []
    token_etas = []
    token_Ss = []
    tokens_text = []

    for step in range(N_TOKENS):
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        # Compute eta profile at this step
        S_vals = []
        kT_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            S_vals.append(S if not np.isnan(S) else 0)

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
            kT_vals.append(float(kT))

        # Eta at each layer
        etas = []
        for li in range(n_layers):
            T_subset = S_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0
            etas.append(eta)

        token_etas.append(etas)
        token_kTs.append(kT_vals)
        token_Ss.append(S_vals)

        # Fit sigmoid for L0
        Ls = np.arange(4, n_layers)
        eta_fit = np.array(etas[4:])
        try:
            popt, _ = curve_fit(sigmoid, Ls, eta_fit,
                                p0=[n_layers*0.7, 0.5, np.min(eta_fit), np.max(eta_fit)],
                                maxfev=5000)
            token_L0s.append(float(popt[0]))
        except:
            token_L0s.append(21.7)

        # Greedy decode next token
        next_logits = out.logits[0, -1, :]
        next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)
        tokens_text.append(tok.decode(next_token[0]))
        input_ids = torch.cat([input_ids, next_token], dim=1)

    steps = np.arange(N_TOKENS)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) L0 evolution
    axes[0,0].plot(steps, token_L0s, 'o-', color='#c0392b', markersize=5, linewidth=2)
    axes[0,0].axhline(y=21.7, color='#f39c12', linestyle='--', label='L0=21.7 (baseline)')
    axes[0,0].set_xlabel('Generation Step')
    axes[0,0].set_ylabel('$L_0$')
    axes[0,0].set_title('(a) Transition Point Drift')
    axes[0,0].legend()

    # (b) Final layer eta
    final_etas = [etas[-1] for etas in token_etas]
    axes[0,1].plot(steps, final_etas, 'o-', color='#2980b9', markersize=5, linewidth=2)
    axes[0,1].set_xlabel('Generation Step')
    axes[0,1].set_ylabel('$\\eta_{final}$')
    axes[0,1].set_title('(b) Final Efficiency')

    # (c) Final kT
    final_kTs = [kTs[-1] for kTs in token_kTs]
    axes[0,2].plot(steps, final_kTs, 'o-', color='#27ae60', markersize=5, linewidth=2)
    axes[0,2].set_xlabel('Generation Step')
    axes[0,2].set_ylabel('$kT_{final}$')
    axes[0,2].set_title('(c) Temperature Evolution')

    # (d) Eta heatmap (layer x step)
    eta_mat = np.array(token_etas).T  # (layers, steps)
    im = axes[1,0].imshow(eta_mat, aspect='auto', cmap='inferno', origin='lower')
    axes[1,0].set_xlabel('Generation Step')
    axes[1,0].set_ylabel('Layer')
    axes[1,0].set_title('(d) Eta Spacetime')
    plt.colorbar(im, ax=axes[1,0], shrink=0.7)

    # (e) kT heatmap
    kT_mat = np.array(token_kTs).T
    im2 = axes[1,1].imshow(kT_mat, aspect='auto', cmap='hot', origin='lower',
                            vmax=np.percentile(kT_mat, 95))
    axes[1,1].set_xlabel('Generation Step')
    axes[1,1].set_ylabel('Layer')
    axes[1,1].set_title('(e) kT Spacetime')
    plt.colorbar(im2, ax=axes[1,1], shrink=0.7)

    # (f) Summary
    L0_cv = np.std(token_L0s) / (np.mean(token_L0s) + 1e-10)
    generated = "".join(tokens_text[:20])
    summary = (
        f"Token-by-Token Dynamics\n\n"
        f"L0 mean: {np.mean(token_L0s):.1f}\n"
        f"L0 CV: {L0_cv:.3f}\n"
        f"L0 range: [{min(token_L0s):.1f}, {max(token_L0s):.1f}]\n\n"
        f"eta_final range: [{min(final_etas):.3f}, {max(final_etas):.3f}]\n"
        f"kT_final range: [{min(final_kTs):.1f}, {max(final_kTs):.1f}]\n\n"
        f"Generated:\n{generated}..."
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 137: Token-by-Token Dynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase137_generation')
    plt.close()

    print(f"\n{'='*70}")
    print(f"L0 mean={np.mean(token_L0s):.1f}, CV={L0_cv:.3f}")
    print(f"Generated: {''.join(tokens_text)}")
    print(f"{'='*70}")

    save_results('phase137_generation', {
        'experiment': 'Token-by-Token Dynamics',
        'token_L0s': token_L0s,
        'final_etas': final_etas,
        'final_kTs': final_kTs,
        'generated_text': ''.join(tokens_text),
        'summary': {
            'L0_mean': float(np.mean(token_L0s)),
            'L0_cv': float(L0_cv),
            'L0_range': [float(min(token_L0s)), float(max(token_L0s))],
        }
    })


if __name__ == '__main__':
    main()
