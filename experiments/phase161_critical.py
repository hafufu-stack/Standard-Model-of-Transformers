# -*- coding: utf-8 -*-
"""
Phase 161: Critical Slowing Down
Near a phase transition, fluctuations increase and correlations grow.
Measure the variance (susceptibility) and autocorrelation of
thermodynamic quantities across prompts, layer by layer.
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
    "DNA encodes the instructions for all living organisms",
    "Thermodynamics governs the flow of energy and entropy",
    "The brain processes information through neural circuits",
    "Climate change is driven by greenhouse gas emissions",
]


def main():
    print("=" * 70)
    print("Phase 161: Critical Slowing Down")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    # Collect per-prompt, per-layer measurements
    all_S = np.zeros((len(PROMPTS), n_layers))
    all_kT = np.zeros((len(PROMPTS), n_layers))
    all_eta = np.zeros((len(PROMPTS), n_layers))

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            all_S[pi, li] = S if not np.isnan(S) else 0
            T_vals.append(all_S[pi, li])

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
            all_kT[pi, li] = float(kT)

        for li in range(n_layers):
            T_sub = T_vals[:li+1]
            if len(T_sub) >= 4:
                T_hot = max(T_sub)
                T_cold = min(T_sub[len(T_sub)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0
            all_eta[pi, li] = eta

    # Compute fluctuation metrics
    var_S = np.var(all_S, axis=0)      # Susceptibility of S
    var_kT = np.var(all_kT, axis=0)    # Susceptibility of kT
    var_eta = np.var(all_eta, axis=0)   # Susceptibility of eta
    mean_S = np.mean(all_S, axis=0)
    mean_kT = np.mean(all_kT, axis=0)

    # Autocorrelation: correlation between layer l and l+1
    autocorr_S = []
    for li in range(n_layers - 1):
        r = np.corrcoef(all_S[:, li], all_S[:, li+1])[0, 1]
        autocorr_S.append(r if not np.isnan(r) else 0)

    # Correlation length: how many layers ahead is S correlated?
    corr_length = []
    for li in range(n_layers):
        length = 0
        for dl in range(1, min(5, n_layers - li)):
            r = np.corrcoef(all_S[:, li], all_S[:, li+dl])[0, 1]
            if not np.isnan(r) and abs(r) > 0.5:
                length = dl
            else:
                break
        corr_length.append(length)

    layers = np.arange(n_layers)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Susceptibility of S
    axes[0,0].plot(layers, var_S, 'o-', color='#c0392b', markersize=4, linewidth=2)
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--', label='$L_0$')
    peak_layer = np.argmax(var_S[4:]) + 4
    axes[0,0].axvline(x=peak_layer, color='#27ae60', linewidth=1.5, linestyle=':',
                      label=f'Peak L{peak_layer}')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('Var($S$)')
    axes[0,0].set_title('(a) Entropy Susceptibility')
    axes[0,0].legend()

    # (b) Susceptibility of eta
    axes[0,1].plot(layers, var_eta, 'o-', color='#8e44ad', markersize=4, linewidth=2)
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    peak_eta = np.argmax(var_eta[4:]) + 4
    axes[0,1].axvline(x=peak_eta, color='#27ae60', linewidth=1.5, linestyle=':',
                      label=f'Peak L{peak_eta}')
    axes[0,1].set_xlabel('Layer')
    axes[0,1].set_ylabel('Var($\\eta$)')
    axes[0,1].set_title('(b) Eta Susceptibility')
    axes[0,1].legend()

    # (c) Susceptibility of kT
    axes[0,2].plot(layers, var_kT, 'o-', color='#2980b9', markersize=4, linewidth=2)
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    peak_kT = np.argmax(var_kT[4:]) + 4
    axes[0,2].axvline(x=peak_kT, color='#27ae60', linewidth=1.5, linestyle=':',
                      label=f'Peak L{peak_kT}')
    axes[0,2].set_xlabel('Layer')
    axes[0,2].set_ylabel('Var($kT$)')
    axes[0,2].set_title('(c) Temperature Susceptibility')
    axes[0,2].legend()

    # (d) Autocorrelation
    axes[1,0].plot(range(len(autocorr_S)), autocorr_S, 'o-', color='#27ae60',
                  markersize=4, linewidth=2)
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].axhline(y=0, color='gray', linewidth=0.5)
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_ylabel('$C(l, l+1)$')
    axes[1,0].set_title('(d) Autocorrelation')

    # (e) Correlation length
    axes[1,1].plot(layers, corr_length, 'o-', color='#e74c3c', markersize=4, linewidth=2)
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Correlation Length')
    axes[1,1].set_title('(e) Correlation Length')

    # (f) Summary
    summary = (
        f"Critical Slowing Down\n\n"
        f"Susceptibility peaks:\n"
        f"  Var(S) peak: L{peak_layer}\n"
        f"  Var(eta) peak: L{peak_eta}\n"
        f"  Var(kT) peak: L{peak_kT}\n\n"
        f"L0 = 21.7\n"
        f"Peak distance from L0:\n"
        f"  |L_peak - L0| = {abs(peak_layer - 21.7):.1f}\n\n"
        f"Critical slowing\n"
        f"{'CONFIRMED' if abs(peak_layer - 21.7) < 5 else 'NOT found'}\n"
        f"near L0"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 161: Critical Slowing Down',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase161_critical')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Var(S) peak: L{peak_layer}")
    print(f"Var(eta) peak: L{peak_eta}")
    print(f"Var(kT) peak: L{peak_kT}")
    print(f"{'='*70}")

    save_results('phase161_critical', {
        'experiment': 'Critical Slowing Down',
        'summary': {
            'peak_S': int(peak_layer),
            'peak_eta': int(peak_eta),
            'peak_kT': int(peak_kT),
        }
    })


if __name__ == '__main__':
    main()
