# -*- coding: utf-8 -*-
"""
Phase 135: Thermodynamic Signature of Hallucination
Can we predict when a model is about to hallucinate by measuring
thermodynamic quantities? Compare "factual" vs "hallucination-prone"
prompts and look for signatures in eta, kT, skewness, dissipation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure

# Factual prompts (model should be confident)
FACTUAL = [
    "The capital of France is",
    "Water freezes at zero degrees",
    "The sun is a star that provides",
    "Two plus two equals",
    "The Earth orbits around the",
    "Oxygen is essential for human",
]

# Ambiguous/uncertain prompts (hallucination-prone)
UNCERTAIN = [
    "The meaning of life according to most experts is",
    "The best approach to solve all problems is",
    "In the year 2050 the world will definitely be",
    "The most important discovery ever made was",
    "Everyone agrees that the best food is",
    "The correct political ideology is clearly",
]


def analyze_prompt_set(model, tok, prompts, device, n_layers):
    """Compute thermodynamic profile for a set of prompts."""
    results = {
        'eta': [[] for _ in range(n_layers)],
        'kT': [[] for _ in range(n_layers)],
        'S': [[] for _ in range(n_layers)],
        'skew': [[] for _ in range(n_layers)],
        'confidence': [],
    }

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            h = hs[0, -1, :].float()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)

            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(S): S = 0
            T_vals.append(S)
            results['S'][li].append(S)

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
            results['kT'][li].append(float(kT))

            # Skewness
            sk = sp_stats.skew(h.cpu().numpy())
            results['skew'][li].append(float(sk) if not np.isnan(sk) else 0)

        # Confidence = max prob of final output
        final_probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        results['confidence'].append(final_probs.max().item())

        # Eta
        for li in range(n_layers):
            T_subset = T_vals[:li+1]
            if len(T_subset) >= 4:
                T_hot = max(T_subset)
                T_cold = min(T_subset[len(T_subset)//2:])
                eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                eta = 0
            results['eta'][li].append(eta)

    return results


def main():
    print("=" * 70)
    print("Phase 135: Thermodynamic Signature of Hallucination")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    factual = analyze_prompt_set(model, tok, FACTUAL, device, n_layers)
    uncertain = analyze_prompt_set(model, tok, UNCERTAIN, device, n_layers)

    layers = np.arange(n_layers)

    # Compute differences
    avg = lambda x: [np.mean(v) if v else 0 for v in x]
    f_eta = avg(factual['eta'])
    u_eta = avg(uncertain['eta'])
    f_kT = avg(factual['kT'])
    u_kT = avg(uncertain['kT'])
    f_S = avg(factual['S'])
    u_S = avg(uncertain['S'])
    f_skew = avg(factual['skew'])
    u_skew = avg(uncertain['skew'])

    f_conf = np.mean(factual['confidence'])
    u_conf = np.mean(uncertain['confidence'])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Eta comparison
    axes[0,0].plot(layers, f_eta, 'o-', color='#27ae60', markersize=3, label='Factual')
    axes[0,0].plot(layers, u_eta, 's-', color='#c0392b', markersize=3, label='Uncertain')
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) Order Parameter')
    axes[0,0].legend()

    # (b) kT comparison
    axes[0,1].plot(layers, f_kT, 'o-', color='#27ae60', markersize=3, label='Factual')
    axes[0,1].plot(layers, u_kT, 's-', color='#c0392b', markersize=3, label='Uncertain')
    axes[0,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,1].set_ylabel('$kT$')
    axes[0,1].set_title('(b) Temperature')
    axes[0,1].legend()

    # (c) Entropy comparison
    axes[0,2].plot(layers, f_S, 'o-', color='#27ae60', markersize=3, label='Factual')
    axes[0,2].plot(layers, u_S, 's-', color='#c0392b', markersize=3, label='Uncertain')
    axes[0,2].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,2].set_ylabel('$S$')
    axes[0,2].set_title('(c) Output Entropy')
    axes[0,2].legend()

    # (d) Skewness comparison
    axes[1,0].plot(layers, f_skew, 'o-', color='#27ae60', markersize=3, label='Factual')
    axes[1,0].plot(layers, u_skew, 's-', color='#c0392b', markersize=3, label='Uncertain')
    axes[1,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,0].axhline(y=0, color='gray', linewidth=0.5)
    axes[1,0].set_ylabel('Skewness')
    axes[1,0].set_xlabel('Layer')
    axes[1,0].set_title('(d) Symmetry Breaking')
    axes[1,0].legend()

    # (e) Delta (uncertain - factual) for all quantities
    d_eta = np.array(u_eta) - np.array(f_eta)
    d_kT = np.array(u_kT) - np.array(f_kT)
    d_S = np.array(u_S) - np.array(f_S)
    axes[1,1].plot(layers, d_eta / (np.max(np.abs(d_eta)) + 1e-10), 'o-',
                   color='#c0392b', markersize=3, label='$\\Delta\\eta$')
    axes[1,1].plot(layers, d_S / (np.max(np.abs(d_S)) + 1e-10), 's-',
                   color='#2980b9', markersize=3, label='$\\Delta S$')
    axes[1,1].plot(layers, d_kT / (np.max(np.abs(d_kT)) + 1e-10), '^-',
                   color='#8e44ad', markersize=3, label='$\\Delta kT$')
    axes[1,1].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[1,1].axhline(y=0, color='gray', linewidth=0.5)
    axes[1,1].set_xlabel('Layer')
    axes[1,1].set_ylabel('Normalized Difference')
    axes[1,1].set_title('(e) Hallucination Signal')
    axes[1,1].legend(fontsize=7)

    # (f) Summary
    # Find the layer where the difference is largest
    max_diff_layer = np.argmax(np.abs(d_S))
    early_signal = np.mean(np.abs(d_S[:15]))
    late_signal = np.mean(np.abs(d_S[15:]))

    summary = (
        f"Hallucination Thermodynamics\n\n"
        f"Confidence: factual={f_conf:.3f}, uncertain={u_conf:.3f}\n\n"
        f"Final entropy: F={f_S[-1]:.2f}, U={u_S[-1]:.2f}\n"
        f"Final kT: F={f_kT[-1]:.2f}, U={u_kT[-1]:.2f}\n"
        f"Final eta: F={f_eta[-1]:.3f}, U={u_eta[-1]:.3f}\n\n"
        f"Max signal layer: L{max_diff_layer}\n"
        f"Early vs late signal: {early_signal:.3f} vs {late_signal:.3f}\n\n"
        f"Hallucination = {'higher kT' if u_kT[-1] > f_kT[-1] else 'lower kT'}\n"
        f"+ {'higher S' if u_S[-1] > f_S[-1] else 'lower S'}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 135: Thermodynamic Signature of Hallucination',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase135_hallucination')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Confidence: factual={f_conf:.3f}, uncertain={u_conf:.3f}")
    print(f"Final S: factual={f_S[-1]:.2f}, uncertain={u_S[-1]:.2f}")
    print(f"Final kT: factual={f_kT[-1]:.2f}, uncertain={u_kT[-1]:.2f}")
    print(f"{'='*70}")

    save_results('phase135_hallucination', {
        'experiment': 'Hallucination Thermodynamics',
        'summary': {
            'factual_conf': float(f_conf),
            'uncertain_conf': float(u_conf),
            'factual_S_final': float(f_S[-1]),
            'uncertain_S_final': float(u_S[-1]),
            'factual_kT_final': float(f_kT[-1]),
            'uncertain_kT_final': float(u_kT[-1]),
            'factual_eta_final': float(f_eta[-1]),
            'uncertain_eta_final': float(u_eta[-1]),
        }
    })


if __name__ == '__main__':
    main()
