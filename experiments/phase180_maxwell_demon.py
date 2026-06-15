# -*- coding: utf-8 -*-
"""
Phase 180: Maxwell's Attention Demon
Test whether the model can extract meaningful information from pure noise tokens.
Add random noise tokens as a "heat bath" and measure if attention extracts signal.
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
]


def measure_with_noise_bath(model, tok, prompt, device, n_noise_tokens=0):
    """Measure thermodynamics with noise tokens appended."""
    inp = tok(prompt, return_tensors='pt')
    input_ids = inp['input_ids'].to(device)

    if n_noise_tokens > 0:
        vocab_size = model.config.vocab_size
        noise_ids = torch.randint(100, vocab_size - 100, (1, n_noise_tokens), device=device)
        input_ids = torch.cat([input_ids, noise_ids], dim=1)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    n_layers = len(out.hidden_states)
    U_vals, T_vals = [], []

    for li in range(n_layers):
        hs = out.hidden_states[li]
        # Measure at the LAST token position (after noise)
        h = hs[0, -1, :].float()
        U = h.norm().item()
        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        U_vals.append(U if not np.isnan(U) else 0)
        T_vals.append(T if not np.isnan(T) else 0)

    # Confidence and top prediction
    final_logits = out.logits[0, -1, :].float()
    probs = torch.softmax(final_logits, dim=-1)
    confidence = probs.max().item()
    top_token = tok.decode([probs.argmax().item()])

    # Entropy of final prediction
    S_final = -(probs * torch.log(probs + 1e-10)).sum().item()

    # eta
    T_hot = np.mean(T_vals[:3])
    T_cold = np.mean(T_vals[-3:])
    eta = 1 - T_cold / (T_hot + 1e-10)

    return {
        'U': U_vals, 'T': T_vals, 'eta': float(eta),
        'confidence': float(confidence), 'S_final': float(S_final),
        'top_token': top_token,
    }


def main():
    print("=" * 70)
    print("Phase 180: Maxwell's Attention Demon")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    noise_counts = [0, 5, 10, 20, 50, 100]
    all_results = {}

    for n_noise in noise_counts:
        print(f"\n--- Noise tokens: {n_noise} ---")
        etas, confs, entropies = [], [], []

        for prompt in PROMPTS:
            r = measure_with_noise_bath(model, tok, prompt, device, n_noise_tokens=n_noise)
            etas.append(r['eta'])
            confs.append(r['confidence'])
            entropies.append(r['S_final'])

        all_results[n_noise] = {
            'mean_eta': float(np.mean(etas)),
            'mean_conf': float(np.mean(confs)),
            'mean_entropy': float(np.mean(entropies)),
            'std_eta': float(np.std(etas)),
            'etas': [float(x) for x in etas],
            'confs': [float(x) for x in confs],
        }
        print(f"  eta={np.mean(etas):.4f}, conf={np.mean(confs):.4f}, S={np.mean(entropies):.2f}")

    # Detailed profile for selected noise levels
    profile_results = {}
    for n_noise in [0, 20, 100]:
        r = measure_with_noise_bath(model, tok, PROMPTS[0], device, n_noise_tokens=n_noise)
        profile_results[n_noise] = r

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    nc = sorted(all_results.keys())

    # (a) eta vs noise
    axes[0, 0].errorbar(nc, [all_results[n]['mean_eta'] for n in nc],
                        yerr=[all_results[n]['std_eta'] for n in nc],
                        fmt='o-', color='#e74c3c', markersize=8, linewidth=2, capsize=4)
    axes[0, 0].axhline(y=0.813, color='black', linestyle='--', alpha=0.3, label='$\\eta = 0.813$')
    axes[0, 0].set_xlabel('Number of Noise Tokens')
    axes[0, 0].set_ylabel('$\\eta$')
    axes[0, 0].set_title('(a) Efficiency vs Noise Bath Size')
    axes[0, 0].legend(fontsize=8)

    # (b) Confidence vs noise
    axes[0, 1].plot(nc, [all_results[n]['mean_conf'] for n in nc],
                    's-', color='#2ecc71', markersize=8, linewidth=2)
    axes[0, 1].set_xlabel('Number of Noise Tokens')
    axes[0, 1].set_ylabel('Mean Confidence')
    axes[0, 1].set_title('(b) Confidence vs Noise Bath')

    # (c) Entropy vs noise
    axes[0, 2].plot(nc, [all_results[n]['mean_entropy'] for n in nc],
                    '^-', color='#3498db', markersize=8, linewidth=2)
    axes[0, 2].set_xlabel('Number of Noise Tokens')
    axes[0, 2].set_ylabel('Output Entropy $S$')
    axes[0, 2].set_title('(c) Prediction Entropy vs Noise')

    # (d) T profile with noise
    for n_noise in [0, 20, 100]:
        if n_noise in profile_results:
            axes[1, 0].plot(profile_results[n_noise]['T'], 'o-', markersize=3,
                            label=f'Noise={n_noise}', linewidth=2)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Temperature $T$')
    axes[1, 0].set_title('(d) Temperature Profile with Noise')
    axes[1, 0].legend(fontsize=8)

    # (e) U profile with noise
    for n_noise in [0, 20, 100]:
        if n_noise in profile_results:
            axes[1, 1].plot(profile_results[n_noise]['U'], 'o-', markersize=3,
                            label=f'Noise={n_noise}', linewidth=2)
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Energy $U$')
    axes[1, 1].set_title('(e) Energy Profile with Noise')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    base = all_results[0]
    max_noise = all_results[max(nc)]
    demon_power = base['mean_conf'] - max_noise['mean_conf']
    summary = (
        f"Maxwell's Attention Demon\n\n"
        f"BASELINE (0 noise):\n"
        f"  eta = {base['mean_eta']:.3f}\n"
        f"  conf = {base['mean_conf']:.4f}\n"
        f"  S = {base['mean_entropy']:.2f}\n\n"
        f"MAX NOISE ({max(nc)} tokens):\n"
        f"  eta = {max_noise['mean_eta']:.3f}\n"
        f"  conf = {max_noise['mean_conf']:.4f}\n"
        f"  S = {max_noise['mean_entropy']:.2f}\n\n"
        f"Demon extraction:\n"
        f"  conf drop = {demon_power:.4f}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 180: Maxwell's Attention Demon", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase180_maxwell_demon')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Baseline: eta={base['mean_eta']:.3f}, conf={base['mean_conf']:.4f}")
    print(f"Max noise: eta={max_noise['mean_eta']:.3f}, conf={max_noise['mean_conf']:.4f}")
    print(f"{'=' * 70}")

    save_results('phase180_maxwell_demon', {
        'experiment': "Maxwell's Attention Demon",
        'results': {str(k): v for k, v in all_results.items()},
    })


if __name__ == '__main__':
    main()
