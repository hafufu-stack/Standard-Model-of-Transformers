# -*- coding: utf-8 -*-
"""
Phase 85: Layer-Dependent Ergodicity Breaking
Test whether ergodicity holds at every layer or breaks at a critical depth.
Phase 83 tested at a single layer; here we test L0-L_max systematically.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

# Short prompts (ensemble)
ENSEMBLE_PROMPTS = [
    "The capital of France is", "Water boils at one hundred",
    "The speed of light equals", "Photosynthesis converts sunlight into",
    "DNA stands for deoxyribonucleic", "The largest planet is Jupiter",
    "Gravity pulls objects toward the", "The periodic table organizes",
    "Machine learning uses data to", "Black holes form when massive",
    "The human genome contains three", "Evolution works through natural",
    "Quantum mechanics describes particles at", "Neural networks learn through",
    "The ocean covers seventy percent", "Electricity flows through conductors",
    "The mitochondria is the powerhouse", "Chemical reactions involve breaking",
    "Plate tectonics drives continental", "The speed of sound in air",
]

# Long prompts (time series)
TIME_SERIES_PROMPTS = [
    "The history of mathematics spans thousands of years and encompasses contributions from civilizations around the world including the ancient Egyptians Greeks Indians and Chinese who each developed unique approaches to numerical reasoning geometric proof and algebraic manipulation that together form the foundation of modern mathematical thought and continue to influence scientific discovery and technological innovation in profound and often unexpected ways",
    "Quantum computing promises to revolutionize information processing by harnessing the principles of quantum mechanics including superposition entanglement and quantum interference to perform calculations that would be practically impossible for classical computers opening new frontiers in cryptography drug discovery materials science and artificial intelligence",
    "The Amazon rainforest contains the greatest biodiversity on Earth with millions of species of plants animals insects and microorganisms interacting in complex ecological networks that regulate global climate patterns water cycles and atmospheric chemistry making its preservation critically important for the future of all life on this planet",
]


def main():
    print("=" * 70)
    print("Phase 85: Layer-Dependent Ergodicity Breaking")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1  # +1 for embedding layer

    # === Ensemble: collect T, PR at every layer for each prompt ===
    print("  Collecting ensemble statistics...")
    ens_T = {l: [] for l in range(n_layers)}  # layer -> list of T values
    ens_PR = {l: [] for l in range(n_layers)}

    for prompt in ENSEMBLE_PROMPTS:
        thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
        for r in thermo:
            li = r['layer']
            ens_T[li].append(r['T'])
            ens_PR[li].append(r['PR'])

    # === Time series: collect T, PR at every layer for each token position ===
    print("  Collecting time series statistics...")
    tim_T = {l: [] for l in range(n_layers)}
    tim_PR = {l: [] for l in range(n_layers)}

    for prompt in TIME_SERIES_PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        seq_len = inp['input_ids'].shape[1]
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for li, hs in enumerate(out.hidden_states):
            if li >= n_layers:
                break
            # Use each token position as a "time" sample
            for pos in range(seq_len):
                h = hs[0, pos, :].float()
                h_sq = h ** 2
                h_prob = h_sq / (h_sq.sum() + 1e-10)
                pr = 1.0 / (h_prob ** 2).sum().item()
                tim_PR[li].append(pr)

                # T from logits at this layer
                with torch.no_grad():
                    normed = model.model.norm(hs[:, pos:pos+1, :])
                    logits = model.lm_head(normed).squeeze().float()
                probs = torch.softmax(logits, dim=-1)
                t_val = -(probs * torch.log(probs + 1e-10)).sum().item()
                if np.isnan(t_val):
                    t_val = 0.0
                tim_T[li].append(t_val)

    # === KS tests at each layer ===
    print("  Running KS tests...")
    results_per_layer = []
    for li in range(n_layers):
        e_t = np.array(ens_T[li])
        t_t = np.array(tim_T[li])
        e_pr = np.array(ens_PR[li])
        t_pr = np.array(tim_PR[li])

        ks_T = sp_stats.ks_2samp(e_t, t_t) if len(e_t) > 2 and len(t_t) > 2 else (0, 1)
        ks_PR = sp_stats.ks_2samp(e_pr, t_pr) if len(e_pr) > 2 and len(t_pr) > 2 else (0, 1)

        results_per_layer.append({
            'layer': li,
            'T_ks_stat': float(ks_T.statistic) if hasattr(ks_T, 'statistic') else float(ks_T[0]),
            'T_ks_p': float(ks_T.pvalue) if hasattr(ks_T, 'pvalue') else float(ks_T[1]),
            'PR_ks_stat': float(ks_PR.statistic) if hasattr(ks_PR, 'statistic') else float(ks_PR[0]),
            'PR_ks_p': float(ks_PR.pvalue) if hasattr(ks_PR, 'pvalue') else float(ks_PR[1]),
            'T_ergodic': bool((ks_T.pvalue if hasattr(ks_T, 'pvalue') else ks_T[1]) > 0.05),
            'PR_ergodic': bool((ks_PR.pvalue if hasattr(ks_PR, 'pvalue') else ks_PR[1]) > 0.05),
            'ens_T_mean': float(np.mean(e_t)),
            'tim_T_mean': float(np.mean(t_t)),
            'ens_PR_mean': float(np.mean(e_pr)),
            'tim_PR_mean': float(np.mean(t_pr)),
        })

    # === Visualization ===
    layers = [r['layer'] for r in results_per_layer]
    T_ps = [r['T_ks_p'] for r in results_per_layer]
    PR_ps = [r['PR_ks_p'] for r in results_per_layer]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) KS p-values
    axes[0].plot(layers, T_ps, 'o-', color='#c0392b', linewidth=2, markersize=4, label='$T$ (temperature)')
    axes[0].plot(layers, PR_ps, 's-', color='#2980b9', linewidth=2, markersize=4, label='$PR$ (participation)')
    axes[0].axhline(y=0.05, color='gray', linestyle='--', linewidth=2, label='$p = 0.05$')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('KS test p-value')
    axes[0].set_title('(a) Ergodicity by Layer')
    axes[0].legend(fontsize=8)
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=1e-6)

    # (b) T pass/fail map
    T_pass = [1 if r['T_ergodic'] else 0 for r in results_per_layer]
    PR_pass = [1 if r['PR_ergodic'] else 0 for r in results_per_layer]
    axes[1].bar(np.array(layers) - 0.2, T_pass, 0.35, color='#c0392b', alpha=0.7, label='T ergodic')
    axes[1].bar(np.array(layers) + 0.2, PR_pass, 0.35, color='#2980b9', alpha=0.7, label='PR ergodic')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Ergodic (1=Pass)')
    axes[1].set_title('(b) Pass/Fail Map')
    axes[1].legend(fontsize=8)

    # (c) Mean comparison
    ens_T_means = [r['ens_T_mean'] for r in results_per_layer]
    tim_T_means = [r['tim_T_mean'] for r in results_per_layer]
    axes[2].plot(layers, ens_T_means, 'o-', color='#c0392b', label='Ensemble T')
    axes[2].plot(layers, tim_T_means, 's--', color='#e74c3c', label='Time series T')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Mean T')
    axes[2].set_title('(c) Ensemble vs Time Mean')
    axes[2].legend(fontsize=8)

    T_ergodic_count = sum(1 for r in results_per_layer if r['T_ergodic'])
    PR_ergodic_count = sum(1 for r in results_per_layer if r['PR_ergodic'])
    fig.suptitle(f'Phase 85: Layer-Dependent Ergodicity (T: {T_ergodic_count}/{n_layers}, PR: {PR_ergodic_count}/{n_layers} ergodic)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase85_ergodicity_layers')
    plt.close()

    # Find breaking point
    T_break = None
    for r in results_per_layer:
        if not r['T_ergodic'] and r['layer'] > 0:
            if T_break is None:
                T_break = r['layer']

    print(f"\n{'='*70}")
    print(f"T ergodic: {T_ergodic_count}/{n_layers} layers")
    print(f"PR ergodic: {PR_ergodic_count}/{n_layers} layers")
    if T_break:
        print(f"T ergodicity breaking point: Layer {T_break}")
    print(f"{'='*70}")

    save_results('phase85_ergodicity_layers', {
        'experiment': 'Layer-Dependent Ergodicity Breaking',
        'per_layer': results_per_layer,
        'summary': {
            'n_layers': n_layers,
            'T_ergodic_count': T_ergodic_count,
            'PR_ergodic_count': PR_ergodic_count,
            'T_breaking_layer': T_break,
        }
    })


if __name__ == '__main__':
    main()
