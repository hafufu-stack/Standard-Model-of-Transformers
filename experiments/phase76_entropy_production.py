# -*- coding: utf-8 -*-
"""
Phase 76: Entropy Production Rate (Second Law of Thermodynamics)
The 2nd law states dS_total >= 0 (entropy of the universe always increases).
For LLMs: the TOTAL entropy (hidden state + output) should never decrease.
But internal entropy CAN decrease if external entropy increases more.
This proves the "information refrigerator" (P73) obeys the 2nd law.
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
    print("Phase 76: Entropy Production Rate (2nd Law)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration",
        "Quantum mechanics describes the behavior of particles at the atomic scale",
        "The human genome contains approximately three billion base pairs encoding",
        "Artificial neural networks process information through layers of connected",
        "Black holes form when massive stars exhaust their nuclear fuel and collapse",
        "The periodic table organizes chemical elements by their atomic number",
        "Evolution by natural selection operates on heritable variation within",
        "Climate change affects ecosystems worldwide through rising temperatures",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen",
        "Machine learning algorithms discover patterns in data without explicit",
        "The cosmic microwave background provides a snapshot of the early universe",
        "General relativity describes gravity as curvature of spacetime caused by",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        S_internal = []  # hidden state entropy
        S_external = []  # output (logit) entropy
        S_total = []

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()

            # Internal entropy: distribution of hidden state activations
            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S_int = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

            # External entropy: logit entropy (output distribution)
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S_ext = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(S_ext):
                S_ext = 0

            S_tot = S_int + S_ext

            S_internal.append(S_int)
            S_external.append(S_ext)
            S_total.append(S_tot)

        # Check 2nd law: dS_total >= 0 at each layer?
        dS_total = [S_total[i+1] - S_total[i] for i in range(len(S_total)-1)]
        dS_internal = [S_internal[i+1] - S_internal[i] for i in range(len(S_internal)-1)]
        dS_external = [S_external[i+1] - S_external[i] for i in range(len(S_external)-1)]

        pct_2nd_law = sum(1 for d in dS_total if d >= 0) / len(dS_total) * 100

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        print(f"  '{safe_p}...' 2nd law: {pct_2nd_law:.0f}% layers")

        all_results.append({
            'prompt': prompt[:60],
            'S_internal': S_internal, 'S_external': S_external, 'S_total': S_total,
            'dS_total': dS_total, 'dS_internal': dS_internal, 'dS_external': dS_external,
            'pct_2nd_law': float(pct_2nd_law),
        })

    n_layers = len(all_results[0]['S_total'])
    overall_2nd = np.mean([r['pct_2nd_law'] for r in all_results])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    layers_x = np.arange(n_layers)

    # (a) Mean entropy profiles
    mean_Si = np.mean([r['S_internal'] for r in all_results], axis=0)
    mean_Se = np.mean([r['S_external'] for r in all_results], axis=0)
    mean_St = np.mean([r['S_total'] for r in all_results], axis=0)
    axes[0, 0].plot(layers_x, mean_Si, 'b-', linewidth=2, label='S_internal')
    axes[0, 0].plot(layers_x, mean_Se, 'r-', linewidth=2, label='S_external')
    axes[0, 0].plot(layers_x, mean_St, 'k-', linewidth=3, label='S_total')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('(a) Entropy Decomposition')
    axes[0, 0].legend()

    # (b) dS_total per layer
    mean_dSt = np.mean([r['dS_total'] for r in all_results], axis=0)
    colors_ds = ['#2ecc71' if d >= 0 else '#e74c3c' for d in mean_dSt]
    axes[0, 1].bar(np.arange(len(mean_dSt)), mean_dSt, color=colors_ds, alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linewidth=1)
    axes[0, 1].set_xlabel('Layer Transition')
    axes[0, 1].set_ylabel('dS_total')
    axes[0, 1].set_title(f'(b) dS_total ({overall_2nd:.0f}% >= 0)')

    # (c) Internal vs External entropy change
    mean_dSi = np.mean([r['dS_internal'] for r in all_results], axis=0)
    mean_dSe = np.mean([r['dS_external'] for r in all_results], axis=0)
    axes[0, 2].plot(mean_dSi, mean_dSe, 'o', color='#9b59b6', markersize=6)
    axes[0, 2].axhline(y=0, color='gray', linestyle='--')
    axes[0, 2].axvline(x=0, color='gray', linestyle='--')
    # Quadrants: top-left = refrigerator (internal decreases, external increases)
    axes[0, 2].set_xlabel('dS_internal')
    axes[0, 2].set_ylabel('dS_external')
    axes[0, 2].set_title('(c) Entropy Flow Phase Space')
    axes[0, 2].text(0.05, 0.95, 'Refrigerator\n(pump out)', transform=axes[0, 2].transAxes,
                   fontsize=8, va='top', color='blue')
    axes[0, 2].text(0.95, 0.05, 'Engine\n(pump in)', transform=axes[0, 2].transAxes,
                   fontsize=8, va='bottom', ha='right', color='red')

    # (d) S_internal profile (drops = information concentration)
    for r in all_results:
        axes[1, 0].plot(r['S_internal'], alpha=0.3, color='#3498db', linewidth=0.8)
    axes[1, 0].plot(mean_Si, 'k-', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('S_internal')
    axes[1, 0].set_title('(d) Internal Entropy (concentration)')
    axes[1, 0].legend()

    # (e) S_external profile (drops = output sharpening)
    for r in all_results:
        axes[1, 1].plot(r['S_external'], alpha=0.3, color='#e74c3c', linewidth=0.8)
    axes[1, 1].plot(mean_Se, 'k-', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('S_external')
    axes[1, 1].set_title('(e) External Entropy (output)')
    axes[1, 1].legend()

    # (f) 2nd law compliance per prompt
    pcts = sorted([r['pct_2nd_law'] for r in all_results])
    axes[1, 2].bar(range(len(pcts)), pcts, color='#2ecc71', alpha=0.7)
    axes[1, 2].axhline(y=50, color='red', linestyle='--', label='Random (50%)')
    axes[1, 2].set_xlabel('Prompt (sorted)')
    axes[1, 2].set_ylabel('% layers obeying 2nd law')
    axes[1, 2].set_title(f'(f) 2nd Law Compliance ({overall_2nd:.0f}%)')
    axes[1, 2].legend()

    holds = overall_2nd > 60
    fig.suptitle(f'Phase 76: Entropy Production Rate (2nd Law: {overall_2nd:.0f}% compliance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase76_entropy_production')
    plt.close()

    # Entropy change direction
    mean_Si_change = mean_Si[-1] - mean_Si[0]
    mean_Se_change = mean_Se[-1] - mean_Se[0]

    print(f"\n{'='*70}")
    print(f"VERDICT: 2nd law compliance={overall_2nd:.0f}%. "
          f"S_int {'decreases' if mean_Si_change < 0 else 'increases'} ({mean_Si_change:.1f}), "
          f"S_ext {'decreases' if mean_Se_change < 0 else 'increases'} ({mean_Se_change:.1f}). "
          f"2nd law {'HOLDS' if holds else 'VIOLATED'} for S_total.")
    print(f"{'='*70}")

    save_results('phase76_entropy_production', {
        'experiment': 'Entropy Production Rate',
        'summary': {
            'pct_2nd_law': float(overall_2nd),
            'S_internal_change': float(mean_Si_change),
            'S_external_change': float(mean_Se_change),
            '2nd_law_holds': bool(holds),
        }
    })


if __name__ == '__main__':
    main()
