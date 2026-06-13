# -*- coding: utf-8 -*-
"""
Phase 79: Clausius Inequality and Irreversibility
The Clausius inequality states: dS >= dQ/T for irreversible processes.
Measure irreversibility of each layer transformation.
If process is perfectly reversible, equality holds.
The "gap" measures how much information is lost (irreversibly created).
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
    print("Phase 79: Clausius Inequality & Irreversibility")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    prompts = [
        "The fundamental theorem of calculus connects differentiation and integration",
        "Quantum mechanics describes the behavior of particles at atomic scale",
        "The human genome contains three billion base pairs encoding all genetic",
        "Neural networks learn representations through layers of interconnected",
        "Black holes form when massive stars exhaust their nuclear fuel",
        "The periodic table organizes elements by their atomic number",
        "Evolution by natural selection operates on heritable variation",
        "Climate models simulate atmospheric dynamics using equations",
        "Photosynthesis converts light energy into chemical energy",
        "Machine learning algorithms discover patterns in data",
        "General relativity describes gravity as spacetime curvature",
        "Cryptographic hash functions produce fixed size output",
    ]

    all_results = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_data = []
        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            S = -(h_prob * torch.log(h_prob + 1e-10)).sum().item()

            layer_data.append({'U': U, 'T': T, 'S': S})

        # Compute Clausius quantities per transition
        irrev_gaps = []
        dQ_list = []
        dS_list = []
        for i in range(1, len(layer_data)):
            dU = layer_data[i]['U'] - layer_data[i-1]['U']
            dS = layer_data[i]['S'] - layer_data[i-1]['S']
            T_avg = (layer_data[i]['T'] + layer_data[i-1]['T']) / 2 + 1e-10

            # dQ = TdS (reversible heat)
            dQ = T_avg * dS

            # Clausius: dS >= dQ/T -> gap = dS - dQ/T
            # But dQ/T = dS for reversible, so this is trivially 0
            # Better: irreversibility = dS_universe = dS_system + dS_surroundings
            # dS_surroundings = -dQ/T_surroundings

            # Use the actual work-heat decomposition:
            # dU = dQ + dW -> dQ = dU - dW
            # For neural net, dW = dF (free energy change)
            F_prev = layer_data[i-1]['U'] - layer_data[i-1]['T'] * layer_data[i-1]['S']
            F_curr = layer_data[i]['U'] - layer_data[i]['T'] * layer_data[i]['S']
            dW = F_curr - F_prev
            dQ_actual = dU - dW

            # Irreversibility: sigma_irr = dS - dQ/T
            sigma_irr = dS - dQ_actual / T_avg

            irrev_gaps.append(float(sigma_irr))
            dQ_list.append(float(dQ_actual))
            dS_list.append(float(dS))

        # Cosine similarity between consecutive hidden states (reversibility proxy)
        cos_sims = []
        for i in range(1, len(out.hidden_states)):
            h1 = out.hidden_states[i-1][0, -1, :].float()
            h2 = out.hidden_states[i][0, -1, :].float()
            cos = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            cos_sims.append(cos)

        safe_p = prompt.encode('ascii', errors='replace').decode('ascii')[:40]
        mean_irr = np.mean(irrev_gaps)
        print(f"  '{safe_p}...' mean_irrev={mean_irr:.3f}, mean_cos={np.mean(cos_sims):.3f}")

        all_results.append({
            'prompt': prompt[:60],
            'irrev_gaps': irrev_gaps,
            'dQ': dQ_list, 'dS': dS_list,
            'cos_sims': cos_sims,
            'mean_irrev': float(mean_irr),
        })

    n_trans = len(all_results[0]['irrev_gaps'])

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Irreversibility profile
    mean_irr = np.mean([r['irrev_gaps'] for r in all_results], axis=0)
    std_irr = np.std([r['irrev_gaps'] for r in all_results], axis=0)
    axes[0, 0].plot(range(n_trans), mean_irr, 'o-', color='#e74c3c', linewidth=2, markersize=3)
    axes[0, 0].fill_between(range(n_trans), mean_irr - std_irr, mean_irr + std_irr,
                            alpha=0.2, color='#e74c3c')
    axes[0, 0].axhline(y=0, color='black', linewidth=1)
    axes[0, 0].set_xlabel('Layer Transition')
    axes[0, 0].set_ylabel('Irreversibility (sigma_irr)')
    axes[0, 0].set_title('(a) Irreversibility per Layer')

    # (b) Cosine similarity (reversibility proxy)
    mean_cos = np.mean([r['cos_sims'] for r in all_results], axis=0)
    axes[0, 1].plot(range(len(mean_cos)), mean_cos, 'o-', color='#2ecc71', linewidth=2, markersize=3)
    axes[0, 1].set_xlabel('Layer Transition')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title('(b) Reversibility (cos sim)')

    # (c) dQ vs dS scatter
    all_dQ = [q for r in all_results for q in r['dQ']]
    all_dS = [s for r in all_results for s in r['dS']]
    axes[0, 2].scatter(all_dQ, all_dS, s=10, alpha=0.3, color='#9b59b6')
    r_val, p_val = stats.pearsonr(all_dQ, all_dS)
    axes[0, 2].set_xlabel('dQ (heat)')
    axes[0, 2].set_ylabel('dS (entropy change)')
    axes[0, 2].set_title(f'(c) dQ vs dS (r={r_val:.3f})')

    # (d) Distribution of irreversibility
    all_irr = [g for r in all_results for g in r['irrev_gaps']]
    axes[1, 0].hist(all_irr, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='blue', linewidth=2, label='Reversible')
    axes[1, 0].set_xlabel('sigma_irr')
    axes[1, 0].set_ylabel('Count')
    pct_irrev = sum(1 for g in all_irr if g > 0) / len(all_irr) * 100
    axes[1, 0].set_title(f'(d) {pct_irrev:.0f}% irreversible')
    axes[1, 0].legend()

    # (e) Irreversibility vs cosine similarity
    all_cos = [c for r in all_results for c in r['cos_sims']]
    irr_for_cos = [g for r in all_results for g in r['irrev_gaps']]
    min_len = min(len(all_cos), len(irr_for_cos))
    r_ic, p_ic = stats.pearsonr(all_cos[:min_len], irr_for_cos[:min_len])
    axes[1, 1].scatter(all_cos[:min_len], irr_for_cos[:min_len], s=10, alpha=0.3, color='#f39c12')
    axes[1, 1].set_xlabel('Cosine Similarity')
    axes[1, 1].set_ylabel('Irreversibility')
    axes[1, 1].set_title(f'(e) Irrev vs Reversibility (r={r_ic:.2f})')

    # (f) Per-prompt irreversibility
    mean_irrs = sorted([r['mean_irrev'] for r in all_results])
    axes[1, 2].bar(range(len(mean_irrs)), mean_irrs, color='#3498db', alpha=0.7)
    axes[1, 2].axhline(y=0, color='black')
    axes[1, 2].set_xlabel('Prompt (sorted)')
    axes[1, 2].set_ylabel('Mean Irreversibility')
    axes[1, 2].set_title('(f) Per-Prompt')

    overall_irr = np.mean(all_irr)
    fig.suptitle(f'Phase 79: Clausius Inequality (mean irrev={overall_irr:.3f}, {pct_irrev:.0f}% irreversible)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase79_clausius')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: {pct_irrev:.0f}% transitions irreversible. "
          f"Mean sigma_irr={overall_irr:.3f}. "
          f"dQ-dS correlation r={r_val:.3f}. "
          f"Mean cos_sim={np.mean(all_cos):.3f}.")
    print(f"{'='*70}")

    save_results('phase79_clausius', {
        'experiment': 'Clausius Inequality',
        'summary': {
            'pct_irreversible': float(pct_irrev),
            'mean_irreversibility': float(overall_irr),
            'dQ_dS_correlation': float(r_val),
            'mean_cos_sim': float(np.mean(all_cos)),
        }
    })


if __name__ == '__main__':
    main()
