# -*- coding: utf-8 -*-
"""
Phase 258: Maxwell's Demon — Semantic vs Complement Entropy
=============================================================
SQ-Q304 discovered: The Transformer acts as Maxwell's Demon.
- Total entropy INCREASES (2nd law obeyed)
- Semantic subspace entropy DECREASES (order created)
- Complement subspace entropy INCREASES (disorder dumped)

This phase replicates Q304's Maxwell's Demon analysis using
SM thermodynamic measurements, then bridges:
  Does the demon score correlate with SM temperature?
  Is the demon stronger at high or low SM-T layers?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
    "The speed of light is constant in all frames",
    "DNA encodes genetic information using four bases",
    "Stars form from collapsing molecular clouds",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
    "Colorless green ideas sleep furiously in",
]


def fit_boltzmann_T(h_np):
    """SQ Hawking temperature."""
    energies = np.sort(h_np ** 2)[::-1]
    probs = energies / (np.sum(energies) + 1e-10)
    ranks = np.arange(1, min(len(probs), 200) + 1).astype(float)
    log_probs = np.log(probs[:len(ranks)] + 1e-15)
    valid = np.isfinite(log_probs)
    if np.sum(valid) < 5:
        return 0.0
    try:
        slope, _ = np.polyfit(ranks[valid], log_probs[valid], 1)
        return float(-1.0 / (slope + 1e-15))
    except Exception:
        return 0.0


def maxwell_demon_analysis(model, tok, device, model_name):
    """Measure Maxwell's Demon effect at every layer."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    D = model.config.hidden_size
    K = 64  # semantic subspace size

    all_total_S = []
    all_semantic_S = []
    all_complement_S = []
    all_T_sm = []
    all_T_sq = []
    all_concentration = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        total_S, sem_S, comp_S, T_sm_l, T_sq_l, conc_l = [], [], [], [], [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            h_np = h.cpu().numpy()

            # Total Shannon entropy (from |h|)
            h_abs = np.abs(h_np)
            h_prob = h_abs / (np.sum(h_abs) + 1e-10)
            total_entropy = float(-np.sum(h_prob[h_prob > 1e-15] * np.log2(h_prob[h_prob > 1e-15] + 1e-15)))

            # Semantic subspace: top-K most active dimensions
            top_k_idx = np.argsort(np.abs(h_np))[-K:]
            complement_idx = np.argsort(np.abs(h_np))[:-K]

            h_sem = h_abs[top_k_idx]
            h_sem_prob = h_sem / (np.sum(h_sem) + 1e-10)
            semantic_entropy = float(-np.sum(h_sem_prob[h_sem_prob > 1e-15] * np.log2(h_sem_prob[h_sem_prob > 1e-15] + 1e-15)))

            h_comp = h_abs[complement_idx]
            h_comp_prob = h_comp / (np.sum(h_comp) + 1e-10)
            complement_entropy = float(-np.sum(h_comp_prob[h_comp_prob > 1e-15] * np.log2(h_comp_prob[h_comp_prob > 1e-15] + 1e-15)))

            # Concentration ratio: energy in top-K
            concentration = float(np.sum(h_abs[top_k_idx]**2) / (np.sum(h_abs**2) + 1e-10))

            # SM Temperature
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs_out = torch.softmax(logits, dim=-1)
            S_out = -(probs_out * torch.log(probs_out + 1e-10)).sum().item()
            T_sm = float(S_out) if not np.isnan(S_out) else 0.0

            # SQ Temperature
            T_sq = fit_boltzmann_T(h_np)

            total_S.append(total_entropy)
            sem_S.append(semantic_entropy)
            comp_S.append(complement_entropy)
            T_sm_l.append(T_sm)
            T_sq_l.append(T_sq)
            conc_l.append(concentration)

        all_total_S.append(total_S)
        all_semantic_S.append(sem_S)
        all_complement_S.append(comp_S)
        all_T_sm.append(T_sm_l)
        all_T_sq.append(T_sq_l)
        all_concentration.append(conc_l)

    # Average across prompts
    n = min(len(t) for t in all_total_S)
    avg = lambda d: np.array([float(np.mean([d[p][i] for p in range(len(d))])) for i in range(n)])
    mean_total = avg(all_total_S)
    mean_semantic = avg(all_semantic_S)
    mean_complement = avg(all_complement_S)
    mean_T_sm = avg(all_T_sm)
    mean_T_sq = avg(all_T_sq)
    mean_conc = avg(all_concentration)

    # Delta S from first to last layer
    dS_total = float(mean_total[-1] - mean_total[0])
    dS_semantic = float(mean_semantic[-1] - mean_semantic[0])
    dS_complement = float(mean_complement[-1] - mean_complement[0])

    # Demon score: how much order is created relative to total disorder
    demon_score = float(-dS_semantic / (dS_total + 1e-10)) if dS_total > 0 else 0

    # Layer-by-layer demon rate: d(S_semantic)/dl
    demon_rate = np.gradient(mean_semantic)

    # Correlation: demon rate vs SM temperature
    r_demon_T, _ = stats.pearsonr(demon_rate[1:], mean_T_sm[1:])

    # Concentration change
    dC = float(mean_conc[-1] - mean_conc[0])

    is_demon = dS_total > 0 and dS_semantic < 0
    verdict = ("MAXWELL'S DEMON CONFIRMED" if is_demon else "NO DEMON EFFECT")

    return {
        'model': model_name,
        'n_layers': n_layers,
        'dS_total': round(dS_total, 4),
        'dS_semantic': round(dS_semantic, 4),
        'dS_complement': round(dS_complement, 4),
        'demon_score': round(demon_score, 4),
        'dConcentration': round(dC, 4),
        'r_demon_T': round(float(r_demon_T), 4),
        'is_demon': is_demon,
        'verdict': verdict,
        'profiles': {
            'total_S': mean_total.tolist(),
            'semantic_S': mean_semantic.tolist(),
            'complement_S': mean_complement.tolist(),
            'T_sm': mean_T_sm.tolist(),
            'T_sq': mean_T_sq.tolist(),
            'concentration': mean_conc.tolist(),
            'demon_rate': demon_rate.tolist(),
        },
    }


def main():
    print("=" * 70)
    print("Phase 258: Maxwell's Demon (SM x SQ)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = maxwell_demon_analysis(model, tok, device, size)
        results[size] = r
        print(f"  dS_total = {r['dS_total']:+.4f}")
        print(f"  dS_semantic = {r['dS_semantic']:+.4f}")
        print(f"  dS_complement = {r['dS_complement']:+.4f}")
        print(f"  Demon score = {r['demon_score']:.4f}")
        print(f"  Concentration change = {r['dConcentration']:+.4f}")
        print(f"  r(demon_rate, T_sm) = {r['r_demon_T']:.4f}")
        print(f"  Verdict: {r['verdict']}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, r in results.items():
        n = len(r['profiles']['total_S'])
        x = range(n)
        c = colors[size]

        # (a) Total, Semantic, Complement entropy profiles
        axes[0, 0].plot(x, r['profiles']['total_S'], '-', color=c, lw=2, label=f'Total ({size})')
        axes[0, 0].plot(x, r['profiles']['semantic_S'], '--', color=c, lw=1.5, alpha=0.7)
        axes[0, 0].plot(x, r['profiles']['complement_S'], ':', color=c, lw=1, alpha=0.5)

        # (b) Concentration profile
        axes[0, 1].plot(x, r['profiles']['concentration'], '-o', color=c, lw=2,
                       markersize=3, label=f"{size} (dC={r['dConcentration']:+.3f})")

        # (c) Demon rate vs layer
        axes[0, 2].plot(x, r['profiles']['demon_rate'], '-', color=c, lw=2, label=size)

        # (d) T_sm profile (for context)
        axes[1, 0].plot(x, r['profiles']['T_sm'], '-', color=c, lw=2, label=f'T_sm ({size})')

        # (e) Demon rate vs T_sm
        axes[1, 1].scatter(r['profiles']['T_sm'][1:], r['profiles']['demon_rate'][1:],
                          c=c, s=20, alpha=0.6, label=f"{size} (r={r['r_demon_T']:.3f})")

    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Entropy (bits)')
    axes[0, 0].set_title("(a) Entropy: Total (solid), Semantic (dash), Complement (dot)")
    axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Concentration')
    axes[0, 1].set_title('(b) Energy Concentration (top-64 dims)')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].set_xlabel('Layer'); axes[0, 2].set_ylabel('dS_semantic/dl')
    axes[0, 2].set_title("(c) Demon Rate (negative = ordering)")
    axes[0, 2].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].set_xlabel('Layer'); axes[1, 0].set_ylabel('T_sm (nats)')
    axes[1, 0].set_title('(d) SM Temperature Profile')
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_xlabel('T_sm'); axes[1, 1].set_ylabel('Demon Rate')
    axes[1, 1].set_title('(e) Demon Rate vs SM Temperature')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=0.3)

    # (f) Summary text
    summary = "MAXWELL'S DEMON ANALYSIS\n\n"
    for size, r in results.items():
        summary += f"{size}:\n"
        summary += f"  dS_total = {r['dS_total']:+.3f}\n"
        summary += f"  dS_semantic = {r['dS_semantic']:+.3f}\n"
        summary += f"  Demon = {r['demon_score']:.3f}\n"
        summary += f"  {r['verdict']}\n\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 258: Maxwell's Demon (SM x SQ Integration)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase258_maxwell_demon')
    plt.close()
    save_results('phase258_maxwell_demon', {
        'experiment': "Maxwell's Demon Analysis",
        'results': results,
    })


if __name__ == '__main__':
    main()
