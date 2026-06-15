# -*- coding: utf-8 -*-
"""
Phase 257: Noether Conservation Cross-Validation
===================================================
SQ discovered: PR * T = 50.1 +/- 14.9 (conserved across layers).
SM discovered: dU = dQ - dW (first law holds, efficiency ~9.7%).

This phase cross-validates:
1. Does PR*T_sm (SM temperature) also give a conserved quantity?
2. Does PR*T_sq (SQ temperature) also hold in SM framework?
3. What is the BEST conserved quantity when combining both frameworks?
4. Test SQ's candidate conserved quantities (norm*std, kurtosis, etc.)
   with SM's output-distribution-based measurements.
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
    "The brain contains billions of neurons",
    "Entropy always increases in closed systems",
    "Purple elephants calculated the square root of",
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


def noether_crossval(model, tok, device, model_name):
    """Test conservation laws from both frameworks."""
    norm_layer = model.model.norm
    lm_head = model.lm_head
    n_layers = len(model.model.layers)
    D = model.config.hidden_size

    # Candidates for conserved quantity
    candidates = {
        'PR_x_Tsm': [],      # SQ's Noether with SM temperature
        'PR_x_Tsq': [],      # SQ's original Noether
        'norm': [],           # Energy
        'norm_x_std': [],     # SQ candidate
        'P1_x_Tsm': [],      # SM order parameter * SM temperature
        'U_x_Tsm': [],       # SM energy * SM temperature
        'kurtosis': [],       # SQ candidate
        'total_var': [],      # SQ candidate (norm^2/D)
    }

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        layer_data = {k: [] for k in candidates}
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            h_np = h.cpu().numpy()

            # SM measurements
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_sm = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T_sm): T_sm = 0
            P1 = float(probs.max().item())

            # SQ measurements
            T_sq = fit_boltzmann_T(h_np)
            h_sq = h_np ** 2
            h_prob = h_sq / (np.sum(h_sq) + 1e-10)
            PR = float(1.0 / (np.sum(h_prob ** 2) + 1e-10))

            U = float(np.linalg.norm(h_np))
            std = float(np.std(h_np))
            mean = float(np.mean(h_np))
            if std > 1e-10:
                kurt = float(np.mean(((h_np - mean) / std) ** 4) - 3)
            else:
                kurt = 0

            layer_data['PR_x_Tsm'].append(PR * T_sm)
            layer_data['PR_x_Tsq'].append(PR * T_sq)
            layer_data['norm'].append(U)
            layer_data['norm_x_std'].append(U * std)
            layer_data['P1_x_Tsm'].append(P1 * T_sm)
            layer_data['U_x_Tsm'].append(U * T_sm)
            layer_data['kurtosis'].append(kurt)
            layer_data['total_var'].append(U**2 / D)

        for k in candidates:
            candidates[k].append(layer_data[k])

    # Compute CV for each candidate
    conservation_scores = {}
    mean_profiles = {}
    for name, all_data in candidates.items():
        n = min(len(d) for d in all_data)
        avg = np.array([float(np.mean([all_data[p][i] for p in range(len(all_data))])) for i in range(n)])
        # Skip embedding layer (index 0), focus on transformer layers
        vals = avg[1:]
        cv = float(np.std(vals) / (np.mean(vals) + 1e-10))
        conservation_scores[name] = round(cv, 4)
        mean_profiles[name] = avg.tolist()

    best = min(conservation_scores, key=conservation_scores.get)
    best_cv = conservation_scores[best]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'conservation_scores': conservation_scores,
        'mean_profiles': mean_profiles,
        'best_conserved': best,
        'best_cv': best_cv,
    }


def main():
    print("=" * 70)
    print("Phase 257: Noether Conservation Cross-Validation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n--- {size} ---")
        model, tok = load_model(device, size=size)
        r = noether_crossval(model, tok, device, size)
        results[size] = r
        print(f"  Best conserved: {r['best_conserved']} (CV={r['best_cv']:.4f})")
        for name, cv in sorted(r['conservation_scores'].items(), key=lambda x: x[1]):
            print(f"    {name:15s}: CV={cv:.4f}")
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    colors_cand = {
        'PR_x_Tsm': '#e74c3c', 'PR_x_Tsq': '#c0392b',
        'norm': '#2ecc71', 'norm_x_std': '#27ae60',
        'P1_x_Tsm': '#3498db', 'U_x_Tsm': '#2980b9',
        'kurtosis': '#f39c12', 'total_var': '#e67e22',
    }

    # Plot each candidate's profile for 1.5B
    r15 = results[list(results.keys())[-1]]
    for idx, (name, profile) in enumerate(r15['mean_profiles'].items()):
        ax = axes[idx // 4, idx % 4]
        cv = r15['conservation_scores'][name]
        color = colors_cand.get(name, 'gray')
        ax.plot(range(len(profile)), profile, '-o', color=color, lw=2, markersize=3)
        is_best = (name == r15['best_conserved'])
        ax.set_title(f'{name}\nCV={cv:.4f}' + (' *BEST*' if is_best else ''),
                    fontweight='bold', fontsize=9,
                    color='red' if is_best else 'black')
        ax.set_xlabel('Layer'); ax.grid(alpha=0.3)

    fig.suptitle("Phase 257: Noether Conservation Cross-Validation\n"
                f"Best: {r15['best_conserved']} (CV={r15['best_cv']:.4f})",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase257_noether_crossval')
    plt.close()
    save_results('phase257_noether_crossval', {
        'experiment': 'Noether Cross-Validation',
        'results': results,
    })


if __name__ == '__main__':
    main()
