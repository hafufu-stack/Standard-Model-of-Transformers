# -*- coding: utf-8 -*-
"""
Phase 282: 1-DOF Reduction -- Master Variable
================================================
Phase 278 found 6 laws compress to 1 dimension (PC1=99.998%).
Identify which law is the "master variable" by analyzing the PC1 eigenvector.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

PROMPTS = [
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "The most effective approach to solving climate change is",
    "Once upon a time in a kingdom far away",
    "Machine learning models can be used to",
    "The fundamental theorem of calculus states that",
    "In the beginning there was nothing and then",
    "The chemical composition of water is",
    "Artificial intelligence will transform society by",
    "The speed of light in vacuum is approximately",
    "Evolution by natural selection explains how",
    "The structure of DNA was discovered by",
    "Thermodynamics tells us that entropy always",
    "The brain processes information through networks of",
    "Climate change is caused primarily by",
    "The periodic table organizes elements by their",
    "Gravity is the weakest of the four fundamental",
    "Democracy requires participation from citizens who",
    "The human genome contains approximately three billion",
    "Photosynthesis converts sunlight into chemical energy through",
]


def compute_6_laws(model, tok, prompt, device):
    """Compute all 6 universal law values for a prompt."""
    results, out = measure_full_thermodynamics(model, tok, prompt, device)

    # Extract per-layer values
    Us = np.array([r['U'] for r in results])
    Ts = np.array([r['T'] for r in results])
    PRs = np.array([r['PR'] for r in results])
    P1Ts = np.array([r['PRT'] for r in results])

    # Law 1: Boltzmann (R2 of U vs layer)
    layers = np.arange(len(results))
    if len(layers) >= 3:
        _, _, r1, _, _ = stats.linregress(layers, Us)
        law1 = r1**2
    else:
        law1 = 0.0

    # Law 2: Heat Capacity (dT/dU stability)
    dT = np.diff(Ts)
    dU = np.diff(Us)
    Cv = dT / (dU + 1e-10)
    law2 = float(np.mean(Cv))

    # Law 3: Radiation (T monotonic decrease fraction)
    decreasing = sum(1 for i in range(len(Ts)-1) if Ts[i+1] <= Ts[i])
    law3 = decreasing / max(len(Ts)-1, 1)

    # Law 4: Carnot (efficiency bound)
    T_max = max(Ts)
    T_min = min(Ts[len(Ts)//2:])  # min in latter half
    law4 = 1 - T_min / max(T_max, 1e-10)

    # Law 5: Information (free energy ratio)
    F = Us - Ts  # simplified free energy
    law5 = float(F[-1] / (F[0] + 1e-10))

    # Law 6: P1*T conservation
    law6 = float(np.mean(P1Ts))

    return [law1, law2, law3, law4, law5, law6]


def main():
    print("=" * 70)
    print("Phase 282: 1-DOF Reduction -- Master Variable")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        law_matrix = []
        for pi, prompt in enumerate(PROMPTS):
            laws = compute_6_laws(model, tok, prompt, device)
            law_matrix.append(laws)
            if pi % 5 == 0:
                print(f"  Processed {pi+1}/{len(PROMPTS)}")

        law_matrix = np.array(law_matrix)  # (n_prompts, 6)

        # PCA
        pca = PCA(n_components=min(6, len(law_matrix)))
        pca.fit(law_matrix)

        # PC1 eigenvector tells us which laws contribute most
        pc1_loadings = pca.components_[0]
        law_names = ['Boltzmann', 'Cv', 'Radiation', 'Carnot', 'Info', 'P1T']

        # Identify master variable (highest absolute loading)
        master_idx = int(np.argmax(np.abs(pc1_loadings)))
        master_var = law_names[master_idx]

        # Correlation of each law with the PC1 score
        pc1_scores = pca.transform(law_matrix)[:, 0]
        correlations = {}
        for i, name in enumerate(law_names):
            r, p = stats.pearsonr(law_matrix[:, i], pc1_scores)
            correlations[name] = {'r': round(float(r), 4), 'p': round(float(p), 6)}

        # Granger-like causality: does master variable predict others?
        prediction_r2 = {}
        for i, name in enumerate(law_names):
            if i == master_idx:
                continue
            s, _, r, _, _ = stats.linregress(law_matrix[:, master_idx], law_matrix[:, i])
            prediction_r2[name] = round(float(r**2), 4)

        all_results[size] = {
            'explained_variance': [round(float(v), 6) for v in pca.explained_variance_ratio_],
            'pc1_loadings': {law_names[i]: round(float(pc1_loadings[i]), 6) for i in range(6)},
            'master_variable': master_var,
            'master_loading': round(float(np.abs(pc1_loadings[master_idx])), 6),
            'correlations': correlations,
            'prediction_r2': prediction_r2,
            'law_means': {law_names[i]: round(float(np.mean(law_matrix[:, i])), 4) for i in range(6)},
            'law_stds': {law_names[i]: round(float(np.std(law_matrix[:, i])), 4) for i in range(6)},
        }

        print(f"  Master variable: {master_var} (loading={np.abs(pc1_loadings[master_idx]):.4f})")
        print(f"  PC1 loadings: {dict(zip(law_names, [f'{v:.3f}' for v in pc1_loadings]))}")
        print(f"  Explained variance: {pca.explained_variance_ratio_[0]:.6f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors_model = {'0.5B': '#3498db', '1.5B': '#e74c3c'}
    law_names = ['Boltzmann', 'Cv', 'Radiation', 'Carnot', 'Info', 'P1T']

    # (a) PC1 loadings comparison
    x = np.arange(6)
    w = 0.35
    for i, (size, data) in enumerate(all_results.items()):
        loadings = [data['pc1_loadings'][n] for n in law_names]
        axes[0, 0].bar(x + (i-0.5)*w, np.abs(loadings), w,
                      color=colors_model[size], label=size, alpha=0.8)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(law_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('|PC1 Loading|')
    axes[0, 0].set_title('(a) PC1 Loadings (Master Variable)', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    # (b) Explained variance
    for size, data in all_results.items():
        ev = data['explained_variance']
        axes[0, 1].bar(range(len(ev)), ev, alpha=0.6, color=colors_model[size], label=size)
    axes[0, 1].set_xlabel('Principal Component')
    axes[0, 1].set_ylabel('Explained Variance Ratio')
    axes[0, 1].set_title('(b) PCA Spectrum', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # (c) Prediction R2 (master -> others)
    for i, (size, data) in enumerate(all_results.items()):
        names = list(data['prediction_r2'].keys())
        r2s = [data['prediction_r2'][n] for n in names]
        axes[0, 2].bar([n[:6] for n in names], r2s, alpha=0.6,
                      color=colors_model[size], label=f"{size} (master={data['master_variable']})")
    axes[0, 2].set_ylabel('R2')
    axes[0, 2].set_title('(c) Master -> Other Laws (Prediction R2)', fontweight='bold')
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(alpha=0.3)

    # (d) Law correlation heatmap (first model)
    first_size = list(all_results.keys())[0]
    corrs = all_results[first_size]['correlations']
    axes[1, 0].barh(law_names, [corrs[n]['r'] for n in law_names],
                   color=['#e74c3c' if abs(corrs[n]['r']) > 0.5 else '#3498db' for n in law_names])
    axes[1, 0].set_xlabel('Correlation with PC1')
    axes[1, 0].set_title(f'(d) Law-PC1 Correlations ({first_size})', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # (e) Law means
    for i, (size, data) in enumerate(all_results.items()):
        means = [data['law_means'][n] for n in law_names]
        axes[1, 1].bar(x + (i-0.5)*w, means, w, color=colors_model[size], label=size)
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(law_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].set_title('(e) Law Mean Values', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    txt = "1-DOF REDUCTION: MASTER VARIABLE\n\n"
    for size, data in all_results.items():
        txt += f"{size}:\n"
        txt += f"  Master: {data['master_variable']}\n"
        txt += f"  Loading: {data['master_loading']:.4f}\n"
        txt += f"  PC1 explains: {data['explained_variance'][0]*100:.2f}%\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 282: 1-DOF Reduction -- Master Variable",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase282_master_variable')
    plt.close()

    save_results('phase282_master_variable', {
        'experiment': '1-DOF Reduction - Master Variable',
        'results': all_results,
    })

    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
