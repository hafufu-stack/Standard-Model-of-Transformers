# -*- coding: utf-8 -*-
"""
Phase 237: Universality Classes
==================================
Classify transformer architectures into universality classes based on
their thermodynamic critical exponents and scaling behavior.
Use ALL locally available models.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from utils import load_any_model, get_model_internals, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
]

# All viable models from cache
MODELS = [
    ('Qwen/Qwen2.5-0.5B',       'Qwen-0.5B'),
    ('Qwen/Qwen2.5-1.5B',       'Qwen-1.5B'),
    ('Qwen/Qwen2.5-3B',         'Qwen-3B'),
    ('meta-llama/Llama-3.2-1B',  'Llama-1B'),
    ('meta-llama/Llama-3.2-3B',  'Llama-3B'),
    ('gpt2',                     'GPT2-S'),
    ('gpt2-medium',              'GPT2-M'),
    ('gpt2-large',               'GPT2-L'),
    ('gpt2-xl',                  'GPT2-XL'),
    ('facebook/opt-1.3b',        'OPT-1.3B'),
    ('tiiuae/falcon-rw-1b',      'Falcon-1B'),
    ('stabilityai/stablelm-2-1_6b', 'StableLM-1.6B'),
]


def extract_features(model, tok, device, model_name):
    """Extract thermodynamic fingerprint for classification."""
    internals = get_model_internals(model)
    norm_layer = internals['norm']
    lm_head = internals['lm_head']
    n_layers = internals['n_layers']

    all_T, all_P1 = [], []
    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        T_l, P1_l = [], []
        for hs in out.hidden_states:
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            P1_l.append(float(probs.max().item()))
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_l.append(float(S) if not np.isnan(S) else 0)
        all_T.append(T_l); all_P1.append(P1_l)

    n = min(len(t) for t in all_T)
    avg_T = [float(np.mean([all_T[p][l] for p in range(len(PROMPTS))])) for l in range(n)]
    avg_P1 = [float(np.mean([all_P1[p][l] for p in range(len(PROMPTS))])) for l in range(n)]

    # Normalize T to [0,1] for comparison
    T_range = max(avg_T) - min(avg_T) if max(avg_T) > min(avg_T) else 1
    T_norm = [(t - min(avg_T)) / T_range for t in avg_T]

    dT = [avg_T[i+1] - avg_T[i] for i in range(n-1)]
    rho_S, _ = stats.spearmanr(range(n), avg_T)
    rho_P1, _ = stats.spearmanr(range(n), avg_P1)

    # Feature vector for classification
    # 1. Arrow direction (rho_S, rho_P1)
    # 2. T range
    # 3. Final T / Initial T ratio
    # 4. P1 final
    # 5. Shape: position of max |dT| (normalized)
    max_dT_pos = np.argmax([abs(x) for x in dT]) / (len(dT)-1) if len(dT) > 1 else 0
    # 6. Curvature: mean d2T
    d2T = [dT[i+1] - dT[i] for i in range(len(dT)-1)]
    mean_d2T = float(np.mean(d2T)) if d2T else 0

    features = [
        float(rho_S),
        float(rho_P1),
        T_range,
        avg_T[-1] / (avg_T[0] + 1e-6) if avg_T[0] != 0 else avg_T[-1],
        avg_P1[-1],
        float(max_dT_pos),
        mean_d2T,
    ]

    return {
        'model': model_name,
        'n_layers': n_layers,
        'features': features,
        'avg_T': avg_T,
        'avg_P1': avg_P1,
        'T_norm': T_norm,
        'rho_S': float(rho_S),
        'rho_P1': float(rho_P1),
        'T_final': avg_T[-1],
        'P1_final': avg_P1[-1],
    }


def main():
    print("=" * 70)
    print("Phase 237: Universality Classes")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    feature_matrix = []
    model_names = []

    for model_id, short_name in MODELS:
        print(f"\n--- {short_name} ({model_id}) ---")
        try:
            model, tok = load_any_model(model_id, device=device)
            r = extract_features(model, tok, device, short_name)
            results[short_name] = r
            feature_matrix.append(r['features'])
            model_names.append(short_name)
            print(f"  {r['n_layers']}L, rho_S={r['rho_S']:.3f}, T_final={r['T_final']:.2f}")
            del model, tok
        except Exception as e:
            print(f"  FAILED: {e}")
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Clustering
    if len(feature_matrix) >= 3:
        X = np.array(feature_matrix)
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        dist_matrix = pdist(X_norm)
        Z = linkage(dist_matrix, method='ward')
        # Cut at 2 and 3 clusters
        labels_2 = fcluster(Z, t=2, criterion='maxclust')
        labels_3 = fcluster(Z, t=3, criterion='maxclust')
    else:
        Z = None
        labels_2 = [1] * len(feature_matrix)
        labels_3 = [1] * len(feature_matrix)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Normalized T profiles
    cmap = plt.cm.Set1
    for i, (name, r) in enumerate(results.items()):
        x = np.linspace(0, 1, len(r['T_norm']))
        axes[0, 0].plot(x, r['T_norm'], '-', color=cmap(i/max(len(results)-1,1)),
                       lw=1.5, label=name)
    axes[0, 0].set_xlabel('Normalized Depth')
    axes[0, 0].set_ylabel('Normalized T')
    axes[0, 0].set_title('(a) Universal T Profile')
    axes[0, 0].legend(fontsize=5, ncol=2)

    # (b) rho_S vs rho_P1 scatter
    for i, name in enumerate(model_names):
        r = results[name]
        c = cmap(labels_3[i] / 3.0) if len(model_names) >= 3 else 'steelblue'
        axes[0, 1].scatter(r['rho_S'], r['rho_P1'], color=c, s=80, zorder=5)
        axes[0, 1].annotate(name, (r['rho_S'], r['rho_P1']), fontsize=6,
                           ha='center', va='bottom')
    axes[0, 1].axhline(y=0, color='gray', ls='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='gray', ls='-', alpha=0.3)
    axes[0, 1].set_xlabel('rho(S)'); axes[0, 1].set_ylabel('rho(P1)')
    axes[0, 1].set_title('(b) Arrow Space')

    # (c) Dendrogram
    if Z is not None:
        dendrogram(Z, labels=model_names, ax=axes[0, 2], leaf_rotation=90,
                  leaf_font_size=7)
    axes[0, 2].set_title('(c) Universality Dendrogram')

    # (d) P1 profiles
    for i, (name, r) in enumerate(results.items()):
        x = np.linspace(0, 1, len(r['avg_P1']))
        axes[1, 0].plot(x, r['avg_P1'], '-', color=cmap(i/max(len(results)-1,1)),
                       lw=1.5, label=name)
    axes[1, 0].set_xlabel('Normalized Depth')
    axes[1, 0].set_ylabel('P1')
    axes[1, 0].set_title('(d) P1 Profiles')
    axes[1, 0].legend(fontsize=5, ncol=2)

    # (e) Feature heatmap
    if feature_matrix:
        feat_labels = ['rho_S', 'rho_P1', 'T_range', 'T_ratio', 'P1_f', 'dT_pos', 'd2T']
        im = axes[1, 1].imshow(np.array(feature_matrix), aspect='auto', cmap='RdBu_r')
        axes[1, 1].set_yticks(range(len(model_names)))
        axes[1, 1].set_yticklabels(model_names, fontsize=6)
        axes[1, 1].set_xticks(range(len(feat_labels)))
        axes[1, 1].set_xticklabels(feat_labels, fontsize=6, rotation=45)
        axes[1, 1].set_title('(e) Feature Matrix')
        fig.colorbar(im, ax=axes[1, 1], shrink=0.7)

    # (f) Summary
    summary = "UNIVERSALITY CLASSES\n\n"
    for i, name in enumerate(model_names):
        summary += f"{name:>14}: class={labels_3[i]}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 237: Universality Classes of Transformer Architectures",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase237_universality_classes')
    plt.close()
    save_results('phase237_universality_classes', {
        'experiment': 'Universality Classes',
        'results': results,
        'labels_2': [int(x) for x in labels_2],
        'labels_3': [int(x) for x in labels_3],
        'model_names': model_names,
    })


if __name__ == '__main__':
    main()
