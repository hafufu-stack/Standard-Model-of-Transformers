# -*- coding: utf-8 -*-
"""
Phase 369: Prompt Difficulty Prediction
=========================================
Test whether thermodynamic variables can predict how "difficult"
a prompt is for the model (measured by perplexity).

Method:
1. Compute perplexity for diverse prompts.
2. Extract thermodynamic features from each prompt.
3. Correlate thermodynamic features with perplexity.
4. Build a predictor: can we estimate difficulty from thermodynamics alone?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure, measure_full_thermodynamics

# Diverse prompts spanning difficulty levels
PROMPTS = [
    # Easy (common knowledge)
    "The capital of France is",
    "Water boils at 100 degrees",
    "The sun rises in the",
    "One plus one equals",
    # Medium (requires reasoning)
    "The relationship between mass and energy is described by",
    "In machine learning, overfitting occurs when",
    "The Pythagorean theorem states that in a right triangle",
    "Quantum entanglement refers to the phenomenon where",
    # Hard (rare/technical)
    "The Riemann hypothesis conjectures that all non-trivial zeros of",
    "In category theory, a monad is an endofunctor together with",
    "The Langlands program seeks to connect number theory with",
    "Topological quantum computing uses anyons to perform",
    # Very hard (nonsensical / adversarial)
    "The chrono-synclastic infundibulum enables simultaneous",
    "Quantum decoherence in biological neural networks suggests that",
    "The metacognitive resonance of transformer attention heads indicates",
    "Hyperdimensional computing with sparse distributed representations achieves",
]


def compute_perplexity(model, tok, prompt, device):
    """Compute perplexity of a prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, labels=inp['input_ids'])
    loss = out.loss.item()
    return np.exp(loss)


def extract_features(thermo_list):
    """Extract features from thermodynamic profile."""
    Us = [t['U'] for t in thermo_list]
    Ts = [t['T'] for t in thermo_list]
    PRs = [t['PR'] for t in thermo_list]

    return {
        'T_final': Ts[-1],
        'T_mean': np.mean(Ts),
        'T_max': max(Ts),
        'U_final': Us[-1],
        'U_growth': Us[-1] / (Us[0] + 1e-10),
        'PR_final': PRs[-1],
        'dT_max': max(np.abs(np.diff(Ts))),
        'mach': abs(np.diff(Us)[-1]) / (np.std(np.diff(Us)) + 1e-10),
        'eta': 1 - min(Ts) / (max(Ts) + 1e-10),
    }


def main():
    print("=" * 70)
    print("Phase 369: Prompt Difficulty Prediction")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== Qwen2.5-{size} ===")
        model, tok = load_model(device=device, size=size)

        perplexities = []
        all_features = []

        for prompt in PROMPTS:
            ppl = compute_perplexity(model, tok, prompt, device)
            thermo, _ = measure_full_thermodynamics(model, tok, prompt, device)
            feats = extract_features(thermo)

            perplexities.append(ppl)
            all_features.append(feats)

        # Correlate each feature with perplexity
        feature_names = list(all_features[0].keys())
        correlations = {}
        for fname in feature_names:
            fvals = [f[fname] for f in all_features]
            r, p = stats.pearsonr(fvals, perplexities)
            correlations[fname] = {'r': float(r), 'p': float(p)}

        # Best predictor
        best_feat = max(correlations, key=lambda k: abs(correlations[k]['r']))
        best_r = correlations[best_feat]['r']

        # Multi-feature prediction (R2)
        X = np.array([[f[n] for n in feature_names] for f in all_features])
        y = np.array(perplexities)
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_multi = 1 - ss_res / (ss_tot + 1e-10)

        results[size] = {
            'correlations': correlations,
            'best_single_predictor': best_feat,
            'best_single_r': float(best_r),
            'multi_feature_r2': float(r2_multi),
            'perplexities': perplexities,
            'prompt_count': len(PROMPTS),
        }

        print(f"  Best predictor: {best_feat} (r={best_r:.3f})")
        print(f"  Multi-feature R2: {r2_multi:.3f}")
        print(f"  Perplexity range: {min(perplexities):.1f} - {max(perplexities):.1f}")

        del model, tok
        torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 369: Prompt Difficulty Prediction", fontweight='bold')

    for si, size in enumerate(['0.5B', '1.5B']):
        ax = axes[si]
        r = results[size]
        corrs = r['correlations']
        names = list(corrs.keys())
        rvals = [abs(corrs[n]['r']) for n in names]
        colors = ['#e74c3c' if corrs[n]['p'] < 0.05 else '#3498db' for n in names]
        ax.barh(names, rvals, color=colors, alpha=0.8)
        ax.set_xlabel('|Correlation with Perplexity|')
        ax.set_title(f'Qwen2.5-{size} (R2={r["multi_feature_r2"]:.3f})', fontweight='bold')
        ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'phase369_difficulty')
    plt.close()

    save_results('phase369_difficulty', {
        'experiment': 'Prompt Difficulty Prediction',
        'results': results,
    })


if __name__ == '__main__':
    main()
