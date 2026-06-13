# -*- coding: utf-8 -*-
"""
Phase 138: kT as Hallucination Detector (AUROC)
Phase 135 showed kT is higher for uncertain prompts.
This phase formalizes: can kT predict factual vs uncertain with AUROC?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from utils import load_model, save_results, save_figure

# Label: 0 = factual, 1 = uncertain/hallucination-prone
PROMPTS_LABELS = [
    # Factual (label=0)
    ("The capital of France is", 0),
    ("Water freezes at zero degrees", 0),
    ("The sun is a star located in", 0),
    ("Two plus two equals", 0),
    ("The Earth orbits around the", 0),
    ("DNA stores genetic information in", 0),
    ("The speed of light in a vacuum is approximately", 0),
    ("Photosynthesis occurs in the chloroplasts of", 0),
    ("The chemical formula for water is", 0),
    ("Newton's first law states that an object at rest", 0),
    ("The largest planet in our solar system is", 0),
    ("The human heart pumps blood through", 0),
    # Uncertain (label=1)
    ("The meaning of life is ultimately", 1),
    ("The best programming language for everything is", 1),
    ("In the year 2100 humanity will certainly", 1),
    ("The most important person in all of history is", 1),
    ("Everyone agrees that the ideal government is", 1),
    ("The correct answer to every philosophical question is", 1),
    ("The perfect diet for all humans is", 1),
    ("The objectively best movie ever made is", 1),
    ("The true nature of consciousness is definitely", 1),
    ("The final theory of physics will certainly be", 1),
    ("All experts unanimously agree that happiness comes from", 1),
    ("The one religion that is objectively correct is", 1),
]


def main():
    print("=" * 70)
    print("Phase 138: kT as Hallucination Detector (AUROC)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    labels = []
    kT_finals = []
    S_finals = []
    eta_finals = []
    confidences = []
    combined_scores = []

    for prompt, label in PROMPTS_LABELS:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Final layer quantities
        hs_final = out.hidden_states[-1]
        with torch.no_grad():
            normed = model.model.norm(hs_final[:, -1:, :])
            logits_f = model.lm_head(normed).squeeze().float()
        probs_f = torch.softmax(logits_f, dim=-1)
        S_final = -(probs_f * torch.log(probs_f + 1e-10)).sum().item()

        # Confidence
        conf = probs_f.max().item()

        # kT
        top_k = 50
        top_probs = torch.topk(probs_f, top_k).values
        log_probs = torch.log(top_probs + 1e-10).cpu().numpy()
        ranks = np.arange(1, top_k + 1, dtype=np.float64)
        if np.std(log_probs) > 0.01:
            slope = np.polyfit(ranks, log_probs, 1)[0]
            kT = -1.0 / (slope + 1e-10)
        else:
            kT = 0.1
        kT = max(0.01, min(kT, 50))

        # Eta
        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits_l = model.lm_head(normed).squeeze().float()
            probs_l = torch.softmax(logits_l, dim=-1)
            T = -(probs_l * torch.log(probs_l + 1e-10)).sum().item()
            T_vals.append(T if not np.isnan(T) else 0)

        T_hot = max(T_vals)
        T_cold = min(T_vals[len(T_vals)//2:])
        eta = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0

        labels.append(label)
        kT_finals.append(kT)
        S_finals.append(S_final)
        eta_finals.append(eta)
        confidences.append(conf)

        # Combined score: kT * S / conf
        combined = (kT * S_final) / (conf + 1e-10)
        combined_scores.append(combined)

    labels = np.array(labels)

    # Compute AUROC for each feature
    features = {
        'kT': np.array(kT_finals),
        'S': np.array(S_finals),
        'Confidence': -np.array(confidences),  # negate so higher = more uncertain
        'eta': -np.array(eta_finals),  # negate
        'Combined': np.array(combined_scores),
    }

    aurocs = {}
    for name, feat in features.items():
        try:
            auc = roc_auc_score(labels, feat)
            aurocs[name] = auc
            print(f"  {name}: AUROC={auc:.3f}")
        except:
            aurocs[name] = 0.5

    best_feat = max(aurocs, key=aurocs.get)

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) ROC curves
    for name, feat in features.items():
        fpr, tpr, _ = roc_curve(labels, feat)
        axes[0,0].plot(fpr, tpr, linewidth=2, label=f'{name} ({aurocs[name]:.3f})')
    axes[0,0].plot([0,1], [0,1], 'k--', alpha=0.5)
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('(a) ROC Curves')
    axes[0,0].legend(fontsize=8)

    # (b) kT distribution
    kT_f = [kT_finals[i] for i in range(len(labels)) if labels[i] == 0]
    kT_u = [kT_finals[i] for i in range(len(labels)) if labels[i] == 1]
    axes[0,1].hist(kT_f, bins=8, alpha=0.6, color='#27ae60', label='Factual', edgecolor='black')
    axes[0,1].hist(kT_u, bins=8, alpha=0.6, color='#c0392b', label='Uncertain', edgecolor='black')
    axes[0,1].set_xlabel('$kT_{final}$')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title(f'(b) kT Distribution (AUROC={aurocs["kT"]:.3f})')
    axes[0,1].legend()

    # (c) S distribution
    S_f = [S_finals[i] for i in range(len(labels)) if labels[i] == 0]
    S_u = [S_finals[i] for i in range(len(labels)) if labels[i] == 1]
    axes[0,2].hist(S_f, bins=8, alpha=0.6, color='#27ae60', label='Factual', edgecolor='black')
    axes[0,2].hist(S_u, bins=8, alpha=0.6, color='#c0392b', label='Uncertain', edgecolor='black')
    axes[0,2].set_xlabel('$S_{final}$')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title(f'(c) Entropy Distribution (AUROC={aurocs["S"]:.3f})')
    axes[0,2].legend()

    # (d) Scatter kT vs S
    for i in range(len(labels)):
        c = '#27ae60' if labels[i] == 0 else '#c0392b'
        m = 'o' if labels[i] == 0 else 's'
        axes[1,0].scatter(kT_finals[i], S_finals[i], c=c, marker=m, s=80, edgecolors='black')
    axes[1,0].set_xlabel('$kT$')
    axes[1,0].set_ylabel('$S$')
    axes[1,0].set_title('(d) Phase Space (green=factual)')

    # (e) AUROC bar chart
    names = list(aurocs.keys())
    vals = [aurocs[n] for n in names]
    bar_c = ['#27ae60' if v > 0.7 else '#f39c12' if v > 0.6 else '#c0392b' for v in vals]
    axes[1,1].bar(range(len(names)), vals, color=bar_c, alpha=0.8, edgecolor='black')
    axes[1,1].set_xticks(range(len(names)))
    axes[1,1].set_xticklabels(names, fontsize=9)
    axes[1,1].axhline(y=0.5, color='gray', linestyle='--', label='Random')
    axes[1,1].set_ylabel('AUROC')
    axes[1,1].set_title('(e) Feature Comparison')
    axes[1,1].legend()

    # (f) Summary
    summary = (
        f"kT as Hallucination Detector\n\n"
        + "\n".join(f"{n}: AUROC={aurocs[n]:.3f}" for n in names)
        + f"\n\nBest: {best_feat} ({aurocs[best_feat]:.3f})\n\n"
        f"kT mean: F={np.mean(kT_f):.1f}, U={np.mean(kT_u):.1f}\n"
        f"S mean: F={np.mean(S_f):.1f}, U={np.mean(S_u):.1f}\n"
        f"Conf mean: F={np.mean([confidences[i] for i in range(len(labels)) if labels[i]==0]):.3f}, "
        f"U={np.mean([confidences[i] for i in range(len(labels)) if labels[i]==1]):.3f}"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 138: Thermodynamic Hallucination Detector',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase138_auroc')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Best detector: {best_feat} (AUROC={aurocs[best_feat]:.3f})")
    print(f"{'='*70}")

    save_results('phase138_auroc', {
        'experiment': 'Thermodynamic Hallucination Detector',
        'aurocs': aurocs,
        'summary': {
            'best_feature': best_feat,
            'best_auroc': float(aurocs[best_feat]),
            'aurocs': {k: float(v) for k, v in aurocs.items()},
        }
    })


if __name__ == '__main__':
    main()
