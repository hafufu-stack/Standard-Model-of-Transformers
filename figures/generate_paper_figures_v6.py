# -*- coding: utf-8 -*-
"""Generate publication-quality figures for paper_v6."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper')
os.makedirs(OUTDIR, exist_ok=True)

# ---- Color Palette ----
C_BLUE   = '#2980b9'
C_RED    = '#c0392b'
C_GREEN  = '#27ae60'
C_ORANGE = '#e67e22'
C_PURPLE = '#8e44ad'
C_TEAL   = '#16a085'
C_GRAY   = '#7f8c8d'
C_GOLD   = '#f39c12'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})


def fig20_applications():
    """Fig 20: S33 application results -- Hallucination, OOD, Difficulty, Pruning."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) Hallucination detection AUROC
    ax = fig.add_subplot(gs[0, 0])
    d = json.load(open(os.path.join(RESULTS, 'phase367_hallucination.json')))
    sizes = list(d['results'].keys())
    aurocs = [d['results'][s]['auroc'] for s in sizes]
    bars = ax.bar(sizes, aurocs, color=[C_BLUE, C_RED], alpha=0.85, edgecolor='white', lw=1.5)
    ax.axhline(0.5, color=C_GRAY, ls='--', alpha=0.5, label='Random')
    ax.set_ylabel('AUROC')
    ax.set_title('(a) Hallucination Detection', fontweight='bold')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
               ha='center', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')

    # (b) OOD detection
    ax = fig.add_subplot(gs[0, 1])
    d = json.load(open(os.path.join(RESULTS, 'phase370_ood.json')))
    for si, size in enumerate(d['results']):
        aurocs_dict = d['results'][size]['single_aurocs']
        fnames = list(aurocs_dict.keys())
        auroc_vals = [aurocs_dict[f] for f in fnames]
        colors_bar = [C_RED if a > 0.8 else C_ORANGE if a > 0.65 else C_BLUE for a in auroc_vals]
        if si == 0:
            ax.barh([f'{f} ' for f in fnames], auroc_vals, color=colors_bar, alpha=0.7, height=0.35,
                   label=f'{size}')
        else:
            ax.barh([f' {f}' for f in fnames], auroc_vals, color=colors_bar, alpha=0.9, height=0.35,
                   label=f'{size}')
    ax.axvline(0.5, color=C_GRAY, ls='--', alpha=0.5)
    ax.set_xlabel('AUROC')
    ax.set_title('(b) OOD Detection Features', fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.2, axis='x')

    # (c) Difficulty prediction
    ax = fig.add_subplot(gs[0, 2])
    d = json.load(open(os.path.join(RESULTS, 'phase369_difficulty.json')))
    sizes = list(d['results'].keys())
    r2s = [d['results'][s]['multi_feature_r2'] for s in sizes]
    bars = ax.bar(sizes, r2s, color=[C_GREEN, C_ORANGE], alpha=0.85, edgecolor='white', lw=1.5)
    ax.set_ylabel('R$^2$ (Perplexity Prediction)')
    ax.set_title('(c) Difficulty Prediction', fontweight='bold')
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
               ha='center', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')

    # (d) Layer pruning KL
    ax = fig.add_subplot(gs[1, 0])
    d = json.load(open(os.path.join(RESULTS, 'phase368_pruning.json')))
    for size in d['results']:
        kl = d['results'][size]['layer_kl']
        L0 = d['results'][size]['L0']
        ax.bar(range(len(kl)), kl, alpha=0.6, label=f'{size} (L0={L0})',
              color=C_BLUE if '0.5' in size else C_RED)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('KL Divergence (Pruning Cost)')
    ax.set_title('(d) Thermodynamic Layer Pruning', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (e) Entropy production rate
    ax = fig.add_subplot(gs[1, 1])
    d = json.load(open(os.path.join(RESULTS, 'phase365_entropy_production.json')))
    for size in d['results']:
        rate = d['results'][size]['mean_rate_profile']
        prig = d['results'][size]['prigogine_ratio']
        ax.plot(range(len(rate)), rate, 'o-', markersize=3, lw=1.5,
               label=f'{size} (Prig={prig:.2f})',
               color=C_BLUE if '0.5' in size else C_RED)
    ax.axhline(0, color=C_GRAY, ls='--', alpha=0.5)
    ax.set_xlabel('Layer Transition')
    ax.set_ylabel('dS/dl')
    ax.set_title('(e) Entropy Production Rate', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (f) TQI comparison
    ax = fig.add_subplot(gs[1, 2])
    d = json.load(open(os.path.join(RESULTS, 'phase371_tqi.json')))
    models = list(d['results'].keys())
    components = ['boltzmann_r2', 'carnot_eta', 'mach_stability', 'p1t_score', 'cv_score']
    comp_labels = ['Boltzmann\n$R^2$', 'Carnot\n$\\eta$', 'Mach\nStab.', '$P_1T$\nScore', '$C_v$\nScore']
    x = np.arange(len(comp_labels))
    width = 0.25
    colors_m = [C_BLUE, C_RED, C_GREEN]
    for mi, m in enumerate(models):
        vals = [d['results'][m][c] for c in components]
        ax.bar(x + mi * width - width, vals, width, label=m, color=colors_m[mi], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('(f) Thermodynamic Quality Index', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2, axis='y')

    fig.suptitle('Predictive Applications of the Thermodynamic Framework',
                fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(OUTDIR, 'fig20_applications.png'),
               dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: fig20_applications.png')
    plt.close()


def fig21_nonequilibrium():
    """Fig 21: Non-equilibrium thermodynamics results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Non-Equilibrium Thermodynamics of Transformers',
                fontsize=14, fontweight='bold')

    # (a) FDT correlation
    ax = axes[0, 0]
    d = json.load(open(os.path.join(RESULTS, 'phase361_fdt.json')))
    sizes = list(d['results'].keys())
    fdt_r = [d['results'][s]['fdt_correlation'] for s in sizes]
    fdt_p = [d['results'][s]['fdt_pvalue'] for s in sizes]
    bars = ax.bar(sizes, fdt_r, color=[C_BLUE, C_RED], alpha=0.85)
    for bar, r, p in zip(bars, fdt_r, fdt_p):
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax.text(bar.get_x() + bar.get_width()/2, max(r, 0) + 0.02,
               f'r={r:.3f}\n({stars})', ha='center', fontsize=9)
    ax.set_ylabel('FDT Correlation (r)')
    ax.set_title('(a) Fluctuation-Dissipation Theorem', fontweight='bold')
    ax.axhline(0, color=C_GRAY, ls='--', alpha=0.5)
    ax.grid(alpha=0.2, axis='y')

    # (b) Jarzynski ratio
    ax = axes[0, 1]
    d = json.load(open(os.path.join(RESULTS, 'phase362_jarzynski.json')))
    for size in d['results']:
        jt = d['results'][size]['jarzynski_test']
        sigmas = [j['sigma'] for j in jt]
        ratios = [j['ratio'] for j in jt]
        ax.plot(sigmas, ratios, 'o-', lw=2, label=size,
               color=C_BLUE if '0.5' in size else C_RED)
    ax.axhline(1.0, color=C_GRAY, ls='--', alpha=0.5, label='Exact equality')
    ax.set_xscale('log')
    ax.set_xlabel('Noise $\\sigma$')
    ax.set_ylabel('Jarzynski Ratio')
    ax.set_title('(b) Jarzynski Equality', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # (c) Onsager symmetry
    ax = axes[1, 0]
    d = json.load(open(os.path.join(RESULTS, 'phase364_onsager.json')))
    sizes = list(d['results'].keys())
    sym = [d['results'][s]['symmetry_ratio'] for s in sizes]
    corr = [d['results'][s]['onsager_correlation'] for s in sizes]
    x = np.arange(len(sizes))
    ax.bar(x - 0.15, sym, 0.3, label='Symmetry Ratio', color=C_BLUE, alpha=0.85)
    ax.bar(x + 0.15, corr, 0.3, label='L$_{ij}$-L$_{ji}$ Corr', color=C_RED, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel('Value')
    ax.set_title('(c) Onsager Reciprocal Relations', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(1.0, color=C_GRAY, ls='--', alpha=0.3)
    ax.grid(alpha=0.2, axis='y')

    # (d) Crooks work distribution
    ax = axes[1, 1]
    d = json.load(open(os.path.join(RESULTS, 'phase363_crooks.json')))
    sizes = list(d['results'].keys())
    wf = [d['results'][s]['mean_forward_work'] for s in sizes]
    wr = [d['results'][s]['mean_reverse_work'] for s in sizes]
    ep = [d['results'][s]['entropy_production_mean'] for s in sizes]
    x = np.arange(len(sizes))
    ax.bar(x - 0.2, wf, 0.2, label='$W_F$', color=C_BLUE, alpha=0.85)
    ax.bar(x, wr, 0.2, label='$W_R$', color=C_RED, alpha=0.85)
    ax.bar(x + 0.2, ep, 0.2, label='$\\sigma_{EP}$', color=C_GREEN, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_ylabel('Value')
    ax.set_title('(d) Crooks Fluctuation Theorem', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(0, color=C_GRAY, ls='--', alpha=0.5)
    ax.grid(alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig21_nonequilibrium.png'),
               dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: fig21_nonequilibrium.png')
    plt.close()


def fig22_overview():
    """Fig 22: Conceptual overview -- The Standard Model of Transformers."""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'The Standard Model of Transformers', fontsize=18,
           ha='center', va='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#2c3e50', edgecolor='none'),
           color='white')

    # Six Universal Laws boxes
    laws = [
        ('Boltzmann\nDistribution\n$R^2 = 0.979$', '#3498db'),
        ('Negative\nSpecific Heat\n$C_v < 0$', '#e74c3c'),
        ('Inverse\nRadiation\n$L \\propto T^{-1.44}$', '#2ecc71'),
        ('Carnot\nEfficiency\n$\\eta = 0.813$', '#f39c12'),
        ('Information\nConcentration\n$F \\uparrow 411\\times$', '#9b59b6'),
        ('$P_1 \\times T$\nConservation\n$\\approx 0.84$', '#e67e22'),
    ]

    for i, (txt, color) in enumerate(laws):
        x = 1.0 + i * 2.1
        rect = FancyBboxPatch((x - 0.8, 5.0), 1.8, 1.8,
                             boxstyle='round,pad=0.1', facecolor=color, alpha=0.15,
                             edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x + 0.1, 5.9, txt, fontsize=8, ha='center', va='center',
               fontweight='bold', color=color)

    # Arrow down
    ax.annotate('', xy=(7, 4.7), xytext=(7, 5.0),
               arrowprops=dict(arrowstyle='->', lw=2, color=C_GRAY))

    # Classification
    classifications = [
        ('Thermodynamic\nEngine', '#3498db', 0.5),
        ('Transonic\nFluid', '#e74c3c', 3.5),
        ('Confining\nQFT', '#27ae60', 6.5),
        ('Curved\nManifold', '#f39c12', 9.5),
        ('Holographic\nSystem', '#9b59b6', 12.5),
    ]

    for txt, color, x in classifications:
        rect = FancyBboxPatch((x, 3.2), 2.5, 1.3,
                             boxstyle='round,pad=0.1', facecolor=color, alpha=0.1,
                             edgecolor=color, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 1.25, 3.85, txt, fontsize=9, ha='center', va='center',
               color=color, fontweight='bold')

    # Applications row
    ax.text(7, 2.5, 'Predictive Applications', fontsize=13, ha='center',
           fontweight='bold', color='#2c3e50')

    apps = [
        ('Hallucination\nDetection\nAUROC=0.984', '#c0392b', 1.5),
        ('OOD Detection\nAUROC=1.0\n($U_{final}$)', '#27ae60', 5.0),
        ('Difficulty\nPrediction\n$R^2=0.73$', '#2980b9', 8.5),
        ('Layer Pruning\nGuided by $L_0$', '#f39c12', 12.0),
    ]

    for txt, color, x in apps:
        rect = FancyBboxPatch((x - 0.7, 1.0), 2.8, 1.3,
                             boxstyle='round,pad=0.1', facecolor=color, alpha=0.1,
                             edgecolor=color, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.7, 1.65, txt, fontsize=8, ha='center', va='center',
               color=color, fontweight='bold')

    # Bottom stats
    ax.text(7, 0.3, '375 experiments  |  33 seasons  |  3 architectures  |  30+ universal laws',
           fontsize=11, ha='center', va='center', color=C_GRAY,
           style='italic')

    fig.savefig(os.path.join(OUTDIR, 'fig22_overview.png'),
               dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: fig22_overview.png')
    plt.close()


if __name__ == '__main__':
    fig20_applications()
    fig21_nonequilibrium()
    fig22_overview()
    print("All paper figures generated.")
