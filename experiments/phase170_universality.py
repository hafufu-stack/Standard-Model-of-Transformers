# -*- coding: utf-8 -*-
"""
Phase 170: Thermodynamic Universality Test
The ULTIMATE test: does EVERY prompt follow the same sigmoid?
Test 50 diverse prompts and measure how well eta collapses
onto a single universal curve when normalized.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_model, save_results, save_figure


def sigmoid(x, L0, k, ymin, ymax):
    return ymin + (ymax - ymin) / (1 + np.exp(-k * (x - L0)))


PROMPTS = [
    # Science
    "The speed of light in vacuum is exactly",
    "DNA stores genetic information using four bases",
    "Black holes emit radiation due to quantum effects",
    "The Higgs boson gives particles their mass",
    "Entropy always increases in closed systems",
    # Math
    "The prime numbers are infinite because",
    "Pi is an irrational number that represents",
    "The Pythagorean theorem states that in a right triangle",
    "Euler's formula connects five fundamental constants",
    "The Fibonacci sequence appears throughout nature",
    # Code
    "def binary_search(arr, target): low, high =",
    "SELECT * FROM users WHERE age >",
    "import tensorflow as tf; model = tf.keras.Sequential",
    "async function fetchData(url) { const response =",
    "git commit -m 'fix: resolve memory leak in",
    # History
    "The Roman Empire fell in the year four hundred",
    "The Renaissance began in Italy during the",
    "World War Two ended when Japan surrendered in",
    "The Industrial Revolution transformed manufacturing",
    "Ancient Egypt built the pyramids using thousands",
    # Philosophy
    "Descartes said I think therefore I am which",
    "The trolley problem asks whether you should",
    "Nietzsche proclaimed that God is dead meaning",
    "Kant argued that moral duty is determined by",
    "Existentialism holds that existence precedes essence",
    # Language
    "The most commonly spoken language in the world",
    "Shakespeare invented many words still used today",
    "Grammar rules vary significantly across different",
    "The Oxford comma is a controversial punctuation",
    "Sign language is a complete natural language",
    # Technology
    "Artificial intelligence is transforming healthcare",
    "Quantum computing uses superposition to process",
    "Blockchain technology ensures transparent and secure",
    "The internet was originally developed for military",
    "Self-driving cars use lidar sensors to navigate",
    # Nature
    "Photosynthesis converts carbon dioxide into oxygen",
    "The deepest point in the ocean is the Mariana",
    "Coral reefs support approximately twenty five percent",
    "Migration patterns of birds follow specific routes",
    "The Amazon rainforest produces twenty percent of",
    # Economics
    "Supply and demand determine the price of goods",
    "Inflation erodes the purchasing power of money",
    "The stock market reflects investor confidence in",
    "Cryptocurrency operates on decentralized networks",
    "GDP measures the total economic output of a",
    # Culture
    "Music activates multiple regions of the brain",
    "The Olympic Games originated in ancient Greece",
    "Architecture reflects the values and technology of",
    "Film is considered the seventh art form because",
    "Cooking traditions vary enormously across different",
]


def main():
    print("=" * 70)
    print("Phase 170: Universality Test (50 prompts)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers) + 1

    all_L0 = []
    all_R2 = []
    all_eta_final = []
    all_S_final = []
    all_eta_profiles = []

    for pi, prompt in enumerate(PROMPTS):
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        T_vals = []
        for li in range(n_layers):
            hs = out.hidden_states[li]
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            S = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_vals.append(S if not np.isnan(S) else 0)

        eta = []
        for li in range(n_layers):
            T_sub = T_vals[:li+1]
            if len(T_sub) >= 4:
                T_hot = max(T_sub)
                T_cold = min(T_sub[len(T_sub)//2:])
                e = 1.0 - T_cold / (T_hot + 1e-10) if T_hot > 0.01 else 0
            else:
                e = 0
            eta.append(e)

        try:
            Ls = np.arange(4, n_layers)
            popt, _ = curve_fit(sigmoid, Ls, eta[4:],
                                p0=[22, 0.5, 0, 0.9], maxfev=10000)
            L0 = popt[0]
            sig_pred = sigmoid(Ls, *popt)
            r2 = 1 - np.sum((np.array(eta[4:]) - sig_pred)**2) / (
                np.sum((np.array(eta[4:]) - np.mean(eta[4:]))**2) + 1e-10)
        except:
            L0 = 22
            r2 = 0

        if 0 < L0 < n_layers * 2:
            all_L0.append(L0)
            all_R2.append(r2)
        all_eta_final.append(eta[-1])
        all_S_final.append(T_vals[-1])
        all_eta_profiles.append(eta)

        if pi % 10 == 0:
            print(f"  [{pi+1}/50] L0={L0:.1f}, R2={r2:.3f}")

    print(f"\n  Valid fits: {len(all_L0)}/{len(PROMPTS)}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) All 50 eta profiles (spaghetti plot)
    for eta in all_eta_profiles:
        axes[0,0].plot(range(n_layers), eta, '-', color='#2980b9', alpha=0.15, linewidth=1)
    # Mean
    mean_eta = np.mean(all_eta_profiles, axis=0)
    axes[0,0].plot(range(n_layers), mean_eta, '-', color='#c0392b', linewidth=3,
                  label='Mean')
    axes[0,0].axvline(x=21.7, color='#f39c12', linewidth=2, linestyle='--')
    axes[0,0].set_xlabel('Layer')
    axes[0,0].set_ylabel('$\\eta$')
    axes[0,0].set_title('(a) 50 Eta Profiles')
    axes[0,0].legend()

    # (b) L0 histogram
    axes[0,1].hist(all_L0, bins=15, color='#8e44ad', alpha=0.7, edgecolor='black')
    mean_L0 = np.mean(all_L0)
    std_L0 = np.std(all_L0)
    axes[0,1].axvline(x=mean_L0, color='#c0392b', linewidth=2,
                      label=f'Mean={mean_L0:.1f}+/-{std_L0:.1f}')
    axes[0,1].set_xlabel('$L_0$')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('(b) L0 Distribution')
    axes[0,1].legend()

    # (c) R2 histogram
    axes[0,2].hist(all_R2, bins=15, color='#27ae60', alpha=0.7, edgecolor='black')
    mean_R2 = np.mean(all_R2)
    axes[0,2].axvline(x=mean_R2, color='#c0392b', linewidth=2,
                      label=f'Mean R2={mean_R2:.3f}')
    axes[0,2].set_xlabel('$R^2$')
    axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('(c) Fit Quality Distribution')
    axes[0,2].legend()

    # (d) L0 vs R2 scatter
    axes[1,0].scatter(all_L0, all_R2, c='#2980b9', s=40, alpha=0.7, edgecolors='black')
    axes[1,0].set_xlabel('$L_0$')
    axes[1,0].set_ylabel('$R^2$')
    axes[1,0].set_title('(d) L0 vs Fit Quality')

    # (e) Final S distribution
    axes[1,1].hist(all_S_final, bins=15, color='#f39c12', alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('$S_{final}$')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('(e) Final Entropy Distribution')

    # (f) Summary
    L0_cv = std_L0 / (mean_L0 + 1e-10)
    pct_good = sum(1 for r in all_R2 if r > 0.9) / len(all_R2) * 100
    summary = (
        f"Universality Test: 50 Prompts\n\n"
        f"L0: {mean_L0:.1f} +/- {std_L0:.1f}\n"
        f"L0 CV: {L0_cv:.3f}\n"
        f"Mean R2: {mean_R2:.3f}\n"
        f"R2 > 0.9: {pct_good:.0f}%\n\n"
        f"UNIVERSALITY:\n"
        f"{'CONFIRMED' if pct_good > 80 and L0_cv < 0.2 else 'PARTIAL'}\n\n"
        f"{pct_good:.0f}% of prompts follow\n"
        f"the sigmoid transition"
    )
    axes[1,2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1,2].axis('off')
    axes[1,2].set_title('(f) Summary')

    fig.suptitle('Phase 170: Universality Test (N=50)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase170_universality')
    plt.close()

    print(f"\n{'='*70}")
    print(f"L0 = {mean_L0:.1f} +/- {std_L0:.1f} (CV={L0_cv:.3f})")
    print(f"Mean R2 = {mean_R2:.3f}")
    print(f"R2 > 0.9: {pct_good:.0f}%")
    print(f"UNIVERSALITY: {'CONFIRMED' if pct_good > 80 and L0_cv < 0.2 else 'PARTIAL'}")
    print(f"{'='*70}")

    save_results('phase170_universality', {
        'experiment': 'Universality Test N=50',
        'summary': {
            'mean_L0': float(mean_L0),
            'std_L0': float(std_L0),
            'L0_cv': float(L0_cv),
            'mean_R2': float(mean_R2),
            'pct_good_fit': float(pct_good),
            'n_prompts': len(PROMPTS),
        }
    })


if __name__ == '__main__':
    main()
