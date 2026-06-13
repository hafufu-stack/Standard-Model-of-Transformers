# -*- coding: utf-8 -*-
"""
Phase 92: Thermodynamic Uncertainty Relation (TUR)
Test whether the TUR sigma^2(J) * sigma^2(F) >= 2kT holds in LLMs.
J = information current (cos similarity change rate between layers)
F = driving force (FFN output norm)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects differentiation and",
    "Quantum mechanics describes particles at the atomic scale",
    "The human genome contains three billion base pairs encoding",
    "Neural networks learn through layers of interconnected nodes",
    "Black holes form from gravitational collapse of massive stars",
    "The periodic table organizes chemical elements by number",
    "Evolution by natural selection operates on heritable variation",
    "Climate change affects ecosystems through rising temperatures",
    "The speed of light in vacuum is approximately three hundred",
    "Artificial intelligence promises to transform every industry",
]


def main():
    print("=" * 70)
    print("Phase 92: Thermodynamic Uncertainty Relation (TUR)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = len(model.model.layers)

    all_J = []  # information currents per prompt
    all_F = []  # driving forces per prompt
    all_kTs = []  # effective temperatures per prompt
    all_TUR_products = []

    for prompt in PROMPTS:
        inp = tok(prompt, return_tensors='pt').to(device)

        # Capture FFN output norms
        ffn_norms = []
        hooks = []

        def make_ffn_hook(storage):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                storage.append(h[0, -1, :].detach().float().norm().item())
            return hook

        for layer in model.model.layers:
            hooks.append(layer.mlp.register_forward_hook(make_ffn_hook(ffn_norms)))

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for h in hooks:
            h.remove()

        # Hidden states
        hs_list = [out.hidden_states[li][0, -1, :].cpu().float() for li in range(len(out.hidden_states))]

        # J = cos similarity change rate (information current)
        J_values = []
        for i in range(1, len(hs_list)):
            cos_sim = torch.nn.functional.cosine_similarity(
                hs_list[i].unsqueeze(0), hs_list[i-1].unsqueeze(0)
            ).item()
            J_values.append(1.0 - cos_sim)  # dissimilarity = information flow

        # F = FFN output norms (driving force)
        F_values = ffn_norms[:len(J_values)]

        # kT = effective temperature from logits at each layer
        kT_values = []
        for li, hs in enumerate(out.hidden_states[1:], 1):
            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0.0
            kT_values.append(T)

        if J_values and F_values:
            J_arr = np.array(J_values)
            F_arr = np.array(F_values[:len(J_values)])
            kT_arr = np.array(kT_values[:len(J_values)])

            var_J = np.var(J_arr)
            var_F = np.var(F_arr)
            mean_kT = np.mean(kT_arr)

            tur_product = var_J * var_F
            tur_bound = 2 * mean_kT

            all_J.append(J_arr)
            all_F.append(F_arr)
            all_kTs.append(mean_kT)
            all_TUR_products.append({
                'prompt': prompt[:40],
                'var_J': float(var_J),
                'var_F': float(var_F),
                'tur_product': float(tur_product),
                'tur_bound': float(tur_bound),
                'tur_satisfied': bool(tur_product >= tur_bound),
                'ratio': float(tur_product / (tur_bound + 1e-10)),
            })

    # === Analysis ===
    n_satisfied = sum(1 for r in all_TUR_products if r['tur_satisfied'])
    ratios = [r['ratio'] for r in all_TUR_products]

    print(f"\n  TUR satisfied: {n_satisfied}/{len(all_TUR_products)}")
    for r in all_TUR_products:
        status = 'PASS' if r['tur_satisfied'] else 'FAIL'
        print(f"    {status}: {r['prompt']}... ratio={r['ratio']:.3f}")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) TUR product vs bound for each prompt
    products = [r['tur_product'] for r in all_TUR_products]
    bounds = [r['tur_bound'] for r in all_TUR_products]
    colors = ['#27ae60' if r['tur_satisfied'] else '#c0392b' for r in all_TUR_products]

    axes[0].scatter(bounds, products, s=100, c=colors, edgecolors='black', zorder=5)
    max_val = max(max(products), max(bounds)) * 1.2
    axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='TUR boundary')
    axes[0].set_xlabel('$2kT$ (TUR bound)')
    axes[0].set_ylabel('$\\sigma^2(J) \\cdot \\sigma^2(F)$ (product)')
    axes[0].set_title(f'(a) TUR Test ({n_satisfied}/{len(all_TUR_products)} pass)')
    axes[0].legend()

    # (b) Ratio histogram
    axes[1].hist(ratios, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=1.0, color='#c0392b', linestyle='--', linewidth=2, label='TUR boundary (1.0)')
    axes[1].set_xlabel('TUR Ratio ($\\sigma^2 J \\cdot \\sigma^2 F / 2kT$)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('(b) TUR Ratio Distribution')
    axes[1].legend()
    mean_ratio = np.mean(ratios)
    axes[1].axvline(x=mean_ratio, color='#27ae60', linestyle=':',
                    label=f'Mean = {mean_ratio:.2f}')
    axes[1].legend()

    # (c) J-F correlation
    if all_J and all_F:
        # Flatten all J and F across prompts
        J_flat = np.concatenate(all_J)
        F_flat = np.concatenate([f[:len(j)] for f, j in zip(all_F, all_J)])
        axes[2].scatter(F_flat, J_flat, s=10, alpha=0.3, color='#8e44ad')
        from scipy import stats as sp_stats
        if len(J_flat) > 2:
            r, p = sp_stats.pearsonr(F_flat, J_flat)
            axes[2].set_title(f'(c) J-F Correlation ($r = {r:.3f}$, $p = {p:.1e}$)')
        axes[2].set_xlabel('Driving Force $F$ (FFN norm)')
        axes[2].set_ylabel('Information Current $J$')

    fig.suptitle(f'Phase 92: Thermodynamic Uncertainty Relation '
                 f'(mean ratio = {mean_ratio:.2f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase92_tur')
    plt.close()

    print(f"\n{'='*70}")
    print(f"TUR satisfied: {n_satisfied}/{len(all_TUR_products)}")
    print(f"Mean TUR ratio: {mean_ratio:.3f}")
    print(f"Verdict: {'TUR HOLDS' if n_satisfied > len(all_TUR_products) * 0.7 else 'TUR VIOLATED'}")
    print(f"{'='*70}")

    save_results('phase92_tur', {
        'experiment': 'Thermodynamic Uncertainty Relation',
        'results': all_TUR_products,
        'summary': {
            'n_satisfied': n_satisfied,
            'n_total': len(all_TUR_products),
            'mean_ratio': float(mean_ratio),
            'tur_holds': bool(n_satisfied > len(all_TUR_products) * 0.7),
        }
    })


if __name__ == '__main__':
    main()
