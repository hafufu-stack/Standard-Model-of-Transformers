# -*- coding: utf-8 -*-
"""
Phase 337: Tensor Network Structure -- MPS/MERA Decomposition
=====================================================
Modern quantum gravity suggests that spacetime emerges from
tensor networks (MERA). Test whether the Transformer's layer
structure exhibits tensor network properties: bond dimension,
entanglement structure, and isometric constraints.
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
    "The theory of general relativity predicts that",
    "In quantum mechanics the uncertainty principle states",
    "Machine learning models are trained by",
    "The speed of light is constant in all reference frames",
    "Evolution explains the diversity of life on Earth through",
    "The laws of thermodynamics govern all energy transformations",
]


def measure_tensor_network(model, tok, prompt, device):
    """Test tensor network structure."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1
    hiddens = [out.hidden_states[li][0, -1, :].float().cpu() for li in range(n_layers + 1)]
    dim = hiddens[0].shape[0]

    # 1. Bond dimension: effective rank of transfer matrix T_l = h_l (x) h_{l+1}
    bond_dims = []
    for li in range(n_layers):
        h1 = hiddens[li]
        h2 = hiddens[li + 1]
        # Transfer matrix approximation
        T = torch.outer(h1[:min(256, dim)], h2[:min(256, dim)])
        try:
            S = torch.linalg.svdvals(T)
            S_norm = S / (S[0] + 1e-30)
            # Effective bond dimension: number of singular values > 1% of max
            bond_dim = int((S_norm > 0.01).sum().item())
            bond_dims.append(bond_dim)
        except:
            bond_dims.append(0)

    # 2. Isometry test: is T^dagger T ~ I?
    isometry_scores = []
    for li in range(n_layers):
        h1 = hiddens[li][:min(128, dim)]
        h2 = hiddens[li + 1][:min(128, dim)]
        T = torch.outer(h1, h2)
        TdT = T.T @ T
        d = TdT.shape[0]
        # Compare with identity
        I = torch.eye(d)
        if float(torch.norm(TdT).item()) > 1e-10:
            TdT_norm = TdT / (torch.norm(TdT).item() / np.sqrt(d))
            iso_score = float(torch.norm(TdT_norm - I).item() / np.sqrt(d))
        else:
            iso_score = 10.0
        isometry_scores.append(round(iso_score, 4))

    # 3. Entanglement area law: S(A) ~ log(|A|) for MERA, S(A) ~ |A| for MPS
    # Use subsystem entanglement at each layer
    ee_values = []
    for li in range(n_layers + 1):
        h = hiddens[li]
        p = torch.softmax(h, dim=0)
        s = float(-torch.sum(p * torch.log(p + 1e-30)).item())
        ee_values.append(s)

    # Fit: S vs log(layer) for MERA, S vs layer for MPS
    layers = np.arange(1, len(ee_values) + 1)
    log_layers = np.log(layers)

    # MERA fit: S = a * log(l) + b
    if len(layers) > 2:
        slope_mera, _, r_mera, _, _ = stats.linregress(log_layers, ee_values)
        r2_mera = r_mera**2

        # MPS fit: S = a * l + b
        slope_mps, _, r_mps, _, _ = stats.linregress(layers, ee_values)
        r2_mps = r_mps**2
    else:
        r2_mera, r2_mps = 0, 0
        slope_mera, slope_mps = 0, 0

    # 4. Causal cone structure: correlation decay with distance
    corr_decay = []
    for sep in range(1, min(n_layers, 15)):
        corrs = []
        for li in range(n_layers + 1 - sep):
            r = float(torch.nn.functional.cosine_similarity(
                hiddens[li].unsqueeze(0), hiddens[li + sep].unsqueeze(0)).item())
            corrs.append(r)
        corr_decay.append(round(float(np.mean(corrs)), 4))

    # Fit exponential decay: corr ~ exp(-sep / xi)
    seps = np.arange(1, len(corr_decay) + 1)
    if len(seps) > 2 and all(c > 0 for c in corr_decay):
        log_corr = np.log(np.array(corr_decay))
        slope_corr, _, r_corr, _, _ = stats.linregress(seps, log_corr)
        xi = -1.0 / (slope_corr + 1e-10)  # Correlation length
    else:
        xi = 0
        r_corr = 0

    return {
        'bond_dims': bond_dims,
        'avg_bond_dim': round(float(np.mean(bond_dims)), 1),
        'isometry_scores': isometry_scores,
        'avg_isometry': round(float(np.mean(isometry_scores)), 4),
        'r2_mera': round(float(r2_mera), 4),
        'r2_mps': round(float(r2_mps), 4),
        'better_fit': 'MERA' if r2_mera > r2_mps else 'MPS',
        'corr_length': round(float(xi), 2),
        'corr_decay': corr_decay,
        'ee_profile': [round(e, 4) for e in ee_values],
    }


def main():
    print("=" * 70)
    print("Phase 337: Tensor Network Structure")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        tn_data = []
        for prompt in PROMPTS:
            t = measure_tensor_network(model, tok, prompt, device)
            tn_data.append(t)

        n = len(tn_data[0]['bond_dims'])
        n_ee = len(tn_data[0]['ee_profile'])
        n_corr = len(tn_data[0]['corr_decay'])

        mera_count = sum(1 for t in tn_data if t['better_fit'] == 'MERA')
        all_results[size] = {
            'avg_bond_dim': round(float(np.mean([t['avg_bond_dim'] for t in tn_data])), 1),
            'bond_profile': [round(float(np.mean([t['bond_dims'][i] for t in tn_data])), 1) for i in range(n)],
            'avg_isometry': round(float(np.mean([t['avg_isometry'] for t in tn_data])), 4),
            'r2_mera': round(float(np.mean([t['r2_mera'] for t in tn_data])), 4),
            'r2_mps': round(float(np.mean([t['r2_mps'] for t in tn_data])), 4),
            'better_fit': 'MERA' if mera_count >= 4 else 'MPS',
            'corr_length': round(float(np.mean([t['corr_length'] for t in tn_data])), 2),
            'ee_profile': [round(float(np.mean([t['ee_profile'][i] for t in tn_data])), 4) for i in range(n_ee)],
            'corr_decay': [round(float(np.mean([t['corr_decay'][i] for t in tn_data])), 4) for i in range(n_corr)],
        }
        print(f"  Bond dim: {all_results[size]['avg_bond_dim']:.1f}")
        print(f"  MERA R2: {all_results[size]['r2_mera']:.4f}")
        print(f"  MPS R2: {all_results[size]['r2_mps']:.4f}")
        print(f"  Better: {all_results[size]['better_fit']}")
        print(f"  Corr length: {all_results[size]['corr_length']:.2f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['bond_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('Bond dimension')
    axes[0, 0].set_title('(a) Bond Dimension', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['ee_profile'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('Entanglement entropy')
    axes[0, 1].set_title('(b) Entanglement Entropy', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 2].plot(data['corr_decay'], '-o', color=colors[size], lw=2, label=size, markersize=4)
    axes[0, 2].set_xlabel('Separation'); axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].set_title('(c) Correlation Decay', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.3
    axes[1, 0].bar(x - w/2, [all_results[s]['r2_mera'] for s in sizes], w,
                  label='MERA', color='#3498db')
    axes[1, 0].bar(x + w/2, [all_results[s]['r2_mps'] for s in sizes], w,
                  label='MPS', color='#e74c3c')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(sizes)
    axes[1, 0].set_title('(d) MERA vs MPS Fit', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axis('off')
    txt = "TENSOR NETWORK STRUCTURE\n\n"
    for s in sizes:
        d = all_results[s]
        txt += f"{s}:\n"
        txt += f"  bond = {d['avg_bond_dim']:.0f}\n"
        txt += f"  MERA R2 = {d['r2_mera']:.3f}\n"
        txt += f"  MPS R2 = {d['r2_mps']:.3f}\n"
        txt += f"  type: {d['better_fit']}\n"
        txt += f"  xi = {d['corr_length']:.1f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 337: Tensor Network Structure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase337_tensor_network')
    plt.close()
    save_results('phase337_tensor_network', {'experiment': 'Tensor Network', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
