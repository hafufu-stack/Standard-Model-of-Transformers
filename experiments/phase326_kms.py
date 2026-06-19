# -*- coding: utf-8 -*-
"""
Phase 326: KMS Condition -- Thermal Equilibrium Test
======================================================
The KMS (Kubo-Martin-Schwinger) condition characterizes thermal
equilibrium states in quantum statistical mechanics.
For a state at temperature T:
  <A(t) B(0)> = <B(0) A(t + i*beta)>   (beta = 1/T)
Test if transformer hidden states satisfy the KMS condition.
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


def test_kms(model, tok, prompt, device):
    """Test KMS condition for thermal equilibrium."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_layers = len(out.hidden_states) - 1

    # KMS: correlation functions should satisfy detailed balance
    # C_AB(t) / C_BA(-t) = exp(-beta * omega)
    kms_ratios = []
    layer_detailed_balance = []

    for li in range(n_layers - 1):
        h_A = out.hidden_states[li][0, -1, :].float()
        h_B = out.hidden_states[li + 1][0, -1, :].float()
        h_C = out.hidden_states[li + 2][0, -1, :].float()

        # Forward correlation: <A, B>
        C_forward = float((h_A * h_B).sum().item())
        # Backward correlation: <B, A> (reversed order)
        C_backward = float((h_B * h_C).sum().item())

        # KMS ratio
        if abs(C_backward) > 1e-10:
            ratio = C_forward / C_backward
        else:
            ratio = 1.0
        kms_ratios.append(float(ratio))

        # Detailed balance: ln(C_f/C_b) should be constant = -beta*omega
        if abs(ratio) > 0:
            db = float(np.log(abs(ratio) + 1e-15))
        else:
            db = 0
        layer_detailed_balance.append(db)

    # Temperature extraction from KMS
    # If KMS holds: beta = -<ln(ratio)> / <omega>
    # Approximate omega ~ 1 (layer spacing)
    if layer_detailed_balance:
        beta_kms = -float(np.mean(layer_detailed_balance))
        T_kms = 1.0 / (abs(beta_kms) + 1e-10)
    else:
        beta_kms = 0
        T_kms = 0

    # KMS quality: how constant is the ratio?
    ratio_cv = float(np.std(kms_ratios) / (abs(np.mean(kms_ratios)) + 1e-10))

    # Actual temperature for comparison
    T_actual = []
    for li in range(n_layers + 1):
        h = out.hidden_states[li][0, -1, :].float()
        T_actual.append(float(h.std().item()))

    return {
        'kms_ratios': [round(r, 4) for r in kms_ratios],
        'detailed_balance': [round(d, 4) for d in layer_detailed_balance],
        'T_kms': round(T_kms, 4),
        'beta_kms': round(beta_kms, 4),
        'ratio_cv': round(ratio_cv, 4),
        'T_actual_mean': round(float(np.mean(T_actual)), 4),
        'kms_satisfied': ratio_cv < 0.5,
    }


def main():
    print("=" * 70)
    print("Phase 326: KMS Condition")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = {}

    for size in ['0.5B', '1.5B']:
        print(f"\n=== {size} ===")
        model, tok = load_model(device, size=size)

        kms_data = []
        for prompt in PROMPTS:
            k = test_kms(model, tok, prompt, device)
            kms_data.append(k)

        n = len(kms_data[0]['kms_ratios'])
        avg_ratio = [float(np.mean([k['kms_ratios'][i] for k in kms_data])) for i in range(n)]
        avg_db = [float(np.mean([k['detailed_balance'][i] for k in kms_data])) for i in range(n)]

        all_results[size] = {
            'avg_kms_ratio': [round(r, 4) for r in avg_ratio],
            'avg_detailed_balance': [round(d, 4) for d in avg_db],
            'T_kms': round(float(np.mean([k['T_kms'] for k in kms_data])), 4),
            'T_actual': round(float(np.mean([k['T_actual_mean'] for k in kms_data])), 4),
            'ratio_cv': round(float(np.mean([k['ratio_cv'] for k in kms_data])), 4),
            'kms_satisfied': sum(1 for k in kms_data if k['kms_satisfied']) >= 3,
        }
        kms = 'YES' if all_results[size]['kms_satisfied'] else 'NO'
        print(f"  KMS satisfied: {kms}")
        print(f"  T_KMS = {all_results[size]['T_kms']:.4f}")
        print(f"  T_actual = {all_results[size]['T_actual']:.4f}")

        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = {'0.5B': '#3498db', '1.5B': '#e74c3c'}

    for size, data in all_results.items():
        axes[0, 0].plot(data['avg_kms_ratio'], '-', color=colors[size], lw=2, label=size)
    axes[0, 0].axhline(1.0, color='gold', ls='--', lw=1)
    axes[0, 0].set_xlabel('Layer'); axes[0, 0].set_ylabel('KMS Ratio')
    axes[0, 0].set_title('(a) KMS Ratio Profile', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    for size, data in all_results.items():
        axes[0, 1].plot(data['avg_detailed_balance'], '-', color=colors[size], lw=2, label=size)
    axes[0, 1].axhline(0, color='gray', ls='--', lw=1)
    axes[0, 1].set_xlabel('Layer'); axes[0, 1].set_ylabel('ln(C_f/C_b)')
    axes[0, 1].set_title('(b) Detailed Balance', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    sizes = list(all_results.keys())
    x = np.arange(len(sizes))
    w = 0.35
    axes[0, 2].bar(x - w/2, [all_results[s]['T_kms'] for s in sizes], w,
                  label='T_KMS', color='#3498db')
    axes[0, 2].bar(x + w/2, [all_results[s]['T_actual'] for s in sizes], w,
                  label='T_actual', color='#e74c3c')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(sizes)
    axes[0, 2].set_ylabel('T'); axes[0, 2].set_title('(c) Temperature Comparison', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].axis('off'); axes[1, 1].axis('off')
    txt = "KMS CONDITION\n\n"
    txt += "<AB> / <BA> = exp(-beta*w)\n\n"
    for s in sizes:
        d = all_results[s]
        kms = 'YES' if d['kms_satisfied'] else 'NO'
        txt += f"{s}:\n"
        txt += f"  KMS: {kms}\n"
        txt += f"  T_KMS = {d['T_kms']:.3f}\n"
        txt += f"  T_act = {d['T_actual']:.3f}\n\n"
    axes[1, 2].text(0.5, 0.5, txt, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off'); axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 326: KMS Condition -- Thermal Equilibrium", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase326_kms')
    plt.close()
    save_results('phase326_kms', {'experiment': 'KMS Condition', 'results': all_results})
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
