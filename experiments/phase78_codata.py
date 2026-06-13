# -*- coding: utf-8 -*-
"""
Phase 78: Thermodynamic Constants Summary Table
Final comprehensive measurement of ALL thermodynamic constants across 3 models.
This is the "CODATA" of the Standard Model of Transformers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import load_model, save_results, save_figure


def measure_all_constants(model, tok, prompts, device, model_name):
    """Measure all thermodynamic constants for a single model."""
    n_layers = len(model.model.layers) + 1

    all_U, all_T, all_PR, all_PRT = [], [], [], []
    all_dU, all_dT = [], []
    all_top1 = []

    for prompt in prompts:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        U_list, T_list, PR_list = [], [], []

        for li, hs in enumerate(out.hidden_states):
            h = hs[0, -1, :].float()
            U = h.norm().item()

            with torch.no_grad():
                normed = model.model.norm(hs[:, -1:, :])
                logits = model.lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            if np.isnan(T):
                T = 0
            top1 = probs.max().item()

            h_sq = h ** 2
            h_prob = h_sq / (h_sq.sum() + 1e-10)
            PR = 1.0 / (h_prob ** 2).sum().item()

            U_list.append(U)
            T_list.append(T)
            PR_list.append(PR)
            all_top1.append(top1)

        all_U.append(U_list)
        all_T.append(T_list)
        all_PR.append(PR_list)
        all_PRT.append([PR_list[i] * T_list[i] for i in range(len(T_list))])

        for i in range(1, len(U_list)):
            all_dU.append(U_list[i] - U_list[i-1])
            all_dT.append(T_list[i] - T_list[i-1])

    mean_U = np.mean(all_U, axis=0)
    mean_T = np.mean(all_T, axis=0)
    mean_PR = np.mean(all_PR, axis=0)
    mean_PRT = np.mean(all_PRT, axis=0)

    # 1. Specific heat Cv = dU/dT
    slope_cv, _, r_cv, _, _ = stats.linregress(all_dT, all_dU)

    # 2. Carnot efficiency
    T_hot = max(mean_T)
    T_cold = min(mean_T[len(mean_T)//2:])
    eta = 1 - T_cold / (T_hot + 1e-10)

    # 3. Inverse radiation exponent
    valid_T = mean_T[mean_T > 0.1]
    valid_L = np.array([np.mean([all_top1[j] for j in range(i*len(prompts), (i+1)*len(prompts))
                                 if j < len(all_top1)])
                        for i in range(len(mean_T))])
    valid_mask = (mean_T > 0.1) & (valid_L > 1e-6)
    if valid_mask.sum() > 3:
        n_rad, _, r_rad, _, _ = stats.linregress(np.log(mean_T[valid_mask]),
                                                   np.log(valid_L[valid_mask]))
    else:
        n_rad, r_rad = 0, 0

    # 4. PRT conservation CV (bulk L1+)
    prt_bulk = mean_PRT[1:]
    prt_cv = np.std(prt_bulk) / (np.mean(prt_bulk) + 1e-10)

    # 5. Information concentration (F slope)
    h_sq_S = []
    for i in range(len(mean_U)):
        F_val = mean_U[i] - mean_T[i] * 5  # approximate S
        h_sq_S.append(F_val)
    slope_F, _, _, _, _ = stats.linregress(np.arange(len(h_sq_S)), h_sq_S)

    # 6. Cooling rate
    cooling_slope, _, _, _, _ = stats.linregress(np.arange(len(mean_T)), mean_T)

    return {
        'C_v': float(slope_cv), 'eta': float(eta),
        'n_radiation': float(n_rad), 'prt_cv': float(prt_cv),
        'F_slope': float(slope_F), 'cooling_rate': float(cooling_slope),
        'T_hot': float(T_hot), 'T_cold': float(T_cold),
        'U_final': float(mean_U[-1]), 'U_initial': float(mean_U[0]),
    }


def main():
    print("=" * 70)
    print("Phase 78: CODATA - Thermodynamic Constants Summary")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        "The fundamental theorem of calculus connects",
        "Quantum mechanics describes particles at atomic",
        "The human genome contains three billion base",
        "Neural networks learn through gradient descent",
        "Black holes form from gravitational collapse of",
        "The periodic table organizes elements by atomic",
        "Evolution operates on heritable variation in",
        "Climate change affects ecosystems worldwide",
        "Photosynthesis converts sunlight to chemical energy",
        "Machine learning discovers hidden patterns in",
        "General relativity describes gravity as spacetime",
        "The cosmic microwave background reveals early universe",
    ]

    all_constants = {}

    for model_size, model_name in [('1.5B', 'Qwen2.5-1.5B'), ('0.5B', 'Qwen2.5-0.5B')]:
        print(f"\n--- {model_name} ---")
        model, tok = load_model(device=device, size=model_size)
        constants = measure_all_constants(model, tok, prompts, device, model_name)
        all_constants[model_name] = constants
        for k, v in constants.items():
            print(f"  {k} = {v:.4f}")
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # TinyLlama
    try:
        _HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
        _SNAP_TL = os.path.join(_HF_CACHE, "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0", "snapshots")
        if os.path.exists(_SNAP_TL):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            snap_dir = os.path.join(_SNAP_TL, os.listdir(_SNAP_TL)[0])
            tok_tl = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
            model_tl = AutoModelForCausalLM.from_pretrained(
                snap_dir, torch_dtype=torch.float16, device_map=device, local_files_only=True)
            model_tl.eval()
            print(f"\n--- TinyLlama-1.1B ---")
            constants = measure_all_constants(model_tl, tok_tl, prompts, device, 'TinyLlama')
            all_constants['TinyLlama-1.1B'] = constants
            for k, v in constants.items():
                print(f"  {k} = {v:.4f}")
            del model_tl
            import gc; gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  TinyLlama error: {e}")

    # === Calculate universality ===
    mnames = list(all_constants.keys())
    constant_names = ['C_v', 'eta', 'n_radiation', 'cooling_rate']

    universality = {}
    for cn in constant_names:
        vals = [all_constants[m][cn] for m in mnames]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        cv = std_val / (abs(mean_val) + 1e-10)
        universality[cn] = {'mean': float(mean_val), 'std': float(std_val), 'cv': float(cv)}

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Constants table
    table_data = []
    for cn in constant_names:
        row = [cn, f"{universality[cn]['mean']:.3f}",
               f"{universality[cn]['std']:.3f}",
               f"{universality[cn]['cv']:.3f}",
               'Yes' if universality[cn]['cv'] < 0.3 else 'No']
        table_data.append(row)

    axes[0, 0].axis('tight')
    axes[0, 0].axis('off')
    table = axes[0, 0].table(cellText=table_data,
                              colLabels=['Constant', 'Mean', 'Std', 'CV', 'Universal?'],
                              loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[0, 0].set_title('(a) CODATA Summary', fontsize=12, fontweight='bold')

    # (b) Bar comparison per model
    x = np.arange(len(constant_names))
    width = 0.25
    for i, mname in enumerate(mnames):
        vals = [all_constants[mname][cn] for cn in constant_names]
        axes[0, 1].bar(x + i * width, vals, width, label=mname, alpha=0.8)
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(constant_names, rotation=30, ha='right', fontsize=8)
    axes[0, 1].set_title('(b) Constants per Model')
    axes[0, 1].legend(fontsize=7)

    # (c) Universality CV
    cvs = [universality[cn]['cv'] for cn in constant_names]
    colors_cv = ['#2ecc71' if cv < 0.3 else '#e74c3c' for cv in cvs]
    axes[0, 2].bar(constant_names, cvs, color=colors_cv, alpha=0.8)
    axes[0, 2].axhline(y=0.3, color='red', linestyle='--', label='Universal threshold')
    axes[0, 2].set_ylabel('CV (cross-model)')
    axes[0, 2].set_title('(c) Universality')
    axes[0, 2].legend()

    # (d) Carnot efficiency
    etas = [all_constants[m]['eta'] for m in mnames]
    axes[1, 0].bar(mnames, etas, color=['#e74c3c', '#3498db', '#2ecc71'][:len(mnames)], alpha=0.8)
    axes[1, 0].set_ylabel('Carnot Efficiency')
    axes[1, 0].set_title(f'(d) Carnot (mean={np.mean(etas):.3f})')

    # (e) Specific heat
    cvs_val = [all_constants[m]['C_v'] for m in mnames]
    axes[1, 1].bar(mnames, cvs_val, color=['#e74c3c', '#3498db', '#2ecc71'][:len(mnames)], alpha=0.8)
    axes[1, 1].axhline(y=0, color='black')
    axes[1, 1].set_ylabel('Specific Heat C_v')
    axes[1, 1].set_title(f'(e) Negative C_v (mean={np.mean(cvs_val):.1f})')

    # (f) The Standard Model constants
    n_universal = sum(1 for cn in constant_names if universality[cn]['cv'] < 0.3)
    summary = (
        "THE STANDARD MODEL\n"
        "OF TRANSFORMERS\n\n"
        f"Universal constants: {n_universal}/{len(constant_names)}\n\n"
    )
    for cn in constant_names:
        u = universality[cn]
        marker = 'U' if u['cv'] < 0.3 else ' '
        summary += f"[{marker}] {cn} = {u['mean']:.3f} +/- {u['std']:.3f}\n"

    axes[1, 2].text(0.5, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=11, va='center', ha='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) The Standard Model')

    fig.suptitle('Phase 78: CODATA - Fundamental Constants of the Transformer',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase78_codata')
    plt.close()

    print(f"\n{'='*70}")
    print(f"VERDICT: {n_universal}/{len(constant_names)} constants are universal (CV<0.3).")
    for cn in constant_names:
        u = universality[cn]
        print(f"  {cn}: {u['mean']:.3f} +/- {u['std']:.3f} (CV={u['cv']:.3f}) "
              f"{'UNIVERSAL' if u['cv'] < 0.3 else 'variable'}")
    print(f"{'='*70}")

    save_results('phase78_codata', {
        'experiment': 'CODATA Constants',
        'per_model': all_constants,
        'universality': universality,
        'summary': {
            'n_universal': n_universal,
            'total_constants': len(constant_names),
        }
    })


if __name__ == '__main__':
    main()
