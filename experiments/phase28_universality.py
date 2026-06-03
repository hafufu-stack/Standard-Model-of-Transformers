# -*- coding: utf-8 -*-
"""
Phase 28: Cross-Model Universality (0.5B vs 1.5B)
====================================================
Are the discovered constants universal across model scales?
Measure dU/dT, dark energy fraction, Lyapunov lambda, virial ratio on 0.5B.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, measure_full_thermodynamics, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 28: Cross-Model Universality")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompts = [
        "The fundamental laws of physics govern all matter",
        "Neural networks learn representations from data",
        "Stars form from collapsing clouds of gas and dust",
        "The gradient descent algorithm minimizes loss functions",
    ]

    model_results = {}
    for size in ['0.5B', '1.5B']:
        print(f"\n--- Model: Qwen2.5-{size} ---")
        model, tok = load_model(device=device, size=size)
        n_layers = model.config.num_hidden_layers
        d_model = model.config.hidden_size

        all_thermo = []
        for p in prompts:
            thermo, _ = measure_full_thermodynamics(model, tok, p, device)
            all_thermo.append(thermo)

        # dU/dT
        all_U = np.mean([[d['U'] for d in t] for t in all_thermo], axis=0)
        all_T = np.mean([[d['T'] for d in t] for t in all_thermo], axis=0)
        try:
            valid = np.isfinite(all_T[1:]) & np.isfinite(all_U[1:])
            dUdT = np.polyfit(all_T[1:][valid], all_U[1:][valid], 1)[0]
        except Exception:
            dUdT = 0

        # Dark energy fraction
        attn_forces = []
        ffn_forces = []
        attn_store = [[] for _ in range(n_layers)]
        ffn_store = [[] for _ in range(n_layers)]

        def make_hook(store):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    store.append(output[0][0, -1, :].float().detach().cpu().norm().item())
                else:
                    store.append(output[0, -1, :].float().detach().cpu().norm().item())
            return hook

        handles = []
        for li in range(n_layers):
            h1 = model.model.layers[li].self_attn.register_forward_hook(make_hook(attn_store[li]))
            h2 = model.model.layers[li].mlp.register_forward_hook(make_hook(ffn_store[li]))
            handles.extend([h1, h2])

        inp = tok(prompts[0], return_tensors='pt').to(device)
        with torch.no_grad():
            model(**inp)

        for h in handles:
            h.remove()

        attn_avg = np.mean([s[0] if s else 0 for s in attn_store])
        ffn_avg = np.mean([s[0] if s else 0 for s in ffn_store])
        dark_frac = ffn_avg / (attn_avg + ffn_avg + 1e-10)

        # Lyapunov (simplified)
        sigma = 0.001
        inp = tok(prompts[0], return_tensors='pt').to(device)
        with torch.no_grad():
            base_out = model(**inp, output_hidden_states=True)
            emb = model.model.embed_tokens(inp['input_ids']).float()
            noise = torch.randn_like(emb) * sigma
            emb_pert = (emb + noise).to(next(model.model.embed_tokens.parameters()).dtype)
            pert_out = model(inputs_embeds=emb_pert, output_hidden_states=True)

        divs = []
        for l in range(len(base_out.hidden_states)):
            b = base_out.hidden_states[l][0, -1, :].float().cpu()
            p = pert_out.hidden_states[l][0, -1, :].float().cpu()
            d = (p - b).norm().item() / (b.norm().item() + 1e-10)
            divs.append(max(d, 1e-15))

        try:
            log_div = np.log(np.array(divs) + 1e-15)
            valid = np.isfinite(log_div)
            lyap = np.polyfit(np.arange(len(log_div))[valid], log_div[valid], 1)[0] if valid.sum() > 3 else 0
        except Exception:
            lyap = 0

        avg_PRT = np.mean([t[-1]['PRT'] for t in all_thermo])

        model_results[size] = {
            'dUdT': dUdT, 'dark_frac': dark_frac, 'lyapunov': lyap,
            'avg_PRT': avg_PRT, 'n_layers': n_layers, 'd_model': d_model,
            'U_profile': all_U.tolist(), 'T_profile': all_T.tolist(),
        }
        print(f"  dU/dT = {dUdT:.2f}")
        print(f"  Dark energy = {dark_frac:.3f}")
        print(f"  Lyapunov = {lyap:.4f}")
        print(f"  PR*T = {avg_PRT:.1f}")

        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sizes = list(model_results.keys())

    ax = axes[0]
    constants = ['dUdT', 'dark_frac', 'lyapunov']
    labels = ['dU/dT', 'Dark Energy %', 'Lyapunov lambda']
    for i, (const, lab) in enumerate(zip(constants, labels)):
        vals = [model_results[s][const] for s in sizes]
        ax2 = axes[i]
        bars = ax2.bar(sizes, vals, color=['#3498db', '#e74c3c'], alpha=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        ax2.set_ylabel(lab)
        ax2.set_title(f'({chr(97+i)}) {lab}')
        ax2.set_xlabel('Model Size')

    fig.suptitle(
        "Phase 28: Cross-Model Universality\n"
        f"0.5B: dU/dT={model_results['0.5B']['dUdT']:.1f}, "
        f"1.5B: dU/dT={model_results['1.5B']['dUdT']:.1f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase28_universality")
    plt.close()

    # Check universality
    dUdT_ratio = model_results['0.5B']['dUdT'] / (model_results['1.5B']['dUdT'] + 1e-10)
    verdict = (f"dU/dT ratio (0.5B/1.5B) = {dUdT_ratio:.2f}. "
               f"Dark energy: {model_results['0.5B']['dark_frac']:.2f} vs {model_results['1.5B']['dark_frac']:.2f}. "
               f"Lyapunov: {model_results['0.5B']['lyapunov']:.4f} vs {model_results['1.5B']['lyapunov']:.4f}.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    save_results("phase28_universality", {
        'name': 'Phase 28: Cross-Model Universality',
        'summary': {'verdict': verdict, 'models': model_results},
    })


if __name__ == '__main__':
    main()
