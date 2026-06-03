# -*- coding: utf-8 -*-
"""
Phase 14: Residual Stream Momentum (Opus Original) - v2
=========================================================
Measure semantic momentum WITHOUT hooks (hook-free approach).
Perturb the input embeddings directly, then compare final hidden states.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure


def main():
    print("=" * 70)
    print("Phase 14: Residual Stream Momentum (Hook-Free)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers

    prompts = [
        "The capital of Japan is",
        "Two plus two equals",
        "The color of the sky is",
        "Water boils at one hundred",
        "The opposite of hot is",
    ]

    perturbation_scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Strategy: perturb embedding at specific token positions
    # Measure deflection at each hidden layer
    print("\n--- Embedding perturbation -> layer-wise deflection ---")
    
    all_deflections = {}  # {scale: [layer_deflections averaged across prompts]}
    
    for scale in perturbation_scales:
        layer_deflections_all = []
        
        for prompt in prompts:
            inp = tok(prompt, return_tensors='pt').to(device)
            
            # Get baseline hidden states
            with torch.no_grad():
                base_out = model(**inp, output_hidden_states=True)
            base_hs = [h[0, -1, :].float().cpu() for h in base_out.hidden_states]
            
            # Get embeddings, perturb last token, run through model
            with torch.no_grad():
                embeddings = model.model.embed_tokens(inp['input_ids'])
                # Perturb last token embedding in fp32
                emb_perturbed = embeddings.clone()
                last_emb = emb_perturbed[:, -1:, :].float()
                noise = torch.randn_like(last_emb)
                noise = noise / (noise.norm() + 1e-10) * scale * last_emb.norm()
                emb_perturbed[:, -1:, :] = (last_emb + noise).to(embeddings.dtype)
                
                # Run perturbed embeddings through the model
                # Use model internals to pass modified embeddings
                perturbed_out = model(
                    inputs_embeds=emb_perturbed,
                    attention_mask=inp.get('attention_mask'),
                    output_hidden_states=True,
                )
            
            perturbed_hs = [h[0, -1, :].float().cpu() for h in perturbed_out.hidden_states]
            
            # Compute deflection at each layer
            layer_deflections = []
            for layer_idx in range(len(base_hs)):
                bh = base_hs[layer_idx]
                ph = perturbed_hs[layer_idx]
                nb = bh.norm()
                np_ = ph.norm()
                if nb > 1e-6 and np_ > 1e-6 and not torch.isnan(ph).any():
                    cos = torch.clamp(torch.dot(bh, ph) / (nb * np_), -1.0, 1.0)
                    deflection = 1.0 - cos.item()
                else:
                    deflection = 1.0
                layer_deflections.append(deflection)
            
            layer_deflections_all.append(layer_deflections)
        
        avg_layer_defl = np.mean(layer_deflections_all, axis=0)
        all_deflections[scale] = avg_layer_defl
        
        # Amplification: deflection at last layer / deflection at first layer
        amp = avg_layer_defl[-1] / (avg_layer_defl[1] + 1e-15)
        print(f"  Scale {scale:.3f}: defl[L0]={avg_layer_defl[1]:.6f}, "
              f"defl[L{n_layers}]={avg_layer_defl[-1]:.6f}, "
              f"amplification={amp:.2f}x")

    # Compute "stiffness" at each layer = d(deflection)/d(scale)
    print("\n--- Layer Stiffness (resistance to perturbation) ---")
    stiffness = []
    for layer_idx in range(n_layers + 1):
        vals = [(s, all_deflections[s][layer_idx]) for s in perturbation_scales]
        # Simple linear regression: deflection = k * scale
        scales_arr = np.array([v[0] for v in vals])
        defl_arr = np.array([v[1] for v in vals])
        if np.any(defl_arr > 0):
            # Susceptibility = slope of deflection vs scale
            try:
                k = np.polyfit(scales_arr, defl_arr, 1)[0]
            except Exception:
                k = 0.0
            stiffness.append(1.0 / (k + 1e-10))  # stiffness = 1/susceptibility
        else:
            stiffness.append(float('inf'))
    
    finite_stiffness = [s for s in stiffness if np.isfinite(s) and s > 0]
    avg_stiffness = np.mean(finite_stiffness) if finite_stiffness else 0
    
    for li in range(0, n_layers + 1, 4):
        s = stiffness[li]
        print(f"  Layer {li}: stiffness = {s:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Deflection profile across layers for each scale
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(perturbation_scales)))
    for idx, scale in enumerate(perturbation_scales):
        ax.plot(range(n_layers + 1), all_deflections[scale], '-', 
                color=cmap[idx], label=f's={scale}', lw=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Deflection (1 - cos_sim)')
    ax.set_title('(a) Perturbation Propagation')
    ax.legend(fontsize=7)
    ax.set_yscale('log')

    # (b) Stiffness profile
    ax = axes[1]
    valid_stiff = [(i, s) for i, s in enumerate(stiffness) if np.isfinite(s) and abs(s) < 1e10]
    if valid_stiff:
        layers_s, vals_s = zip(*valid_stiff)
        ax.bar(layers_s, vals_s, color='#e74c3c', alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Stiffness (1/susceptibility)')
    ax.set_title('(b) Semantic Stiffness Profile')

    # (c) Amplification factor
    ax = axes[2]
    amps = []
    for scale in perturbation_scales:
        d = all_deflections[scale]
        amp = d[-1] / (d[1] + 1e-15)
        amps.append(amp)
    ax.plot(perturbation_scales, amps, 'o-', color='#3498db', ms=8)
    ax.set_xlabel('Perturbation Scale')
    ax.set_ylabel('Amplification (last/first)')
    ax.set_title('(c) Perturbation Amplification')
    ax.set_xscale('log')

    fig.suptitle(
        f"Phase 14: Residual Stream Momentum\n"
        f"Avg stiffness = {avg_stiffness:.1f} | "
        f"{'HIGH' if avg_stiffness > 10 else 'LOW'} semantic inertia",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_figure(fig, "phase14_residual_momentum")
    plt.close()

    if avg_stiffness > 10:
        verdict = (f"HIGH SEMANTIC INERTIA: stiffness={avg_stiffness:.1f}. "
                   f"Residual stream strongly resists perturbation.")
    elif avg_stiffness > 1:
        verdict = (f"MODERATE INERTIA: stiffness={avg_stiffness:.1f}. "
                   f"Residual stream is somewhat deflectable.")
    else:
        verdict = (f"LOW INERTIA: stiffness={avg_stiffness:.1f}. "
                   f"Perturbations propagate and amplify through layers.")

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    result = {
        'name': 'Phase 14: Residual Stream Momentum',
        'summary': {'verdict': verdict, 'avg_stiffness': avg_stiffness},
        'deflections': {str(s): all_deflections[s].tolist() for s in perturbation_scales},
    }
    save_results("phase14_residual_momentum", result)
    return result


if __name__ == '__main__':
    main()
