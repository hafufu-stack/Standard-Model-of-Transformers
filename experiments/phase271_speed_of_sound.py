# -*- coding: utf-8 -*-
"""
Phase 271: Speed of Sound in Token Flow
==========================================
If P1*T = const defines an "equation of state" for the token fluid,
then acoustic waves (perturbations) should propagate through it at a
measurable speed.

Inject impulse noise at position 0, measure how it propagates through
attention to affect later positions, quantifying the "speed of sound."
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

PROMPTS = [
    "The fundamental theorem of calculus connects differentiation and integration through a beautiful mathematical relationship that has been studied for centuries by mathematicians around the world",
    "In the beginning there was nothing but darkness and silence until the first stars formed from clouds of hydrogen gas collapsing under their own gravity in the early universe",
    "Machine learning algorithms learn patterns from data by iteratively adjusting their parameters to minimize a loss function that measures the difference between predictions and truth",
]

NOISE_SIGMAS = [0.01, 0.05, 0.1, 0.5, 1.0]


def measure_perturbation_propagation(model, tok, prompt, device, sigma=0.1):
    """Inject noise at position 0 and measure its effect at each position."""
    inp = tok(prompt, return_tensors='pt').to(device)
    seq_len = inp['input_ids'].shape[1]
    norm_layer = model.model.norm
    lm_head = model.lm_head

    # Baseline: clean forward pass
    with torch.no_grad():
        out_clean = model(**inp, output_hidden_states=True)
    clean_hs = [h.clone() for h in out_clean.hidden_states]

    # Get baseline logits at each position (final layer)
    clean_logits_all = []
    hs_final = clean_hs[-1]
    for pos in range(seq_len):
        with torch.no_grad():
            normed = norm_layer(hs_final[:, pos:pos+1, :])
            logits = lm_head(normed).squeeze().float()
        clean_logits_all.append(logits.cpu())

    # Perturbed: inject noise at embedding layer (position 0 only)
    # We use a hook on the embedding output
    perturbation_applied = [False]

    def embed_hook(module, input, output):
        if not perturbation_applied[0]:
            perturbation_applied[0] = True
            noise = torch.randn(1, 1, output.shape[-1], device=output.device, dtype=output.dtype) * sigma
            # Only perturb position 0
            output_mod = output.clone()
            output_mod[:, 0:1, :] += noise
            return output_mod
        return output

    # Hook on the embedding layer
    embed_layer = model.model.embed_tokens
    handle = embed_layer.register_forward_hook(embed_hook)

    with torch.no_grad():
        out_perturbed = model(**inp, output_hidden_states=True)
    handle.remove()

    perturbed_hs = [h.clone() for h in out_perturbed.hidden_states]

    # Measure perturbation at each layer and position
    n_layers = len(clean_hs)
    delta_map = np.zeros((n_layers, seq_len))  # |perturbed - clean| normalized

    for li in range(n_layers):
        for pos in range(seq_len):
            clean_vec = clean_hs[li][0, pos, :].float()
            pert_vec = perturbed_hs[li][0, pos, :].float()
            delta = (pert_vec - clean_vec).norm().item()
            baseline_norm = clean_vec.norm().item() + 1e-10
            delta_map[li, pos] = delta / baseline_norm

    # Measure KL divergence of logits at each position (final layer)
    kl_per_pos = []
    hs_pert_final = perturbed_hs[-1]
    for pos in range(seq_len):
        with torch.no_grad():
            normed = norm_layer(hs_pert_final[:, pos:pos+1, :])
            pert_logits = lm_head(normed).squeeze().float()
        clean_p = torch.softmax(clean_logits_all[pos].to(device), dim=-1)
        pert_p = torch.softmax(pert_logits, dim=-1)
        kl = (clean_p * (torch.log(clean_p + 1e-10) - torch.log(pert_p + 1e-10))).sum().item()
        kl_per_pos.append(max(kl, 0))

    # "Speed of sound": how many positions per layer does the perturbation spread?
    # Find the position at which delta drops below threshold at each layer
    threshold = 0.01  # 1% relative change
    wavefront_per_layer = []
    for li in range(n_layers):
        # Find furthest position with delta > threshold
        furthest = 0
        for pos in range(seq_len):
            if delta_map[li, pos] > threshold:
                furthest = pos
        wavefront_per_layer.append(furthest)

    # Speed = positions_reached / layers_traversed
    if n_layers > 1:
        speeds = []
        for li in range(1, n_layers):
            if li > 0:
                speed = wavefront_per_layer[li] / li
                speeds.append(speed)
        mean_speed = float(np.mean(speeds)) if speeds else 0
    else:
        mean_speed = 0

    return {
        'seq_len': seq_len,
        'n_layers': n_layers,
        'sigma': sigma,
        'delta_map': delta_map,
        'kl_per_pos': kl_per_pos,
        'wavefront_per_layer': wavefront_per_layer,
        'mean_speed': round(mean_speed, 4),
        'max_reach': wavefront_per_layer[-1] if wavefront_per_layer else 0,
    }


def main():
    print("=" * 70)
    print("Phase 271: Speed of Sound in Token Flow")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    all_results = {}

    # Test with different noise levels
    for sigma in NOISE_SIGMAS:
        print(f"\n--- Sigma = {sigma} ---")
        sigma_results = []
        for pi, prompt in enumerate(PROMPTS):
            r = measure_perturbation_propagation(model, tok, prompt, device, sigma)
            sigma_results.append(r)
            print(f"  Prompt {pi}: speed={r['mean_speed']:.3f} pos/layer, "
                  f"reach={r['max_reach']}/{r['seq_len']}")
        all_results[str(sigma)] = sigma_results

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Delta heatmap (sigma=0.1, first prompt)
    r0 = all_results['0.1'][0]
    im = axes[0, 0].imshow(r0['delta_map'], aspect='auto', cmap='hot',
                            interpolation='nearest')
    axes[0, 0].set_xlabel('Token Position')
    axes[0, 0].set_ylabel('Layer')
    axes[0, 0].set_title('(a) Perturbation Propagation (sigma=0.1)', fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0], label='Relative Delta')

    # (b) Wavefront per layer
    for sigma_str, results in all_results.items():
        wf = results[0]['wavefront_per_layer']
        axes[0, 1].plot(range(len(wf)), wf, '-o', markersize=3,
                       label=f'sigma={sigma_str}')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Wavefront Position')
    axes[0, 1].set_title('(b) Wavefront Propagation', fontweight='bold')
    axes[0, 1].legend(fontsize=7); axes[0, 1].grid(alpha=0.3)

    # (c) KL divergence per position (final layer)
    for sigma_str in ['0.01', '0.1', '1.0']:
        if sigma_str in all_results:
            kl = all_results[sigma_str][0]['kl_per_pos']
            axes[0, 2].plot(range(len(kl)), kl, '-', lw=1.5,
                           label=f'sigma={sigma_str}')
    axes[0, 2].set_xlabel('Token Position')
    axes[0, 2].set_ylabel('KL Divergence')
    axes[0, 2].set_title('(c) Output Perturbation by Position', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    # (d) Speed vs sigma
    sigmas = [float(s) for s in all_results.keys()]
    speeds = [np.mean([r['mean_speed'] for r in all_results[s]]) for s in all_results.keys()]
    axes[1, 0].plot(sigmas, speeds, 'ro-', lw=2, markersize=8)
    axes[1, 0].set_xlabel('Noise Sigma')
    axes[1, 0].set_ylabel('Speed (positions/layer)')
    axes[1, 0].set_title('(d) Speed of Sound vs Noise Amplitude', fontweight='bold')
    axes[1, 0].set_xscale('log'); axes[1, 0].grid(alpha=0.3)

    # (e) Delta at position 0 vs layer (decay of source)
    for sigma_str in ['0.01', '0.1', '1.0']:
        if sigma_str in all_results:
            d = all_results[sigma_str][0]['delta_map']
            axes[1, 1].plot(range(d.shape[0]), d[:, 0], '-', lw=1.5,
                           label=f'sigma={sigma_str}')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Relative Delta at pos=0')
    axes[1, 1].set_title('(e) Source Perturbation Decay', fontweight='bold')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    # (f) Summary
    summary = "SPEED OF SOUND IN TOKEN FLOW\n\n"
    summary += "Noise injected at position 0\n"
    summary += "Propagation measured via attention\n\n"
    for sigma_str, results in all_results.items():
        sp = np.mean([r['mean_speed'] for r in results])
        mx = np.mean([r['max_reach'] for r in results])
        summary += f"sigma={sigma_str}: speed={sp:.2f}, reach={mx:.0f}\n"
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                   transform=axes[1, 2].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 271: Speed of Sound in Token Flow",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase271_speed_of_sound')
    plt.close()

    # Serialize results (remove numpy arrays)
    serializable = {}
    for sigma_str, results in all_results.items():
        serializable[sigma_str] = [{
            'seq_len': r['seq_len'], 'n_layers': r['n_layers'],
            'sigma': r['sigma'], 'mean_speed': r['mean_speed'],
            'max_reach': r['max_reach'],
            'kl_per_pos': [round(x, 6) for x in r['kl_per_pos']],
            'wavefront_per_layer': r['wavefront_per_layer'],
        } for r in results]

    save_results('phase271_speed_of_sound', {
        'experiment': 'Speed of Sound in Token Flow',
        'results': serializable,
    })

    del model, tok
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
