# -*- coding: utf-8 -*-
"""
Phase 206: LayerNorm Flywheel (Waste Heat Recovery)
=====================================================
LayerNorm normalizes hidden states, discarding energy (variance reduction).
This "waste heat" is normally lost. We recapture it by injecting noise
proportional to the discarded energy into the next FFN layer.

Physics: Brownian ratchet / Stochastic resonance recycling.
If the model is "confused" (high waste), exploration increases automatically.
If the model is "confident" (low waste), it cruises efficiently.
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
    "The fundamental theorem of calculus connects",
    "Quantum mechanics describes particles at atomic scale",
    "Neural networks learn through gradient descent",
    "Black holes form from gravitational collapse",
    "The periodic table organizes chemical elements",
    "Evolution operates on heritable variation",
    "Photosynthesis converts sunlight to chemical energy",
    "Machine learning discovers hidden patterns",
    "General relativity describes gravity as spacetime curvature",
    "Protein folding determines biological function",
]

# Alpha values: how much waste heat to reinject (0 = baseline, no recycling)
ALPHA_VALUES = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


class FlywheelState:
    """Shared state for LayerNorm waste heat tracking."""
    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.delta_u_history = []  # Per-layer waste heat
        self.current_delta_u = 0.0
        self.noise_injected = []


def make_layernorm_measure_hook(state, layer_idx):
    """Hook on post_attention_layernorm: measure energy before/after norm."""
    def hook(module, input, output):
        h_in = input[0].float()
        h_out = output.float()
        # Energy = L2 norm of the last token
        u_before = h_in[0, -1, :].norm().item()
        u_after = h_out[0, -1, :].norm().item()
        delta_u = max(u_before - u_after, 0)  # Only recover positive waste
        state.current_delta_u = delta_u
        state.delta_u_history.append((layer_idx, delta_u))
    return hook


def make_ffn_inject_hook(state, layer_idx):
    """Hook on MLP: inject noise proportional to waste heat from layernorm."""
    def hook(module, input, output):
        if state.alpha <= 0 or state.current_delta_u <= 0:
            state.noise_injected.append((layer_idx, 0.0))
            return output
        sigma = state.alpha * state.current_delta_u
        h = output if isinstance(output, torch.Tensor) else output[0]
        noise = torch.randn_like(h.float()) * sigma
        h_mod = (h.float() + noise).to(h.dtype)
        h_mod = torch.nan_to_num(h_mod, nan=0.0, posinf=65000.0, neginf=-65000.0)
        state.noise_injected.append((layer_idx, sigma))
        if isinstance(output, tuple):
            return (h_mod,) + output[1:]
        return h_mod
    return hook


def run_with_flywheel(model, tok, device, prompt, alpha):
    """Forward pass with flywheel waste heat recovery."""
    state = FlywheelState(alpha=alpha)
    handles = []

    # Install hooks on each layer
    for li, layer in enumerate(model.model.layers):
        h1 = layer.post_attention_layernorm.register_forward_hook(
            make_layernorm_measure_hook(state, li))
        h2 = layer.mlp.register_forward_hook(
            make_ffn_inject_hook(state, li))
        handles.extend([h1, h2])

    try:
        inp = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        # Measure thermodynamics
        norm_layer = model.model.norm
        lm_head = model.lm_head
        n_hs = len(out.hidden_states)

        U_list, T_list = [], []
        for hs in out.hidden_states:
            h = hs[0, -1, :].float()
            U_list.append(h.norm().item())
            with torch.no_grad():
                normed = norm_layer(hs[:, -1:, :])
                logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T if not np.isnan(T) else 0)

        T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
        T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
        eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

        # Output quality
        final_logits = out.logits[0, -1, :].float()
        final_probs = torch.softmax(final_logits, dim=-1)
        output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()
        top1_prob = final_probs.max().item()
        top_token = tok.decode(final_logits.argmax().item())

        # Waste heat stats
        waste_per_layer = [d for _, d in state.delta_u_history]
        noise_per_layer = [n for _, n in state.noise_injected]

    finally:
        for h in handles:
            h.remove()

    return {
        'eta': eta, 'T_hot': T_hot, 'T_cold': T_cold,
        'output_entropy': output_entropy, 'top1_prob': top1_prob,
        'top_token': top_token,
        'U': U_list, 'T': T_list,
        'waste_per_layer': waste_per_layer,
        'noise_per_layer': noise_per_layer,
        'total_waste': sum(waste_per_layer),
        'total_noise': sum(noise_per_layer),
    }


def main():
    print("=" * 70)
    print("Phase 206: LayerNorm Flywheel (Waste Heat Recovery)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    results_by_alpha = {}
    for ai, alpha in enumerate(ALPHA_VALUES):
        label = f"alpha_{alpha}"
        print(f"\n[{ai+1}/{len(ALPHA_VALUES)}] alpha={alpha} "
              f"{'(baseline)' if alpha == 0 else ''}")

        all_eta, all_ent, all_top1, all_waste = [], [], [], []
        example_waste, example_noise, example_T = None, None, None

        for pi, prompt in enumerate(PROMPTS):
            r = run_with_flywheel(model, tok, device, prompt, alpha)
            all_eta.append(r['eta'])
            all_ent.append(r['output_entropy'])
            all_top1.append(r['top1_prob'])
            all_waste.append(r['total_waste'])
            if pi == 0:
                example_waste = r['waste_per_layer']
                example_noise = r['noise_per_layer']
                example_T = r['T']

        results_by_alpha[label] = {
            'alpha': alpha,
            'eta_mean': float(np.mean(all_eta)),
            'eta_std': float(np.std(all_eta)),
            'entropy_mean': float(np.mean(all_ent)),
            'top1_mean': float(np.mean(all_top1)),
            'total_waste_mean': float(np.mean(all_waste)),
            'example_waste': [float(x) for x in example_waste] if example_waste else [],
            'example_noise': [float(x) for x in example_noise] if example_noise else [],
            'example_T': [float(x) for x in example_T] if example_T else [],
        }
        print(f"  eta={np.mean(all_eta):.4f}, entropy={np.mean(all_ent):.3f}, "
              f"top1={np.mean(all_top1):.4f}, waste={np.mean(all_waste):.1f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    alphas = [r['alpha'] for r in results_by_alpha.values()]
    alpha_labels = [str(a) for a in alphas]

    # (a) eta vs alpha
    etas = [r['eta_mean'] for r in results_by_alpha.values()]
    axes[0, 0].plot(range(len(etas)), etas, 'o-', color='#e74c3c', markersize=8, lw=2)
    axes[0, 0].set_xticks(range(len(alpha_labels)))
    axes[0, 0].set_xticklabels(alpha_labels, fontsize=8)
    axes[0, 0].set_xlabel('Flywheel Alpha (recycling rate)')
    axes[0, 0].set_ylabel('Carnot Efficiency eta')
    axes[0, 0].set_title('(a) Efficiency vs Recycling Rate')

    # (b) Output entropy vs alpha
    ents = [r['entropy_mean'] for r in results_by_alpha.values()]
    axes[0, 1].plot(range(len(ents)), ents, 's-', color='#3498db', markersize=8, lw=2)
    axes[0, 1].set_xticks(range(len(alpha_labels)))
    axes[0, 1].set_xticklabels(alpha_labels, fontsize=8)
    axes[0, 1].set_xlabel('Flywheel Alpha')
    axes[0, 1].set_ylabel('Output Entropy (nats)')
    axes[0, 1].set_title('(b) Output Quality vs Recycling')

    # (c) Top-1 prob vs alpha
    top1s = [r['top1_mean'] for r in results_by_alpha.values()]
    axes[0, 2].plot(range(len(top1s)), top1s, 'D-', color='#2ecc71', markersize=8, lw=2)
    axes[0, 2].set_xticks(range(len(alpha_labels)))
    axes[0, 2].set_xticklabels(alpha_labels, fontsize=8)
    axes[0, 2].set_xlabel('Flywheel Alpha')
    axes[0, 2].set_ylabel('Top-1 Probability')
    axes[0, 2].set_title('(c) Confidence vs Recycling')

    # (d) Waste heat profile (per layer, baseline)
    base = results_by_alpha['alpha_0']
    if base['example_waste']:
        axes[1, 0].bar(range(len(base['example_waste'])), base['example_waste'],
                       color='#e67e22', alpha=0.7)
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Waste Heat (delta U)')
        axes[1, 0].set_title('(d) Waste Heat Profile (baseline)')

    # (e) Temperature profiles: baseline vs best flywheel
    best_alpha_key = None
    best_ent_drop = 0
    base_ent = results_by_alpha['alpha_0']['entropy_mean']
    for k, v in results_by_alpha.items():
        if v['alpha'] > 0:
            drop = base_ent - v['entropy_mean']
            if drop > best_ent_drop:
                best_ent_drop = drop
                best_alpha_key = k

    if base['example_T']:
        axes[1, 1].plot(range(len(base['example_T'])), base['example_T'],
                        '-', color='#95a5a6', lw=2, label='baseline')
    if best_alpha_key and results_by_alpha[best_alpha_key]['example_T']:
        best_T = results_by_alpha[best_alpha_key]['example_T']
        axes[1, 1].plot(range(len(best_T)), best_T,
                        '-', color='#e74c3c', lw=2,
                        label=f'flywheel ({best_alpha_key})')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Temperature T')
    axes[1, 1].set_title('(e) Temperature: Baseline vs Flywheel')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    base_r = results_by_alpha['alpha_0']
    summary_text = (
        f"LayerNorm Flywheel\n\n"
        f"Baseline (alpha=0):\n"
        f"  eta = {base_r['eta_mean']:.4f}\n"
        f"  entropy = {base_r['entropy_mean']:.3f}\n"
        f"  top1 = {base_r['top1_mean']:.4f}\n"
        f"  total waste = {base_r['total_waste_mean']:.1f}\n\n"
    )
    if best_alpha_key:
        best_r = results_by_alpha[best_alpha_key]
        pct = (best_r['entropy_mean'] - base_r['entropy_mean']) / (base_r['entropy_mean'] + 1e-10) * 100
        summary_text += (
            f"Best flywheel ({best_alpha_key}):\n"
            f"  eta = {best_r['eta_mean']:.4f}\n"
            f"  entropy = {best_r['entropy_mean']:.3f}\n"
            f"  entropy change: {pct:+.1f}%\n"
        )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 206: LayerNorm Flywheel", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase206_flywheel')
    plt.close()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Baseline entropy: {base_r['entropy_mean']:.3f}")
    if best_alpha_key:
        best_r = results_by_alpha[best_alpha_key]
        print(f"Best flywheel ({best_alpha_key}): {best_r['entropy_mean']:.3f}")
    print(f"{'=' * 70}")

    save_results('phase206_flywheel', {
        'experiment': 'LayerNorm Flywheel',
        'results': results_by_alpha,
    })


if __name__ == '__main__':
    main()
