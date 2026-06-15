# -*- coding: utf-8 -*-
"""
Phase 203: Information Bottleneck Forcing
==========================================
Phase 198 showed the IB transition at layer 9. Can we exploit this
by forcing SVD compression at L9 and teleporting to L21 (skipping
layers 10-20)?

This is "physics-based model compression": use the IB transition
as a natural compression point.
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

L0 = 21
L_IB = 9  # Information bottleneck transition layer
SVD_KEEP_FRACTIONS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
TELEPORT_CONFIGS = [
    ('Full', None, None),          # No teleportation
    ('Skip 10-15', 10, 16),        # Mild skip
    ('Skip 10-20', 10, 21),        # Deep Think's proposal: skip to L0
    ('Skip 5-20', 5, 21),          # Aggressive skip
]


def run_with_svd_and_teleport(model, tok, device, prompt, svd_keep=1.0,
                               skip_start=None, skip_end=None):
    """Forward pass with optional SVD compression at L_IB and layer teleportation."""
    n_layers = len(model.model.layers)
    norm_layer = model.model.norm
    lm_head = model.lm_head

    inp = tok(prompt, return_tensors='pt').to(device)
    input_ids = inp['input_ids']

    # Manual forward pass to control the flow
    with torch.no_grad():
        # Embedding
        hidden = model.model.embed_tokens(input_ids)

        U_list = [hidden[0, -1, :].float().norm().item()]
        T_list = []

        # Measure initial T
        normed = norm_layer(hidden[:, -1:, :])
        logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_list.append(T_val if not np.isnan(T_val) else 0)

        # Compute position embeddings once (needed by Qwen2 layers)
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)

        layers_executed = 0
        for li in range(n_layers):
            # Skip layers if teleporting
            if skip_start is not None and skip_end is not None:
                if skip_start <= li < skip_end:
                    continue

            # Run layer with position embeddings
            layer = model.model.layers[li]
            layer_out = layer(hidden, position_embeddings=position_embeddings)
            hidden = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]
            layers_executed += 1

            # SVD compression at L_IB
            if li == L_IB and svd_keep < 1.0:
                h = hidden[0].float()  # (seq, hidden)
                U_svd, S_svd, Vh = torch.linalg.svd(h, full_matrices=False)
                k = max(1, int(len(S_svd) * svd_keep))
                # Reconstruct with top-k singular values
                h_compressed = (U_svd[:, :k] * S_svd[:k]) @ Vh[:k, :]
                hidden = h_compressed.unsqueeze(0).to(hidden.dtype)

            # Measure thermodynamics
            h_vec = hidden[0, -1, :].float()
            U_list.append(h_vec.norm().item())
            normed = norm_layer(hidden[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
            probs = torch.softmax(logits, dim=-1)
            T_val = -(probs * torch.log(probs + 1e-10)).sum().item()
            T_list.append(T_val if not np.isnan(T_val) else 0)

        # Final logits for PPL
        normed = norm_layer(hidden)
        final_logits = lm_head(normed)

    # Calculate output entropy (proxy for PPL)
    final_probs = torch.softmax(final_logits[0, -1, :].float(), dim=-1)
    output_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum().item()

    # Top-1 token
    top_token = tok.decode(final_logits[0, -1, :].argmax().item())

    T_hot = max(T_list[1:]) if len(T_list) > 1 else 0
    T_cold = min(T_list[1:]) if len(T_list) > 1 else 0
    eta = 1 - T_cold / (T_hot + 1e-10) if T_hot > 0 else 0

    return {
        'U': U_list, 'T': T_list, 'eta': eta,
        'output_entropy': output_entropy, 'top_token': top_token,
        'layers_executed': layers_executed,
    }


def main():
    print("=" * 70)
    print("Phase 203: Information Bottleneck Forcing")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device)

    # === Part 1: SVD compression at L_IB ===
    print("\n=== Part 1: SVD Compression at L_IB ===")
    svd_results = {}
    for frac in SVD_KEEP_FRACTIONS:
        all_entropy = []
        all_eta = []
        for prompt in PROMPTS:
            r = run_with_svd_and_teleport(model, tok, device, prompt, svd_keep=frac)
            all_entropy.append(r['output_entropy'])
            all_eta.append(r['eta'])
        mean_ent = np.mean(all_entropy)
        mean_eta = np.mean(all_eta)
        svd_results[frac] = {'entropy': mean_ent, 'eta': mean_eta}
        print(f"  SVD keep={frac:.0%}: entropy={mean_ent:.3f}, eta={mean_eta:.4f}")

    # === Part 2: Layer Teleportation ===
    print("\n=== Part 2: Layer Teleportation ===")
    teleport_results = {}
    for name, skip_s, skip_e in TELEPORT_CONFIGS:
        all_entropy = []
        all_eta = []
        all_layers = []
        for prompt in PROMPTS:
            r = run_with_svd_and_teleport(model, tok, device, prompt,
                                          skip_start=skip_s, skip_end=skip_e)
            all_entropy.append(r['output_entropy'])
            all_eta.append(r['eta'])
            all_layers.append(r['layers_executed'])
        mean_ent = np.mean(all_entropy)
        mean_eta = np.mean(all_eta)
        mean_layers = np.mean(all_layers)
        teleport_results[name] = {'entropy': mean_ent, 'eta': mean_eta,
                                   'layers': mean_layers}
        print(f"  {name}: entropy={mean_ent:.3f}, eta={mean_eta:.4f}, "
              f"layers={mean_layers:.0f}")

    # === Part 3: Combined SVD + Teleport ===
    print("\n=== Part 3: SVD(30%) + Skip(10-20) ===")
    combo_results = []
    for prompt in PROMPTS:
        r = run_with_svd_and_teleport(model, tok, device, prompt,
                                      svd_keep=0.3, skip_start=10, skip_end=21)
        combo_results.append(r)
    combo_ent = np.mean([r['output_entropy'] for r in combo_results])
    combo_eta = np.mean([r['eta'] for r in combo_results])
    combo_layers = np.mean([r['layers_executed'] for r in combo_results])
    print(f"  Combined: entropy={combo_ent:.3f}, eta={combo_eta:.4f}, "
          f"layers={combo_layers:.0f}")

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) SVD keep fraction vs output entropy
    fracs = list(svd_results.keys())
    ents = [svd_results[f]['entropy'] for f in fracs]
    axes[0, 0].plot(fracs, ents, 'o-', color='#e74c3c', markersize=8, linewidth=2)
    axes[0, 0].set_xlabel('SVD Keep Fraction')
    axes[0, 0].set_ylabel('Output Entropy (nats)')
    axes[0, 0].set_title('(a) SVD Compression Quality')
    axes[0, 0].invert_xaxis()

    # (b) SVD keep fraction vs eta
    etas = [svd_results[f]['eta'] for f in fracs]
    axes[0, 1].plot(fracs, etas, 's-', color='#3498db', markersize=8, linewidth=2)
    axes[0, 1].set_xlabel('SVD Keep Fraction')
    axes[0, 1].set_ylabel('Carnot Efficiency eta')
    axes[0, 1].set_title('(b) Efficiency under Compression')
    axes[0, 1].invert_xaxis()

    # (c) Teleportation: layers saved vs entropy degradation
    names = list(teleport_results.keys())
    t_ents = [teleport_results[n]['entropy'] for n in names]
    t_layers = [teleport_results[n]['layers'] for n in names]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = axes[0, 2].bar(names, t_ents, color=colors[:len(names)],
                          edgecolor='black', alpha=0.8)
    axes[0, 2].set_ylabel('Output Entropy')
    axes[0, 2].set_title('(c) Teleportation Quality')
    axes[0, 2].tick_params(axis='x', rotation=15)

    # (d) Temperature profiles for teleportation configs
    for ci, (name, skip_s, skip_e) in enumerate(TELEPORT_CONFIGS):
        r = run_with_svd_and_teleport(model, tok, device, PROMPTS[0],
                                      skip_start=skip_s, skip_end=skip_e)
        axes[1, 0].plot(range(len(r['T'])), r['T'], 'o-', color=colors[ci],
                        label=name, markersize=3, linewidth=1.5)
    axes[1, 0].axvline(x=L0, color='#f39c12', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Layer (executed)')
    axes[1, 0].set_ylabel('Temperature T')
    axes[1, 0].set_title('(d) T Profiles (Teleportation)')
    axes[1, 0].legend(fontsize=7)

    # (e) FLOPs savings
    full_layers = teleport_results['Full']['layers']
    savings = [(full_layers - teleport_results[n]['layers']) / full_layers * 100
               for n in names]
    axes[1, 1].bar(names, savings, color=colors[:len(names)],
                   edgecolor='black', alpha=0.8)
    axes[1, 1].set_ylabel('FLOPs Saved (%)')
    axes[1, 1].set_title('(e) Computational Savings')
    axes[1, 1].tick_params(axis='x', rotation=15)

    # (f) Summary
    baseline_ent = svd_results[1.0]['entropy']
    best_skip = 'Skip 10-20'
    summary_text = (
        f"IB Forcing Results\n\n"
        f"Baseline entropy: {baseline_ent:.3f}\n\n"
        f"SVD at L9 (keep 30%):\n"
        f"  entropy: {svd_results[0.3]['entropy']:.3f}\n"
        f"  degradation: {(svd_results[0.3]['entropy']-baseline_ent)/baseline_ent*100:.1f}%\n\n"
        f"Teleport (skip 10-20):\n"
        f"  entropy: {teleport_results[best_skip]['entropy']:.3f}\n"
        f"  layers: {teleport_results[best_skip]['layers']:.0f}/{full_layers:.0f}\n"
        f"  savings: {savings[names.index(best_skip)]:.0f}%\n\n"
        f"Combined (SVD+Skip):\n"
        f"  entropy: {combo_ent:.3f}\n"
        f"  layers: {combo_layers:.0f}"
    )
    axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle("Phase 203: Information Bottleneck Forcing", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase203_ib_forcing')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"SVD(30%) entropy: {svd_results[0.3]['entropy']:.3f} "
          f"(baseline: {baseline_ent:.3f})")
    print(f"Skip(10-20) saves {savings[names.index(best_skip)]:.0f}% FLOPs")
    print(f"{'=' * 70}")

    save_results('phase203_ib_forcing', {
        'experiment': 'Information Bottleneck Forcing',
        'svd_results': {str(k): v for k, v in svd_results.items()},
        'teleport_results': teleport_results,
        'combined': {'entropy': combo_ent, 'eta': combo_eta, 'layers': combo_layers},
        'summary': {
            'baseline_entropy': baseline_ent,
            'best_svd_keep': 0.3,
            'svd_degradation_pct': (svd_results[0.3]['entropy'] - baseline_ent) / baseline_ent * 100,
            'skip_savings_pct': savings[names.index(best_skip)],
        }
    })


if __name__ == '__main__':
    main()
