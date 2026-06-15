# -*- coding: utf-8 -*-
"""
Phase 178: Thermodynamic Regularization
Test whether constraining generation to maintain eta~0.813 reduces hallucination.
Compare free generation vs eta-constrained generation quality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, save_results, save_figure

GENERATION_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The theory of general relativity was developed by",
    "The chemical formula for water is",
    "The largest planet in our solar system is",
    "Photosynthesis occurs in the",
    "The speed of light in vacuum is approximately",
    "DNA stands for",
]


def compute_layer_eta(model, hidden_states):
    """Compute eta from hidden states."""
    n = len(hidden_states)
    T_vals = []
    for li in range(n):
        hs = hidden_states[li]
        with torch.no_grad():
            normed = model.model.norm(hs[:, -1:, :])
            logits = model.lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()
        T_vals.append(T if not np.isnan(T) else 0.0)
    T_hot = np.mean(T_vals[:3])
    T_cold = np.mean(T_vals[-3:])
    eta = 1 - T_cold / (T_hot + 1e-10)
    return eta, T_hot, T_cold


def generate_tokens(model, tok, prompt, device, n_tokens=30, eta_target=None, eta_penalty=2.0):
    """Generate tokens, optionally with eta regularization."""
    input_ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
    generated = []
    etas = []
    confidences = []

    for step in range(n_tokens):
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)

        logits = out.logits[0, -1, :].float()
        eta, T_hot, T_cold = compute_layer_eta(model, out.hidden_states)
        etas.append(eta)

        if eta_target is not None and not np.isnan(eta):
            # Penalize tokens whose eta deviates from target
            eta_dev = abs(eta - eta_target)
            # Temperature scaling: increase temperature when eta is off
            temp = 1.0 + eta_penalty * eta_dev
            logits = logits / temp

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(0)
        confidence = probs.max().item()
        confidences.append(confidence)

        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        # Truncate context if too long
        if input_ids.shape[1] > 256:
            input_ids = input_ids[:, -256:]

    text = tok.decode(generated, skip_special_tokens=True)
    return text, etas, confidences


def main():
    print("=" * 70)
    print("Phase 178: Thermodynamic Regularization")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)

    eta_target = 0.813
    n_tokens = 30
    results_free = []
    results_reg = []

    for prompt in GENERATION_PROMPTS:
        print(f"\nPrompt: {prompt}")

        # Free generation
        text_free, etas_free, conf_free = generate_tokens(
            model, tok, prompt, device, n_tokens=n_tokens)
        results_free.append({
            'prompt': prompt, 'text': text_free,
            'etas': etas_free, 'confidences': conf_free,
        })
        print(f"  Free: {text_free[:60]}...")

        # Eta-regularized generation
        text_reg, etas_reg, conf_reg = generate_tokens(
            model, tok, prompt, device, n_tokens=n_tokens,
            eta_target=eta_target, eta_penalty=2.0)
        results_reg.append({
            'prompt': prompt, 'text': text_reg,
            'etas': etas_reg, 'confidences': conf_reg,
        })
        print(f"  Reg:  {text_reg[:60]}...")

    # === Analysis ===
    free_etas_all = [e for r in results_free for e in r['etas']]
    reg_etas_all = [e for r in results_reg for e in r['etas']]
    free_conf_all = [c for r in results_free for c in r['confidences']]
    reg_conf_all = [c for r in results_reg for c in r['confidences']]

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) eta distribution
    axes[0, 0].hist(free_etas_all, bins=20, alpha=0.6, color='#e74c3c', label='Free', density=True)
    axes[0, 0].hist(reg_etas_all, bins=20, alpha=0.6, color='#3498db', label='Regularized', density=True)
    axes[0, 0].axvline(x=eta_target, color='black', linestyle='--', label=f'Target $\\eta={eta_target}$')
    axes[0, 0].set_xlabel('$\\eta$')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('(a) Efficiency Distribution')
    axes[0, 0].legend(fontsize=8)

    # (b) eta over generation steps
    for i, (rf, rr) in enumerate(zip(results_free[:3], results_reg[:3])):
        axes[0, 1].plot(rf['etas'], '--', alpha=0.5, color=f'C{i}')
        axes[0, 1].plot(rr['etas'], '-', alpha=0.8, color=f'C{i}', label=rf['prompt'][:20])
    axes[0, 1].axhline(y=eta_target, color='black', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Generation Step')
    axes[0, 1].set_ylabel('$\\eta$')
    axes[0, 1].set_title('(b) $\\eta$ During Generation (solid=reg, dash=free)')
    axes[0, 1].legend(fontsize=6)

    # (c) Confidence comparison
    axes[0, 2].hist(free_conf_all, bins=20, alpha=0.6, color='#e74c3c', label='Free', density=True)
    axes[0, 2].hist(reg_conf_all, bins=20, alpha=0.6, color='#3498db', label='Regularized', density=True)
    axes[0, 2].set_xlabel('Token Confidence')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('(c) Confidence Distribution')
    axes[0, 2].legend(fontsize=8)

    # (d) eta deviation from target
    free_dev = [abs(e - eta_target) for e in free_etas_all]
    reg_dev = [abs(e - eta_target) for e in reg_etas_all]
    axes[1, 0].boxplot([free_dev, reg_dev], labels=['Free', 'Regularized'])
    axes[1, 0].set_ylabel('$|\\eta - 0.813|$')
    axes[1, 0].set_title('(d) Deviation from Target $\\eta$')

    # (e) Per-prompt eta mean
    free_means = [np.mean(r['etas']) for r in results_free]
    reg_means = [np.mean(r['etas']) for r in results_reg]
    x = np.arange(len(GENERATION_PROMPTS))
    axes[1, 1].bar(x - 0.2, free_means, 0.35, label='Free', color='#e74c3c', alpha=0.7)
    axes[1, 1].bar(x + 0.2, reg_means, 0.35, label='Regularized', color='#3498db', alpha=0.7)
    axes[1, 1].axhline(y=eta_target, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Prompt Index')
    axes[1, 1].set_ylabel('Mean $\\eta$')
    axes[1, 1].set_title('(e) Per-Prompt Mean $\\eta$')
    axes[1, 1].legend(fontsize=8)

    # (f) Summary
    summary = (
        f"Thermodynamic Regularization\n\n"
        f"Target: eta = {eta_target}\n\n"
        f"FREE generation:\n"
        f"  Mean eta: {np.mean(free_etas_all):.3f}\n"
        f"  Mean conf: {np.mean(free_conf_all):.3f}\n"
        f"  eta std: {np.std(free_etas_all):.3f}\n\n"
        f"REGULARIZED generation:\n"
        f"  Mean eta: {np.mean(reg_etas_all):.3f}\n"
        f"  Mean conf: {np.mean(reg_conf_all):.3f}\n"
        f"  eta std: {np.std(reg_etas_all):.3f}\n\n"
        f"Closer to target: {'REG' if np.mean(reg_dev) < np.mean(free_dev) else 'FREE'}"
    )
    axes[1, 2].text(0.5, 0.5, summary, ha='center', va='center',
                    transform=axes[1, 2].transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    family='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('(f) Summary')

    fig.suptitle('Phase 178: Thermodynamic Regularization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'phase178_thermo_regularization')
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"Free: eta={np.mean(free_etas_all):.3f}, conf={np.mean(free_conf_all):.3f}")
    print(f"Reg:  eta={np.mean(reg_etas_all):.3f}, conf={np.mean(reg_conf_all):.3f}")
    print(f"{'=' * 70}")

    save_results('phase178_thermo_regularization', {
        'experiment': 'Thermodynamic Regularization',
        'eta_target': eta_target,
        'free': {'mean_eta': float(np.mean(free_etas_all)), 'mean_conf': float(np.mean(free_conf_all))},
        'regularized': {'mean_eta': float(np.mean(reg_etas_all)), 'mean_conf': float(np.mean(reg_conf_all))},
        'results_free': [{'prompt': r['prompt'], 'text': r['text'],
                          'mean_eta': float(np.mean(r['etas']))} for r in results_free],
        'results_reg': [{'prompt': r['prompt'], 'text': r['text'],
                         'mean_eta': float(np.mean(r['etas']))} for r in results_reg],
    })


if __name__ == '__main__':
    main()
