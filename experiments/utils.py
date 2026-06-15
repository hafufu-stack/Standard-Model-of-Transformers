# -*- coding: utf-8 -*-
"""
Standard Model of Transformers: Shared utilities
Extends S-Qubit utils with cross-project functionality.
"""
import os, sys, torch, json, time, csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Model paths (local snapshots)
# ============================================================
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
_SNAP_1B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-1.5B",
                         "snapshots", "8faed761d45a263340a0528343f099c05c9a4323")
_SNAP_0B5 = os.path.join(_HF_CACHE, "models--Qwen--Qwen2.5-0.5B",
                         "snapshots", "060db6499f32faf8b98477b0a26969ef7d8b9987")

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(EXPERIMENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model(device=None, dtype=None, size='1.5B'):
    """Load Qwen2.5 with local_files_only=True."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dtype is None:
        dtype = torch.float16 if device == 'cuda' else torch.float32

    if size == '0.5B':
        mid = _SNAP_0B5
    else:
        mid = _SNAP_1B5 if os.path.exists(_SNAP_1B5) else _SNAP_0B5

    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=dtype, device_map=device, local_files_only=True,
        # NOTE: attn_implementation='eager' causes NaN in transformers 5.0.0 + fp16
        # Removed for Season 3 (no raw attention weights needed)
    )
    model.eval()
    return model, tok


def get_hidden_states(model, tok, prompt, device='cuda'):
    """Extract hidden states from all layers for a prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # out.hidden_states: tuple of (batch, seq, hidden) for each layer
    return [h[0, -1, :].cpu().float() for h in out.hidden_states]


def get_logits(model, tok, prompt, device='cuda'):
    """Single forward pass, return final logits (vocab,)."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1, :]


def inject_hook(model, layer_idx, hook_fn):
    """Register a forward hook on model.model.layers[layer_idx]."""
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    return handle


def measure_temperature(hidden_states):
    """Measure 'temperature' (L2 norm variance) at each layer."""
    temps = []
    for h in hidden_states:
        temps.append(h.norm().item())
    return temps


def measure_participation_ratio(logits):
    """Measure participation ratio from output logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    pr = 1.0 / (probs ** 2).sum().item()
    return pr


def save_results(name, data):
    """Save experiment results as JSON."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved: {path}")
    return path


def save_figure(fig, name):
    """Save matplotlib figure."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {path}")
    return path


def load_any_model(model_id, device=None, dtype=None):
    """Load any CausalLM from HF cache (local_files_only).
    
    Args:
        model_id: HuggingFace model ID, e.g. 'meta-llama/Llama-3.2-1B'
        device: 'cuda' or 'cpu'
        dtype: torch dtype
    
    Returns:
        model, tokenizer
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dtype is None:
        dtype = torch.float16 if device == 'cuda' else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device,
        local_files_only=True, trust_remote_code=True,
    )
    model.eval()
    return model, tok


def get_model_internals(model):
    """Get norm_layer, lm_head, and transformer layers for any architecture.
    
    Returns:
        dict with keys: 'norm', 'lm_head', 'layers', 'n_layers'
    """
    # Try common attribute patterns
    # Qwen, Llama, Mistral, StableLM: model.model.norm, model.lm_head, model.model.layers
    # GPT2: model.transformer.ln_f, model.lm_head, model.transformer.h
    # OPT: model.model.decoder.final_layer_norm, model.lm_head, model.model.decoder.layers
    # Bloom: model.transformer.ln_f, model.lm_head, model.transformer.h
    # Phi-2: model.model.final_layernorm, model.lm_head, model.model.layers
    # Falcon: model.transformer.ln_f, model.lm_head, model.transformer.h

    lm_head = model.lm_head

    # Try each pattern
    patterns = [
        # (norm_path, layers_path)
        ('model.model.norm', 'model.model.layers'),           # Qwen, Llama, Mistral, Phi-3
        ('model.transformer.ln_f', 'model.transformer.h'),    # GPT2, Bloom, Falcon
        ('model.model.decoder.final_layer_norm', 'model.model.decoder.layers'),  # OPT
        ('model.model.final_layernorm', 'model.model.layers'),  # Phi-2
    ]

    for norm_path, layers_path in patterns:
        try:
            norm = model
            for attr in norm_path.split('.')[1:]:  # Skip 'model' (it's the param)
                norm = getattr(norm, attr)
            layers = model
            for attr in layers_path.split('.')[1:]:
                layers = getattr(layers, attr)
            return {
                'norm': norm,
                'lm_head': lm_head,
                'layers': layers,
                'n_layers': len(layers),
            }
        except AttributeError:
            continue

    raise ValueError(f"Cannot find model internals for {type(model).__name__}")


def make_safe_noise_hook(sigma):
    """Deep Think's Safety Valve Hook: fp32 noise + nan_to_num clamp."""
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h_fp32 = h.to(torch.float32)
        noise = torch.randn_like(h_fp32) * sigma
        h_mod = h_fp32 + noise
        h_mod = torch.nan_to_num(h_mod, nan=0.0, posinf=65000.0, neginf=-65000.0)
        result = h_mod.to(h.dtype)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook


def measure_full_thermodynamics(model, tok, prompt, device):
    """Measure U, T, PR, PRT at every layer for a prompt."""
    inp = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    n_hs = len(out.hidden_states)
    lm_head = model.lm_head
    norm_layer = model.model.norm  # RMSNorm before lm_head

    results = []
    for layer_idx, hs in enumerate(out.hidden_states):
        h = hs[0, -1, :].float()
        U = h.norm().item()

        # PR from hidden state activation distribution
        h_sq = h ** 2
        h_prob = h_sq / (h_sq.sum() + 1e-10)
        PR = 1.0 / (h_prob ** 2).sum().item()

        # T from logits (apply final norm + lm_head)
        with torch.no_grad():
            normed = norm_layer(hs[:, -1:, :])
            logits = lm_head(normed).squeeze().float()
        probs = torch.softmax(logits, dim=-1)
        T = -(probs * torch.log(probs + 1e-10)).sum().item()

        if np.isnan(T):
            T = 0.0
        if np.isnan(PR):
            PR = 1.0

        results.append({'layer': layer_idx, 'U': U, 'T': T, 'PR': PR, 'PRT': PR * T})
    return results, out

