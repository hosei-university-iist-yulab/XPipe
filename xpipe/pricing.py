#!/usr/bin/env python3
# xpipe/pricing.py
# ======================================================================
# Model Pricing Database for Cost Tracking
#
# Models supported (7 total):
#   - LOCAL FREE (5): gpt2, distilgpt2, Qwen-3B, Phi-3, TinyLlama
#   - API PAID (2): Claude-4.5-Sonnet, Gemini-1.5-Flash [FROZEN for now]
#
# Note: Paid APIs are frozen for initial implementation.
#       Uncomment when ready to use with API keys.
#
# Usage:
#   from xpipe.pricing import calculate_cost, list_models
#   cost = calculate_cost("gpt2", 1000, 500)
# ======================================================================

from __future__ import annotations
from typing import Dict, Optional, List

# ======================================================================
# MODEL PRICING DATABASE (USD per 1M tokens)
# ======================================================================

MODEL_PRICING: Dict[str, Dict[str, any]] = {

    # ---- LOCAL MODELS (Free) - ACTIVE ----
    "gpt2": {
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
        "provider": "local",
        "params": "124M",
        "active": True
    },
    "distilgpt2": {
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
        "provider": "local",
        "params": "82M",
        "active": True
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
        "provider": "local",
        "params": "3B",
        "active": True
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
        "provider": "local",
        "params": "3.8B",
        "active": True
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "input_per_1m": 0.0,
        "output_per_1m": 0.0,
        "provider": "local",
        "params": "1.1B",
        "active": True
    },

    # ---- ANTHROPIC CLAUDE (API) - FROZEN ----
    # Uncomment 'active': True when ready to use
    "claude-sonnet-4-5-20250929": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "provider": "anthropic",
        "params": "~200B",
        "active": False,  # FROZEN - set to True when API key ready
        "env_key": "ANTHROPIC_API_KEY"
    },

    # ---- GOOGLE GEMINI (API) - FROZEN ----
    # Uncomment 'active': True when ready to use
    "gemini-1.5-flash": {
        "input_per_1m": 0.075,
        "output_per_1m": 0.30,
        "provider": "google",
        "params": "~50B",
        "active": False,  # FROZEN - set to True when API key ready
        "env_key": "GOOGLE_API_KEY"
    },
}


# ======================================================================
# PRICING ALIASES AND SHORTCUTS
# ======================================================================

MODEL_ALIASES: Dict[str, str] = {
    # Anthropic
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "claude-4.5-sonnet": "claude-sonnet-4-5-20250929",
    "claude-4.5": "claude-sonnet-4-5-20250929",
    "claude": "claude-sonnet-4-5-20250929",

    # Google
    "gemini-flash": "gemini-1.5-flash",
    "gemini": "gemini-1.5-flash",

    # Local models (short names)
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


# ======================================================================
# COST CALCULATION FUNCTIONS
# ======================================================================

def get_model_pricing(model_id: str) -> Optional[Dict[str, any]]:
    """Get pricing information for a model."""
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    if model_id in MODEL_ALIASES:
        return MODEL_PRICING.get(MODEL_ALIASES[model_id])
    model_lower = model_id.lower()
    for key in MODEL_PRICING:
        if key.lower() == model_lower:
            return MODEL_PRICING[key]
    return None


def calculate_cost(
    model_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
) -> float:
    """
    Calculate the cost for a model invocation.

    Returns 0.0 for local/free models or unknown models.
    """
    pricing = get_model_pricing(model_id)
    if pricing is None:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output_per_1m"]
    return round(input_cost + output_cost, 6)


def get_cost_summary(model_id: str) -> str:
    """Get a human-readable cost summary for a model."""
    pricing = get_model_pricing(model_id)
    if pricing is None:
        return f"Unknown model: {model_id}"

    if pricing["input_per_1m"] == 0.0:
        status = "" if pricing.get("active", True) else " [FROZEN]"
        return f"FREE ({pricing['provider']}, {pricing['params']}){status}"

    status = "" if pricing.get("active", True) else " [FROZEN]"
    return (f"${pricing['input_per_1m']:.2f}/${pricing['output_per_1m']:.2f} "
            f"per 1M ({pricing['provider']}, {pricing['params']}){status}")


def estimate_query_cost(
    model_id: str,
    avg_prompt_tokens: int = 800,
    avg_completion_tokens: int = 150
) -> str:
    """Estimate the cost per query for a model."""
    cost = calculate_cost(model_id, avg_prompt_tokens, avg_completion_tokens)
    if cost == 0.0:
        return "FREE"
    elif cost < 0.001:
        return f"${cost:.6f} per query"
    elif cost < 0.01:
        return f"${cost:.4f} per query"
    else:
        return f"${cost:.3f} per query"


# ======================================================================
# MODEL UTILITIES
# ======================================================================

def list_models(active_only: bool = True) -> List[str]:
    """Return list of available model IDs."""
    if active_only:
        return [m for m, info in MODEL_PRICING.items() if info.get("active", True)]
    return list(MODEL_PRICING.keys())


def list_active_models() -> List[str]:
    """Return list of active (non-frozen) model IDs."""
    return list_models(active_only=True)


def list_frozen_models() -> List[str]:
    """Return list of frozen (inactive) model IDs."""
    return [m for m, info in MODEL_PRICING.items() if not info.get("active", True)]


def is_model_active(model_id: str) -> bool:
    """Check if a model is active (not frozen)."""
    pricing = get_model_pricing(model_id)
    if pricing is None:
        return False
    return pricing.get("active", True)


def activate_model(model_id: str) -> bool:
    """Activate a frozen model (for use with API key)."""
    if model_id in MODEL_PRICING:
        MODEL_PRICING[model_id]["active"] = True
        return True
    if model_id in MODEL_ALIASES:
        canonical = MODEL_ALIASES[model_id]
        MODEL_PRICING[canonical]["active"] = True
        return True
    return False


def print_pricing_database() -> None:
    """Print all available models and their status."""
    print("\nXPipe Model Pricing Database")
    print("=" * 75)

    # Active models
    active = list_active_models()
    print(f"\nACTIVE MODELS ({len(active)}):")
    print("-" * 75)
    for model_id in active:
        info = MODEL_PRICING[model_id]
        cost = get_cost_summary(model_id)
        print(f"  {model_id:<45} {cost}")

    # Frozen models
    frozen = list_frozen_models()
    if frozen:
        print(f"\nFROZEN MODELS ({len(frozen)}) - Activate when API keys ready:")
        print("-" * 75)
        for model_id in frozen:
            info = MODEL_PRICING[model_id]
            cost = get_cost_summary(model_id)
            env_key = info.get("env_key", "N/A")
            print(f"  {model_id:<45} {cost}")
            print(f"    └─ Requires: {env_key}")

    print("\n" + "=" * 75)
    print(f"Total: {len(MODEL_PRICING)} models ({len(active)} active, {len(frozen)} frozen)")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print_pricing_database()

    print("\n\nModel Aliases:")
    print("-" * 75)
    for alias, target in sorted(MODEL_ALIASES.items()):
        status = "" if is_model_active(target) else " frozen"
        print(f"  {alias:<20} → {target:<40} [{status}]")

    print("\n\nTo activate frozen models:")
    print("  from xpipe.pricing import activate_model")
    print("  activate_model('claude-4.5')  # After setting ANTHROPIC_API_KEY")
