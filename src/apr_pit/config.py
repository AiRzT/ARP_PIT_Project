from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and perform lightweight consistency checks."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    required = {"seed", "domain", "scales", "physics", "source", "model", "sampling", "training"}
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Missing configuration sections: {sorted(missing)}")
    if config["model"]["d_model"] % config["model"]["attention_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.attention_heads")
    if config["model"]["correction_heads"] > config["model"]["attention_heads"]:
        raise ValueError("correction_heads cannot exceed attention_heads")
    spatial_dimensions = int(
        config["model"].get("spatial_dimensions", int(config["model"]["input_dim"]) - 1)
    )
    expected = {2: (3, 5), 3: (4, 6)}
    if spatial_dimensions not in expected:
        raise ValueError("model.spatial_dimensions must be 2 or 3")
    expected_input, expected_output = expected[spatial_dimensions]
    if int(config["model"]["input_dim"]) != expected_input:
        raise ValueError(
            f"model.input_dim must be {expected_input} for {spatial_dimensions}D physics"
        )
    if int(config["model"]["output_dim"]) != expected_output:
        raise ValueError(
            f"model.output_dim must be {expected_output} for {spatial_dimensions}D physics"
        )
    return config


def smoke_test_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny configuration that exercises the complete algorithm on CPU/MPS."""
    cfg = copy.deepcopy(config)
    cfg["seed"] = 7
    cfg["model"].update(
        {
            "transformer_layers": 2,
            "d_model": 64,
            "attention_heads": 4,
            "correction_heads": 1,
            "ffn_width": 128,
            "mlp_layers": 2,
            "mlp_width": 64,
            "fourier_frequencies": 8,
            "local_window": 8,
            "global_anchors": 4,
            "sparse_switch": 64,
            "modulation_layers": 1,
            "modulation_width": 32,
        }
    )
    cfg["sampling"].update(
        {
            "initial_interior": 96,
            "initial_boundary": 48,
            "initial_initial": 48,
            "max_points": 160,
            "adapt_interval": 1,
            "evaluation_points": 48,
        }
    )
    cfg["training"].update(
        {
            "adam_epochs": 2,
            "pde_batch": 12,
            "bc_batch": 8,
            "ic_batch": 8,
            "log_interval": 1,
            "checkpoint_interval": 2,
        }
    )
    cfg["training"]["lbfgs"].update({"iterations": 1, "history_size": 5, "inner_iterations": 1})
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
