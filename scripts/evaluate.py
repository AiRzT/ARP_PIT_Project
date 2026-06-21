#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/apr_pit_matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apr_pit.model import APRPiT  # noqa: E402
from apr_pit.utils import choose_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate APR-PiT on center-plane grids.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[10.0, 60.0, 120.0])
    parser.add_argument("--nx", type=int, default=401)
    parser.add_argument("--nz", type=int, default=121)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def evaluate_grid(
    model: APRPiT,
    config: dict,
    time_seconds: float,
    nx: int,
    nz: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, float(config["domain"]["length_x"]), nx)
    z = np.linspace(0.0, float(config["domain"]["height_z"]), nz)
    xx, zz = np.meshgrid(x, z)
    normalized = np.stack(
        (
            np.full_like(xx, time_seconds / float(config["domain"]["duration"])),
            xx / float(config["domain"]["length_x"]),
            zz / float(config["domain"]["height_z"]),
        ),
        axis=-1,
    ).reshape(-1, 3)
    fields: dict[str, list[np.ndarray]] = {name: [] for name in model.field_names}
    model.eval()
    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            coordinates = torch.from_numpy(normalized[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            prediction = model(coordinates)
            for name in fields:
                fields[name].append(prediction[name].cpu().numpy())
    result = {name: np.concatenate(values).reshape(nz, nx) for name, values in fields.items()}
    result.update({"x": x, "z": z})
    return result


def save_figure(result: dict[str, np.ndarray], time_seconds: float, path: Path) -> None:
    x, z = result["x"], result["z"]
    speed = np.sqrt(result["u"] ** 2 + result["w"] ** 2)
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), constrained_layout=True, sharex=True)
    panels = (
        (speed, "viridis", r"Velocity magnitude (m s$^{-1}$)"),
        (result["temperature"], "inferno", "Temperature (K)"),
        (result["smoke"], "turbo", "Smoke mass fraction (-)"),
    )
    for axis, (field, cmap, label) in zip(axes, panels, strict=True):
        image = axis.pcolormesh(x, z, field, shading="auto", cmap=cmap)
        axis.set_ylabel("Height z (m)")
        colorbar = figure.colorbar(image, ax=axis, pad=0.015)
        colorbar.set_label(label)
    axes[-1].set_xlabel("Longitudinal position x (m)")
    figure.suptitle(f"APR-PiT center-plane prediction at t = {time_seconds:g} s")
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = APRPiT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    output = args.output or args.checkpoint.parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for time_seconds in args.times:
        result = evaluate_grid(
            model, config, time_seconds, args.nx, args.nz, args.batch_size, device
        )
        stem = f"fields_t{time_seconds:g}s"
        np.savez_compressed(output / f"{stem}.npz", **result)
        save_figure(result, time_seconds, output / f"{stem}.png")
        print(output / f"{stem}.png")


if __name__ == "__main__":
    main()
