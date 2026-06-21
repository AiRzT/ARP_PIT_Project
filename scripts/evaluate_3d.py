#!/usr/bin/env python3
"""Evaluate a trained 3D APR-PiT checkpoint on a structured tunnel grid.

The output is HDF5 rather than CSV because a single 1601 x 81 x 49 field
contains 6,354,369 nodes. HRRPUV is deliberately not labelled as a network
prediction: the file stores the prescribed analytical heat-source field used
by the PDE residual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apr_pit.model import APRPiT  # noqa: E402
from apr_pit.physics import TunnelFirePhysics  # noqa: E402
from apr_pit.utils import choose_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fully 3D APR-PiT checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[10.0, 40.0, 80.0, 120.0])
    parser.add_argument("--nx", type=int, default=1601)
    parser.add_argument("--ny", type=int, default=81)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if int(config["model"].get("spatial_dimensions", 2)) != 3:
        raise ValueError("evaluate_3d.py requires a checkpoint trained with spatial_dimensions: 3")
    model = APRPiT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    physics = TunnelFirePhysics(config)

    output = args.output or args.checkpoint.parent / "evaluation_3d.h5"
    output.parent.mkdir(parents=True, exist_ok=True)
    length_x = float(config["domain"]["length_x"])
    width_y = float(config["domain"]["width_y"])
    height_z = float(config["domain"]["height_z"])
    duration = float(config["domain"]["duration"])
    gas_constant = float(config["physics"]["gas_constant"])
    p0 = float(config["physics"]["p0"])
    x = np.linspace(0.0, length_x, args.nx, dtype=np.float32)
    y = np.linspace(-width_y / 2.0, width_y / 2.0, args.ny, dtype=np.float32)
    z = np.linspace(0.0, height_z, args.nz, dtype=np.float32)
    yz_count = args.ny * args.nz
    x_per_chunk = max(1, args.batch_size // yz_count)

    with h5py.File(output, "w") as h5:
        h5.attrs["model"] = "APR-PiT fully three-dimensional"
        h5.attrs["coordinate_order"] = "x,y,z"
        h5.attrs["hr_note"] = (
            "prescribed_heat_source_W_m3 is the PDE source, not a learned HRRPUV output"
        )
        coordinates_group = h5.create_group("coordinates")
        coordinates_group.create_dataset("x_m", data=x)
        coordinates_group.create_dataset("y_m", data=y)
        coordinates_group.create_dataset("z_m", data=z)
        fields_group = h5.create_group("fields")

        for time_seconds in args.times:
            token = f"t{time_seconds:g}s"
            group = fields_group.create_group(token)
            group.attrs["time_s"] = float(time_seconds)
            shape = (args.nx, args.ny, args.nz)
            datasets = {
                name: group.create_dataset(
                    name,
                    shape=shape,
                    dtype="f4",
                    chunks=(min(8, args.nx), min(16, args.ny), min(16, args.nz)),
                    compression="gzip",
                    compression_opts=4,
                )
                for name in (
                    "u_m_s",
                    "v_m_s",
                    "w_m_s",
                    "pressure_Pa",
                    "temperature_K",
                    "smoke_mass_fraction",
                    "soot_density_kg_m3",
                    "prescribed_heat_source_W_m3",
                )
            }
            for start in range(0, args.nx, x_per_chunk):
                stop = min(start + x_per_chunk, args.nx)
                xx, yy, zz = np.meshgrid(x[start:stop], y, z, indexing="ij")
                normalized = np.stack(
                    (
                        np.full_like(xx, time_seconds / duration),
                        xx / length_x,
                        yy / width_y + 0.5,
                        zz / height_z,
                    ),
                    axis=-1,
                ).reshape(-1, 4)
                output_parts: dict[str, list[np.ndarray]] = {
                    name: [] for name in model.field_names
                }
                source_parts: list[np.ndarray] = []
                with torch.no_grad():
                    for batch_start in range(0, normalized.shape[0], args.batch_size):
                        batch = torch.from_numpy(
                            normalized[batch_start : batch_start + args.batch_size]
                        ).to(device=device, dtype=torch.float32)
                        prediction = model(batch)
                        for name in model.field_names:
                            output_parts[name].append(prediction[name].cpu().numpy())
                        source_parts.append(physics.heat_source(batch).cpu().numpy())
                predicted = {
                    name: np.concatenate(parts).reshape(stop - start, args.ny, args.nz)
                    for name, parts in output_parts.items()
                }
                temperature_safe = np.clip(predicted["temperature"], 150.0, 2500.0)
                density = p0 / (gas_constant * temperature_safe)
                mapping = {
                    "u_m_s": predicted["u"],
                    "v_m_s": predicted["v"],
                    "w_m_s": predicted["w"],
                    "pressure_Pa": predicted["pressure"],
                    "temperature_K": predicted["temperature"],
                    "smoke_mass_fraction": predicted["smoke"],
                    "soot_density_kg_m3": density * predicted["smoke"],
                    "prescribed_heat_source_W_m3": np.concatenate(source_parts).reshape(
                        stop - start, args.ny, args.nz
                    ),
                }
                for name, values in mapping.items():
                    datasets[name][start:stop] = values.astype(np.float32, copy=False)
                print(f"{token}: x nodes {stop}/{args.nx}", flush=True)
    print(output)


if __name__ == "__main__":
    main()
