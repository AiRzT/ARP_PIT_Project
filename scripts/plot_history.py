#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/apr_pit_matplotlib")

import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot APR-PiT JSONL training history.")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records = []
    with args.history.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("stage") in {"adam", "lbfgs"}:
                records.append(record)
    if not records:
        raise RuntimeError(f"No optimizer records found in {args.history}")

    x = list(range(1, len(records) + 1))
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for key, label in (("total", "Total"), ("pde", "PDE"), ("bc", "BC"), ("ic", "IC")):
        axis.semilogy(x, [record[key] for record in records], label=label)
    axis.set_xlabel("Logged optimization step")
    axis.set_ylabel("Loss")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    output = args.output or args.history.with_name("training_history.png")
    figure.savefig(output, dpi=300, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()

