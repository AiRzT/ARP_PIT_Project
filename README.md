# APR-PiT

[中文使用说明](README_CN.md)

This repository provides a runnable PyTorch implementation of **APR-PiT**, a residual-adaptive physics-informed Transformer for center-plane reconstruction of buoyancy-driven heat and smoke transport in tunnel fires.

The code is a clean-room implementation based on the equations, model components, and parameter settings described in the associated manuscript. It is intended to reproduce the algorithmic workflow and manuscript-scale configuration, but it is not a claim of byte-for-byte identity with any historical training code.

## Repository status

This repository is prepared for research reproducibility and method demonstration. The implementation supports CPU smoke tests and GPU manuscript-scale training. Reference FDS fields are not included unless explicitly provided by the user, because large CFD output files are usually stored separately from the source code.

## Implemented method

- Pure physics-informed training; FDS snapshots are not used in the optimization loss.
- Dimensionless input coordinates `(t, x, z)` and physical outputs `(u, w, pi, T, C_s)`.
- Fourier positional encoding with four Transformer blocks.
- Physics-guided attention bias based on normalized PDE residuals and vertical alignment.
- Sparse local attention windows plus global high-residual anchors.
- Residual-conditioned feature modulation and auxiliary latent-physics loss.
- Quasi-2D low-Mach mass, momentum, temperature, and smoke transport equations.
- Smagorinsky eddy viscosity, Gaussian heat source, simplified radiation loss, and wall heat transfer.
- Adaptive point refinement and pruning under a hard 300,000-point collocation budget.
- Two-stage optimization using Adam followed by L-BFGS refinement.
- Fixed evaluation-set loss and residual tracking using the manuscript stopping criteria.
- Checkpoint saving, JSONL loss history, field export, error comparison, and journal-style field plotting.

The attention implementation uses a coordinate-diagonal automatic-differentiation path. Neighbor K/V context remains trainable, but its coordinate graph is stopped so that PINN derivatives do not accidentally sum cross-token Jacobian entries.

## Project layout

```text
APR-PiT_Project/
├── configs/tunnel_2d.yaml       # manuscript-scale configuration
├── scripts/train.py             # Adam -> APR -> L-BFGS training
├── scripts/evaluate.py          # grid inference and field figures
├── scripts/compare_reference.py # same-grid FDS/reference error metrics
├── scripts/plot_history.py      # loss-history figure
├── src/apr_pit/
│   ├── model.py                 # Fourier + sparse Transformer + decoder
│   ├── physics.py               # PDE, IC, and BC residuals
│   ├── sampling.py              # LHS pool and APR controller
│   ├── trainer.py               # optimization and checkpoints
│   ├── config.py
│   └── utils.py
└── tests/test_core.py
```

## Environment

A CUDA-enabled PyTorch installation is recommended for manuscript-scale training. CPU execution is sufficient for the smoke test and unit tests.

Create a dedicated Conda environment, for example:

```bash
conda create -n apr-pit python=3.10 -y
conda activate apr-pit
```

Install the required packages:

```bash
pip install torch numpy pyyaml matplotlib
```

For GPU training, install a PyTorch build that matches your CUDA version. See the official PyTorch installation instructions for the appropriate command for your workstation.

## Quick verification

From the repository root, run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run the complete algorithmic pipeline with tiny point counts and two Adam steps:

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/train.py \
  --smoke-test --device cpu --output outputs/tunnel_2d
```

The smoke-test checkpoint is written to:

```text
outputs/tunnel_2d_smoke/
```

## Manuscript-scale training

On a CUDA workstation, run:

```bash
python scripts/train.py \
  --config configs/tunnel_2d.yaml \
  --device cuda \
  --output outputs/tunnel_2d_full
```

The full configuration is set to match the manuscript-scale settings: 150,000 initial interior points, 25,000 boundary-condition points, 25,000 initial-condition points, a 300,000-point collocation cap, APR every 10,000 epochs, 30,000 Adam epochs, and up to 20,000 L-BFGS iterations.

## Field evaluation

Evaluate a trained checkpoint on a regular grid:

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/evaluate.py \
  outputs/tunnel_2d_full/checkpoint_final.pt \
  --times 10 60 120 --device cuda
```

For each requested time, the script exports a compressed `.npz` field file and a white-background PNG containing velocity magnitude, temperature, and smoke mass fraction.

Plot the training history:

```bash
MPLCONFIGDIR=/tmp/apr_pit_matplotlib \
python scripts/plot_history.py \
  outputs/tunnel_2d_full/history.jsonl
```

## Reference-field comparison

If same-grid FDS or experimental reference fields are available, compare the exported prediction with a reference `.npz` file:

```bash
python scripts/compare_reference.py \
  outputs/tunnel_2d_full/evaluation/fields_t120s.npz \
  data/fds_fields_t120s.npz \
  --fields temperature smoke
```

The reference `.npz` file should contain fields on the same grid as the exported prediction. Large FDS files are not included in this repository by default. Users should prepare same-grid reference files according to the expected field names used by `scripts/compare_reference.py`.

## Notes on reproducibility

This implementation is designed to reproduce the algorithmic workflow of APR-PiT, including physics-informed training, residual-adaptive collocation updates, sparse Transformer context encoding, and field-level evaluation. Exact numerical values may vary with hardware, PyTorch version, random seed, CUDA kernels, and optimizer settings.

## Citation

If you use this code, please cite the associated manuscript:

```text
Citation information will be added after publication.
```

## License

This project is released under the MIT License. See `LICENSE` for details.
