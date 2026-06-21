from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import save_config
from .model import APRPiT
from .physics import TunnelFirePhysics
from .sampling import APRController, CollocationPool, sample_boundary, sample_initial, sample_interior
from .utils import append_jsonl, trainable_parameter_count


class APRPiTTrainer:
    def __init__(self, config: dict[str, Any], device: torch.device, output_dir: str | Path) -> None:
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.output_dir / "resolved_config.yaml")

        self.generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
        self.model = APRPiT(config).to(device)
        self.physics = TunnelFirePhysics(config)
        self.apr = APRController(config)
        sampling = config["sampling"]
        dimensions = int(config["model"]["input_dim"])
        self.pool = CollocationPool.initialize(
            int(sampling["initial_interior"]), self.generator, dimensions
        )
        self.boundary_points, self.boundary_types = sample_boundary(
            int(sampling["initial_boundary"]), self.generator, dimensions
        )
        boundary_order = torch.randperm(self.boundary_points.shape[0], generator=self.generator)
        self.boundary_points = self.boundary_points[boundary_order]
        self.boundary_types = self.boundary_types[boundary_order]
        self.initial_points = sample_initial(
            int(sampling["initial_initial"]), self.generator, dimensions
        )
        self.evaluation_points = sample_interior(
            int(sampling["evaluation_points"]), self.generator, dimensions
        )

        training = config["training"]
        betas = tuple(float(value) for value in training["adam_betas"])
        self.adam = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training["learning_rate"]),
            betas=betas,
            eps=float(training["adam_epsilon"]),
        )
        interval = int(training["lr_decay_interval"])
        decay = float(training["lr_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.adam, lr_lambda=lambda step: decay ** (step // max(interval, 1))
        )
        self.history_path = self.output_dir / "history.jsonl"
        self.start_time = time.perf_counter()
        self.previous_fixed_loss: float | None = None
        self.convergence_streak = 0

    def _random_subset(
        self,
        points: torch.Tensor,
        batch_size: int,
        tags: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = min(batch_size, points.shape[0])
        index = torch.randint(points.shape[0], (batch_size,), generator=self.generator)
        selected_tags = tags[index] if tags is not None else None
        return points[index], selected_tags

    def _draw_batches(self) -> dict[str, torch.Tensor]:
        training = self.config["training"]
        index, pde_points, residual_hint, physics_features = self.pool.sample(
            int(training["pde_batch"]), self.generator
        )
        bc_points, bc_types = self._random_subset(
            self.boundary_points, int(training["bc_batch"]), self.boundary_types
        )
        ic_points, _ = self._random_subset(self.initial_points, int(training["ic_batch"]), None)
        assert bc_types is not None
        return {
            "pool_index": index,
            "pde_points": pde_points,
            "residual_hint": residual_hint,
            "physics_features": physics_features,
            "bc_points": bc_points,
            "bc_types": bc_types,
            "ic_points": ic_points,
        }

    def _losses(self, batch: dict[str, torch.Tensor], update_pool: bool) -> dict[str, torch.Tensor]:
        pde_coordinates = batch["pde_points"].to(self.device).requires_grad_(True)
        residual_hint = batch["residual_hint"].to(self.device)
        cached_features = batch["physics_features"].to(self.device)
        pde_prediction = self.model(pde_coordinates, residual_hint, cached_features)
        residual = self.physics.residuals(pde_prediction, pde_coordinates)
        pde_loss = self.physics.pde_loss(residual)
        latent_loss = torch.mean(
            (pde_prediction["auxiliary_physics"] - residual["physics_features"].detach()).square()
        )

        if update_pool:
            index = batch["pool_index"]
            self.pool.residual_scores[index] = residual["scalar"].detach().cpu()
            self.pool.physics_features[index] = residual["physics_features"].detach().cpu()

        bc_coordinates = batch["bc_points"].to(self.device).requires_grad_(True)
        bc_prediction = self.model(bc_coordinates)
        bc_loss = self.physics.boundary_loss(
            bc_prediction, bc_coordinates, batch["bc_types"].to(self.device)
        )

        ic_coordinates = batch["ic_points"].to(self.device).requires_grad_(True)
        ic_prediction = self.model(ic_coordinates)
        ic_loss = self.physics.initial_loss(ic_prediction)

        training = self.config["training"]
        total = (
            float(training["lambda_pde"]) * pde_loss
            + float(training["lambda_bc"]) * bc_loss
            + float(training["lambda_ic"]) * ic_loss
            + float(training["lambda_latent"]) * latent_loss
        )
        return {
            "total": total,
            "pde": pde_loss,
            "bc": bc_loss,
            "ic": ic_loss,
            "latent": latent_loss,
            "mean_residual": residual["scalar"].mean(),
        }

    def _log(self, stage: str, step: int, losses: dict[str, torch.Tensor], **extra: Any) -> None:
        record: dict[str, Any] = {
            "stage": stage,
            "step": int(step),
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "pool_size": len(self.pool),
        }
        for name, value in losses.items():
            record[name] = float(value.detach().cpu())
        record.update(extra)
        append_jsonl(self.history_path, record)
        print(
            f"[{stage} {step:>6}] total={record.get('total', float('nan')):.4e} "
            f"pde={record.get('pde', float('nan')):.4e} pool={len(self.pool)}"
        )

    def score_points(self, points: torch.Tensor, chunk_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        if chunk_size is None:
            chunk_size = min(int(self.config["training"]["pde_batch"]), 4096)
        scores: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start : start + chunk_size].to(self.device).requires_grad_(True)
            prediction = self.model(chunk)
            residual = self.physics.residuals(prediction, chunk)
            scores.append(residual["scalar"].detach().cpu())
            features.append(residual["physics_features"].detach().cpu())
            del prediction, residual, chunk
        self.model.train()
        return torch.cat(scores), torch.cat(features)

    def adapt(self, epoch: int) -> dict[str, int | float]:
        scores, features = self.score_points(self.pool.points)
        self.pool.replace_scores(scores, features)
        statistics = self.apr.refine(self.pool, self.generator)
        append_jsonl(
            self.history_path,
            {"stage": "apr", "step": epoch, **statistics, "elapsed_seconds": time.perf_counter() - self.start_time},
        )
        print(
            f"[apr {epoch:>7}] {statistics['points_before']} -> {statistics['points_after']} "
            f"(pruned={statistics['pruned']}, added={statistics['added']})"
        )
        return statistics

    def evaluate_convergence(self, epoch: int) -> bool:
        """Evaluate the manuscript stopping criteria on fixed point sets."""
        self.model.eval()
        chunk_size = min(int(self.config["training"]["pde_batch"]), 4096)
        pde_sum = 0.0
        latent_sum = 0.0
        residual_sum = 0.0
        interior_count = 0
        for start in range(0, self.evaluation_points.shape[0], chunk_size):
            coordinates = self.evaluation_points[start : start + chunk_size].to(
                self.device
            ).requires_grad_(True)
            prediction = self.model(coordinates)
            residual = self.physics.residuals(prediction, coordinates)
            count = coordinates.shape[0]
            pde_sum += float(residual["vector"].square().mean().detach().cpu()) * count
            latent_sum += float(
                (prediction["auxiliary_physics"] - residual["physics_features"].detach())
                .square()
                .mean()
                .detach()
                .cpu()
            ) * count
            residual_sum += float(residual["scalar"].mean().detach().cpu()) * count
            interior_count += count
            del coordinates, prediction, residual

        bc_sum = 0.0
        for start in range(0, self.boundary_points.shape[0], chunk_size):
            coordinates = self.boundary_points[start : start + chunk_size].to(
                self.device
            ).requires_grad_(True)
            tags = self.boundary_types[start : start + chunk_size].to(self.device)
            prediction = self.model(coordinates)
            count = coordinates.shape[0]
            bc_sum += float(
                self.physics.boundary_loss(prediction, coordinates, tags).detach().cpu()
            ) * count
            del coordinates, tags, prediction

        ic_sum = 0.0
        for start in range(0, self.initial_points.shape[0], chunk_size):
            coordinates = self.initial_points[start : start + chunk_size].to(self.device)
            prediction = self.model(coordinates)
            count = coordinates.shape[0]
            ic_sum += float(self.physics.initial_loss(prediction).detach().cpu()) * count
            del coordinates, prediction

        pde = pde_sum / max(interior_count, 1)
        latent = latent_sum / max(interior_count, 1)
        bc = bc_sum / max(self.boundary_points.shape[0], 1)
        ic = ic_sum / max(self.initial_points.shape[0], 1)
        training = self.config["training"]
        fixed_loss = (
            float(training["lambda_pde"]) * pde
            + float(training["lambda_bc"]) * bc
            + float(training["lambda_ic"]) * ic
            + float(training["lambda_latent"]) * latent
        )
        mean_residual = residual_sum / max(interior_count, 1)
        if self.previous_fixed_loss is None:
            relative_change: float | None = None
        else:
            relative_change = abs(fixed_loss - self.previous_fixed_loss) / (
                abs(self.previous_fixed_loss) + float(self.config["physics"]["epsilon"])
            )
        self.previous_fixed_loss = fixed_loss

        passed = (
            relative_change is not None
            and relative_change < float(training["convergence_loss"])
            and mean_residual < float(training["convergence_residual"])
        )
        self.convergence_streak = self.convergence_streak + 1 if passed else 0
        required = int(training["convergence_cycles"])
        append_jsonl(
            self.history_path,
            {
                "stage": "convergence",
                "step": epoch,
                "fixed_loss": fixed_loss,
                "relative_loss_change": relative_change,
                "fixed_mean_residual": mean_residual,
                "convergence_streak": self.convergence_streak,
                "required_streak": required,
                "elapsed_seconds": time.perf_counter() - self.start_time,
            },
        )
        print(
            f"[eval {epoch:>6}] fixed_loss={fixed_loss:.4e} "
            f"residual={mean_residual:.4e} streak={self.convergence_streak}/{required}"
        )
        self.model.train()
        return self.convergence_streak >= required

    def save_checkpoint(self, name: str, step: int) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "step": step,
                "config": self.config,
                "model": self.model.state_dict(),
                "adam": self.adam.state_dict(),
                "pool": {
                    "points": self.pool.points,
                    "residual_scores": self.pool.residual_scores,
                    "physics_features": self.pool.physics_features,
                },
            },
            path,
        )
        return path

    def train_adam(self) -> None:
        training = self.config["training"]
        epochs = int(training["adam_epochs"])
        log_interval = int(training["log_interval"])
        checkpoint_interval = int(training["checkpoint_interval"])
        adapt_interval = int(self.config["sampling"]["adapt_interval"])
        self.model.train()
        converged = False
        for epoch in range(1, epochs + 1):
            batch = self._draw_batches()
            self.adam.zero_grad(set_to_none=True)
            losses = self._losses(batch, update_pool=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), float(training["gradient_clip"]))
            self.adam.step()
            self.scheduler.step()

            if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
                self._log(
                    "adam",
                    epoch,
                    losses,
                    learning_rate=self.adam.param_groups[0]["lr"],
                )
            if adapt_interval > 0 and epoch % adapt_interval == 0:
                self.adapt(epoch)
                converged = self.evaluate_convergence(epoch)
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_adam_{epoch:06d}.pt", epoch)
            if converged:
                print(f"Fixed-set convergence criteria reached at Adam epoch {epoch}.")
                break
        self.save_checkpoint("checkpoint_adam_final.pt", epoch)

    def train_lbfgs(self) -> None:
        config = self.config["training"]["lbfgs"]
        iterations = int(config["iterations"])
        if iterations <= 0:
            return
        inner = max(1, int(config.get("inner_iterations", 20)))
        outer_steps = int(math.ceil(iterations / inner))
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=inner,
            history_size=int(config["history_size"]),
            tolerance_grad=float(config["tolerance_grad"]),
            tolerance_change=float(config["tolerance_change"]),
            line_search_fn="strong_wolfe",
        )
        fixed_batch = self._draw_batches()
        latest: dict[str, torch.Tensor] = {}

        for outer in range(1, outer_steps + 1):
            def closure() -> torch.Tensor:
                nonlocal latest
                optimizer.zero_grad(set_to_none=True)
                latest = self._losses(fixed_batch, update_pool=False)
                latest["total"].backward()
                return latest["total"]

            optimizer.step(closure)
            completed = min(outer * inner, iterations)
            self._log("lbfgs", completed, latest)
        self.save_checkpoint("checkpoint_final.pt", int(self.config["training"]["adam_epochs"]) + iterations)

    def train(self) -> Path:
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {trainable_parameter_count(self.model):,}")
        self.train_adam()
        self.train_lbfgs()
        return self.output_dir / "checkpoint_final.pt"
