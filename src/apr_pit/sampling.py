from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def latin_hypercube(n: int, dimensions: int, generator: torch.Generator) -> torch.Tensor:
    """Latin-hypercube samples in the normalized unit cube."""
    if n <= 0:
        return torch.empty((0, dimensions), dtype=torch.float32)
    result = torch.empty((n, dimensions), dtype=torch.float32)
    for dimension in range(dimensions):
        permutation = torch.randperm(n, generator=generator)
        jitter = torch.rand(n, generator=generator)
        result[:, dimension] = (permutation.to(torch.float32) + jitter) / n
    return result


def coordinate_dimensions(config: dict[str, Any]) -> int:
    return int(config["model"]["input_dim"])


def sample_interior(n: int, generator: torch.Generator, dimensions: int = 3) -> torch.Tensor:
    return latin_hypercube(n, dimensions, generator)


def sample_initial(n: int, generator: torch.Generator, dimensions: int = 3) -> torch.Tensor:
    points = latin_hypercube(n, dimensions, generator)
    points[:, 0] = 0.0
    return points


def sample_boundary(
    n: int, generator: torch.Generator, dimensions: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample all physical boundaries in a 2D or 3D tunnel.

    Tags 0 and 1 denote the streamwise inlet and outlet. In 2D, tags 2 and 3
    denote floor and ceiling. In 3D, tags 2--5 denote y-min, y-max, floor,
    and ceiling.
    """
    if dimensions not in (3, 4):
        raise ValueError("coordinate dimensions must be 3 (2D) or 4 (3D)")
    boundary_count = 4 if dimensions == 3 else 6
    counts = [n // boundary_count] * boundary_count
    for index in range(n % boundary_count):
        counts[index] += 1
    point_blocks: list[torch.Tensor] = []
    tag_blocks: list[torch.Tensor] = []
    for tag, count in enumerate(counts):
        points = latin_hypercube(count, dimensions, generator)
        if tag == 0:
            points[:, 1] = 0.0
        elif tag == 1:
            points[:, 1] = 1.0
        elif dimensions == 3 and tag == 2:
            points[:, -1] = 0.0
        elif dimensions == 3:
            points[:, -1] = 1.0
        elif tag == 2:
            points[:, 2] = 0.0
        elif tag == 3:
            points[:, 2] = 1.0
        elif tag == 4:
            points[:, 3] = 0.0
        else:
            points[:, 3] = 1.0
        point_blocks.append(points)
        tag_blocks.append(torch.full((count,), tag, dtype=torch.long))
    return torch.cat(point_blocks), torch.cat(tag_blocks)


@dataclass
class CollocationPool:
    points: torch.Tensor
    residual_scores: torch.Tensor
    physics_features: torch.Tensor

    @classmethod
    def initialize(
        cls, n: int, generator: torch.Generator, dimensions: int = 3
    ) -> "CollocationPool":
        points = sample_interior(n, generator, dimensions)
        return cls(
            points=points,
            residual_scores=torch.zeros((n, 1), dtype=torch.float32),
            physics_features=torch.zeros((n, 3), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self.points.shape[0]

    def normalized_scores(self, epsilon: float = 1.0e-8) -> torch.Tensor:
        maximum = self.residual_scores.max().clamp_min(epsilon)
        return (self.residual_scores / maximum).clamp(0.0, 1.0)

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, len(self))
        index = torch.randint(len(self), (batch_size,), generator=generator)
        return (
            index,
            self.points[index],
            self.normalized_scores()[index],
            self.physics_features[index],
        )

    def replace_scores(self, scores: torch.Tensor, features: torch.Tensor) -> None:
        if scores.shape != self.residual_scores.shape:
            raise ValueError(f"Score shape mismatch: {scores.shape} != {self.residual_scores.shape}")
        if features.shape != self.physics_features.shape:
            raise ValueError(f"Feature shape mismatch: {features.shape} != {self.physics_features.shape}")
        self.residual_scores = scores.detach().cpu().to(torch.float32)
        self.physics_features = features.detach().cpu().to(torch.float32)


class APRController:
    """Residual-driven point refinement and pruning with a hard point budget."""

    def __init__(self, config: dict[str, Any]) -> None:
        sampling = config["sampling"]
        self.max_points = int(sampling["max_points"])
        self.top_fraction = float(sampling["top_fraction"])
        self.gamma = float(sampling["sampling_exponent"])
        self.prune_threshold = float(sampling["prune_threshold"])
        self.alpha = float(sampling["growth_gain"])
        self.tau = float(sampling["saturation"])
        self.jitter_std = float(sampling.get("jitter_std", 0.01))
        self.epsilon = float(config["physics"]["epsilon"])

    def refine(self, pool: CollocationPool, generator: torch.Generator) -> dict[str, int | float]:
        if len(pool) == 0:
            raise ValueError("Cannot refine an empty collocation pool")
        normalized = pool.normalized_scores(self.epsilon).reshape(-1)
        keep_mask = normalized >= self.prune_threshold
        minimum_keep = max(1, int(math.ceil(self.top_fraction * len(pool))))
        if int(keep_mask.sum()) < minimum_keep:
            top = torch.topk(normalized, k=minimum_keep).indices
            keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            keep_mask[top] = True

        kept_points = pool.points[keep_mask]
        kept_scores = pool.residual_scores[keep_mask]
        kept_features = pool.physics_features[keep_mask]
        pruned = len(pool) - kept_points.shape[0]

        mean_residual = float(normalized.mean())
        proposed = int(math.floor(self.alpha * len(pool) * mean_residual / (mean_residual + self.tau)))
        capacity = max(0, self.max_points - kept_points.shape[0])
        added = min(proposed, capacity)

        if added > 0:
            top_count = max(1, int(math.ceil(self.top_fraction * len(pool))))
            top_index = torch.topk(normalized, k=top_count).indices
            probability = normalized[top_index].clamp_min(self.epsilon).pow(self.gamma)
            probability = probability / probability.sum()
            parent_local = torch.multinomial(probability, added, replacement=True, generator=generator)
            parent_index = top_index[parent_local]
            noise = torch.randn(
                (added, pool.points.shape[1]), generator=generator
            ) * self.jitter_std
            new_points = (pool.points[parent_index] + noise).clamp(0.0, 1.0)
            new_scores = pool.residual_scores[parent_index]
            new_features = pool.physics_features[parent_index]
            kept_points = torch.cat((kept_points, new_points), dim=0)
            kept_scores = torch.cat((kept_scores, new_scores), dim=0)
            kept_features = torch.cat((kept_features, new_features), dim=0)

        pool.points = kept_points[: self.max_points]
        pool.residual_scores = kept_scores[: self.max_points]
        pool.physics_features = kept_features[: self.max_points]
        return {
            "points_before": int(len(pool) + pruned - added),
            "points_after": len(pool),
            "pruned": int(pruned),
            "added": int(added),
            "mean_normalized_residual": mean_residual,
        }
