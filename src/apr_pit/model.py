from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class FourierEncoding(nn.Module):
    def __init__(self, input_dim: int, frequencies: int, sigma: float, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(frequencies, input_dim, generator=generator) * sigma
        self.register_buffer("projection", projection)

    @property
    def output_dim(self) -> int:
        return 2 * self.projection.shape[0]

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * coordinates @ self.projection.T
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current, hidden_dim), nn.Tanh()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class PhysicsGuidedSparseAttention(nn.Module):
    """Multi-head attention with local windows and high-residual global anchors.

    When ``diagonal_ad_context`` is enabled, keys and values receive a
    stop-gradient copy of the token tensor. Parameters in the K/V projections
    remain trainable, while each output token remains differentiable only with
    respect to its own coordinates. This makes the usual PINN derivative
    ``grad(output.sum(), coordinates)`` a diagonal spatial derivative rather
    than a sum of cross-token Jacobian entries.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.d_model = int(config["d_model"])
        self.heads = int(config["attention_heads"])
        self.correction_heads = int(config["correction_heads"])
        self.head_dim = self.d_model // self.heads
        self.local_window = int(config["local_window"])
        self.global_anchors = int(config["global_anchors"])
        self.sparse_switch = int(config["sparse_switch"])
        self.kappa_residual = float(config["kappa_residual"])
        self.kappa_buoyancy = float(config["kappa_buoyancy"])
        self.diagonal_ad_context = bool(config.get("diagonal_ad_context", True))

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(config.get("dropout", 0.0)))

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[0], self.heads, self.head_dim)

    def _neighbor_indices(self, coordinates: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        n = coordinates.shape[0]
        window = min(self.local_window, n)
        # A deterministic space-time key supplies a bounded local ordering in
        # either (t, x, z) or (t, x, y, z) coordinates.
        weights = coordinates.new_tensor(
            [1000.0 ** (coordinates.shape[1] - index - 1) for index in range(coordinates.shape[1])]
        )
        key = torch.sum(coordinates * weights, dim=1)
        order = torch.argsort(key)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n, device=coordinates.device)
        offsets = torch.arange(window, device=coordinates.device) - window // 2
        positions = (rank[:, None] + offsets[None, :]).clamp(0, n - 1)
        local = order[positions]

        anchors = min(self.global_anchors, n)
        if anchors == 0:
            return local
        global_index = torch.topk(residual, k=anchors, largest=True, sorted=False).indices
        global_index = global_index[None, :].expand(n, -1)
        return torch.cat((local, global_index), dim=1)

    def _physics_bias_dense(self, coordinates: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        residual_bias = self.kappa_residual * residual[:, None] * residual[None, :]
        spatial = coordinates[:, 1:]
        delta = spatial[None, :, :] - spatial[:, None, :]
        norm = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)
        vertical_alignment = torch.abs(delta[..., -1]) / norm
        return residual_bias + self.kappa_buoyancy * vertical_alignment

    def _physics_bias_sparse(
        self,
        coordinates: torch.Tensor,
        residual: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_residual = residual[neighbors]
        residual_bias = self.kappa_residual * residual[:, None] * neighbor_residual
        spatial = coordinates[:, 1:]
        delta = spatial[neighbors] - spatial[:, None, :]
        norm = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)
        vertical_alignment = torch.abs(delta[..., -1]) / norm
        return residual_bias + self.kappa_buoyancy * vertical_alignment

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        residual_hint: torch.Tensor,
    ) -> torch.Tensor:
        n = tokens.shape[0]
        residual = residual_hint.reshape(-1).detach().clamp(0.0, 1.0)
        detached_coordinates = coordinates.detach()
        kv_tokens = tokens.detach() if self.diagonal_ad_context else tokens
        q = self._reshape(self.q_proj(tokens))
        k = self._reshape(self.k_proj(kv_tokens))
        v = self._reshape(self.v_proj(kv_tokens))

        if n <= self.sparse_switch:
            logits = torch.einsum("nhd,mhd->hnm", q, k) / math.sqrt(self.head_dim)
            logits = logits + self._physics_bias_dense(detached_coordinates, residual).unsqueeze(0)
            weights = self.dropout(torch.softmax(logits, dim=-1))
            attended = torch.einsum("hnm,mhd->nhd", weights, v)
        else:
            neighbors = self._neighbor_indices(detached_coordinates, residual)
            neighbor_k = k[neighbors]
            neighbor_v = v[neighbors]
            logits = torch.einsum("nhd,nkhd->nhk", q, neighbor_k) / math.sqrt(self.head_dim)
            logits = logits + self._physics_bias_sparse(
                detached_coordinates, residual, neighbors
            ).unsqueeze(1)
            weights = self.dropout(torch.softmax(logits, dim=-1))
            attended = torch.einsum("nhk,nkhd->nhd", weights, neighbor_v)

        if self.correction_heads:
            gate = 1.0 + residual[:, None, None]
            attended = torch.cat(
                (
                    attended[:, : self.heads - self.correction_heads],
                    attended[:, self.heads - self.correction_heads :] * gate,
                ),
                dim=1,
            )
        return self.out_proj(attended.reshape(n, self.d_model))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.attention = PhysicsGuidedSparseAttention(config)
        self.norm_attention = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, int(config["ffn_width"])),
            nn.Tanh(),
            nn.Linear(int(config["ffn_width"]), d_model),
            nn.Dropout(float(config.get("dropout", 0.0))),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        residual_hint: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.norm_attention(tokens + self.attention(tokens, coordinates, residual_hint))
        return self.norm_ffn(tokens + self.feed_forward(tokens))


class APRPiT(nn.Module):
    """APR-PiT network for quasi-2D or fully three-dimensional tunnel flow."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        self.scales = config["scales"]
        self.physics = config["physics"]
        input_dim = int(model["input_dim"])
        output_dim = int(model["output_dim"])
        d_model = int(model["d_model"])
        self.spatial_dimensions = int(model.get("spatial_dimensions", input_dim - 1))
        if self.spatial_dimensions == 2:
            self.field_names = ("u", "w", "pressure", "temperature", "smoke")
            expected_output_dim = 5
        elif self.spatial_dimensions == 3:
            self.field_names = ("u", "v", "w", "pressure", "temperature", "smoke")
            expected_output_dim = 6
        else:
            raise ValueError("model.spatial_dimensions must be 2 or 3")
        if input_dim != self.spatial_dimensions + 1:
            raise ValueError("model.input_dim must equal spatial_dimensions + 1 for time")
        if output_dim != expected_output_dim:
            raise ValueError(
                f"model.output_dim must be {expected_output_dim} for "
                f"{self.spatial_dimensions} spatial dimensions"
            )

        self.fourier = FourierEncoding(
            input_dim,
            int(model["fourier_frequencies"]),
            float(model["fourier_sigma"]),
            int(config["seed"]),
        )
        self.point_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Linear(self.fourier.output_dim, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(model) for _ in range(int(model["transformer_layers"]))
        )

        modulation_input = input_dim + 3
        modulation_output = 2 * d_model
        self.modulation = _mlp(
            modulation_input,
            int(model["modulation_width"]),
            int(model["modulation_layers"]),
            modulation_output,
        )
        self.point_skip = nn.Linear(input_dim, d_model)
        self.decoder = _mlp(
            d_model,
            int(model["mlp_width"]),
            int(model["mlp_layers"]),
            output_dim,
        )
        self.auxiliary_decoder = nn.Linear(d_model, 3)

    def forward(
        self,
        coordinates: torch.Tensor,
        residual_hint: torch.Tensor | None = None,
        physics_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        n = coordinates.shape[0]
        if residual_hint is None:
            residual_hint = coordinates.new_zeros(n, 1)
        if physics_features is None:
            physics_features = coordinates.new_zeros(n, 3)

        tokens = self.point_embedding(coordinates) + self.position_embedding(self.fourier(coordinates))
        for block in self.blocks:
            tokens = block(tokens, coordinates, residual_hint)

        modulation = self.modulation(torch.cat((coordinates, physics_features), dim=-1))
        raw_gamma, beta = modulation.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(raw_gamma)
        context = gamma * tokens + beta
        features = context + self.point_skip(coordinates)
        raw = self.decoder(features)

        rho_ref = float(self.physics["ambient_density"])
        velocity_ref = float(self.scales["velocity"])
        temperature_ref = float(self.scales["temperature"])
        smoke_ref = float(self.scales["smoke"])
        ambient_temperature = float(self.physics["ambient_temperature"])

        if self.spatial_dimensions == 2:
            physical = {
                "u": raw[:, 0:1] * velocity_ref,
                "w": raw[:, 1:2] * velocity_ref,
                "pressure": raw[:, 2:3] * rho_ref * velocity_ref**2,
                "temperature": ambient_temperature + raw[:, 3:4] * temperature_ref,
                "smoke": raw[:, 4:5] * smoke_ref,
            }
        else:
            physical = {
                "u": raw[:, 0:1] * velocity_ref,
                "v": raw[:, 1:2] * velocity_ref,
                "w": raw[:, 2:3] * velocity_ref,
                "pressure": raw[:, 3:4] * rho_ref * velocity_ref**2,
                "temperature": ambient_temperature + raw[:, 4:5] * temperature_ref,
                "smoke": raw[:, 5:6] * smoke_ref,
            }
        physical.update(
            {
                "auxiliary_physics": self.auxiliary_decoder(context),
                "normalized_output": raw,
            }
        )
        return physical
