from __future__ import annotations

import math
from typing import Any

import torch


def _gradient(field: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        field,
        coordinates,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]


def _mean_square(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.new_zeros(())
    return torch.mean(value.square())


class TunnelFirePhysics:
    """Low-Mach tunnel-fire equations in two or three spatial dimensions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.domain = config["domain"]
        self.scales = config["scales"]
        self.constants = config["physics"]
        self.source_config = config["source"]
        self.spatial_dimensions = int(
            config["model"].get("spatial_dimensions", int(config["model"]["input_dim"]) - 1)
        )
        if self.spatial_dimensions not in (2, 3):
            raise ValueError("Only two- and three-dimensional physics are supported")
        self._source_normalizer = self._gaussian_domain_probability()

    @property
    def spatial_lengths(self) -> list[float]:
        lengths = [float(self.domain["length_x"])]
        if self.spatial_dimensions == 3:
            lengths.append(float(self.domain["width_y"]))
        lengths.append(float(self.domain["height_z"]))
        return lengths

    @property
    def velocity_names(self) -> tuple[str, ...]:
        return ("u", "w") if self.spatial_dimensions == 2 else ("u", "v", "w")

    def _gaussian_domain_probability(self) -> float:
        length = float(self.domain["length_x"])
        width = float(self.domain["width_y"])
        height = float(self.domain["height_z"])
        center = [float(v) for v in self.source_config["center"]]
        sigma = [float(v) for v in self.source_config["sigma"]]
        bounds = ((0.0, length), (-width / 2.0, width / 2.0), (0.0, height))
        probability = 1.0
        for (lower, upper), mean, std in zip(bounds, center, sigma, strict=True):
            denominator = std * math.sqrt(2.0)
            probability *= 0.5 * (
                math.erf((upper - mean) / denominator)
                - math.erf((lower - mean) / denominator)
            )
        return max(probability, 1.0e-12)

    def physical_coordinates(self, normalized: torch.Tensor) -> tuple[torch.Tensor, ...]:
        time = normalized[:, 0:1] * float(self.domain["duration"])
        x = normalized[:, 1:2] * float(self.domain["length_x"])
        if self.spatial_dimensions == 2:
            z = normalized[:, 2:3] * float(self.domain["height_z"])
            return time, x, z
        y = (normalized[:, 2:3] - 0.5) * float(self.domain["width_y"])
        z = normalized[:, 3:4] * float(self.domain["height_z"])
        return time, x, y, z

    def heat_source(self, normalized: torch.Tensor) -> torch.Tensor:
        physical = self.physical_coordinates(normalized)
        if self.spatial_dimensions == 2:
            _, x, z = physical
            y = torch.zeros_like(x)
        else:
            _, x, y, z = physical
        center = normalized.new_tensor(self.source_config["center"]).reshape(1, 3)
        sigma = normalized.new_tensor(self.source_config["sigma"]).reshape(1, 3)
        position = torch.cat((x, y, z), dim=1)
        exponent = -0.5 * torch.sum(((position - center) / sigma).square(), dim=1, keepdim=True)
        gaussian_scale = (2.0 * math.pi) ** 1.5 * torch.prod(sigma)
        total_hrr = float(self.source_config["total_hrr"])
        return total_hrr * torch.exp(exponent) / (self._source_normalizer * gaussian_scale)

    def residuals(
        self,
        prediction: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not coordinates.requires_grad:
            raise ValueError("coordinates must require gradients for PDE residual evaluation")

        duration = float(self.domain["duration"])
        lengths = self.spatial_lengths
        characteristic_length = float(self.domain["height_z"])
        velocity_ref = float(self.scales["velocity"])
        temperature_ref = float(self.scales["temperature"])
        smoke_ref = float(self.scales["smoke"])
        eps = float(self.constants["epsilon"])

        velocity = [prediction[name] for name in self.velocity_names]
        pressure = prediction["pressure"]
        temperature = prediction["temperature"]
        smoke = prediction["smoke"]
        temperature_safe = temperature.clamp(min=150.0, max=2500.0)
        rho = float(self.constants["p0"]) / (
            float(self.constants["gas_constant"]) * temperature_safe
        )

        def derivatives(field: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
            gradient = _gradient(field, coordinates)
            time_derivative = gradient[:, 0:1] / duration
            spatial = [
                gradient[:, axis + 1 : axis + 2] / lengths[axis]
                for axis in range(self.spatial_dimensions)
            ]
            return time_derivative, spatial

        velocity_derivatives = [derivatives(component) for component in velocity]
        _, pressure_gradient = derivatives(pressure)
        temperature_t, temperature_gradient = derivatives(temperature)
        rho_t, rho_gradient = derivatives(rho)

        divergence = sum(
            velocity_derivatives[axis][1][axis]
            for axis in range(self.spatial_dimensions)
        )
        strain: list[list[torch.Tensor]] = []
        for i in range(self.spatial_dimensions):
            row = []
            for j in range(self.spatial_dimensions):
                row.append(
                    0.5
                    * (
                        velocity_derivatives[i][1][j]
                        + velocity_derivatives[j][1][i]
                    )
                )
            strain.append(row)
        strain_square = sum(
            strain[i][j].square()
            for i in range(self.spatial_dimensions)
            for j in range(self.spatial_dimensions)
        )
        strain_magnitude = torch.sqrt(2.0 * strain_square + eps)
        mu_t = (
            rho
            * (float(self.constants["smagorinsky"]) * float(self.constants["filter_width"]))
            ** 2
            * strain_magnitude
        )
        mu_eff = float(self.constants["dynamic_viscosity"]) + mu_t

        stress: list[list[torch.Tensor]] = []
        for i in range(self.spatial_dimensions):
            row = []
            for j in range(self.spatial_dimensions):
                trace_correction = (2.0 / 3.0) * divergence if i == j else 0.0
                row.append(mu_eff * (2.0 * strain[i][j] - trace_correction))
            stress.append(row)

        momentum: list[torch.Tensor] = []
        for i in range(self.spatial_dimensions):
            convection = sum(
                velocity[j] * velocity_derivatives[i][1][j]
                for j in range(self.spatial_dimensions)
            )
            stress_divergence = sum(
                derivatives(stress[i][j])[1][j]
                for j in range(self.spatial_dimensions)
            )
            equation = (
                rho * (velocity_derivatives[i][0] + convection)
                + pressure_gradient[i]
                - stress_divergence
            )
            if i == self.spatial_dimensions - 1:
                equation = equation + (
                    rho - float(self.constants["ambient_density"])
                ) * float(self.constants["gravity"])
            momentum.append(equation)

        mass = rho_t + sum(
            velocity[axis] * rho_gradient[axis]
            + rho * velocity_derivatives[axis][1][axis]
            for axis in range(self.spatial_dimensions)
        )

        k_eff = float(self.constants["thermal_conductivity"]) + mu_t * float(
            self.constants["cp"]
        ) / float(self.constants["turbulent_prandtl"])
        heat_diffusion = sum(
            derivatives(k_eff * temperature_gradient[axis])[1][axis]
            for axis in range(self.spatial_dimensions)
        )
        source = self.heat_source(coordinates)
        radiation = float(self.constants["radiation_beta"]) * (
            temperature_safe.pow(4) - float(self.constants["ambient_temperature"]) ** 4
        )
        thermal_convection = sum(
            velocity[axis] * temperature_gradient[axis]
            for axis in range(self.spatial_dimensions)
        )
        energy = (
            rho
            * float(self.constants["cp"])
            * (temperature_t + thermal_convection)
            - heat_diffusion
            - float(self.constants["sensible_heat_fraction"]) * source
            + radiation
        )

        rho_smoke = rho * smoke
        smoke_mass_t, smoke_mass_gradient = derivatives(rho_smoke)
        _, smoke_gradient = derivatives(smoke)
        diffusivity = rho * float(self.constants["smoke_diffusivity"]) + mu_t / float(
            self.constants["turbulent_schmidt"]
        )
        smoke_diffusion = sum(
            derivatives(diffusivity * smoke_gradient[axis])[1][axis]
            for axis in range(self.spatial_dimensions)
        )
        smoke_transport = (
            smoke_mass_t
            + sum(
                velocity[axis] * smoke_mass_gradient[axis]
                + rho_smoke * velocity_derivatives[axis][1][axis]
                for axis in range(self.spatial_dimensions)
            )
            - smoke_diffusion
            - float(self.constants["energy_smoke_yield"]) * source
        )

        rho_ref = float(self.constants["ambient_density"])
        cp = float(self.constants["cp"])
        mass_scale = rho_ref / duration + rho_ref * velocity_ref / characteristic_length
        momentum_scale = (
            rho_ref * velocity_ref / duration
            + rho_ref * velocity_ref**2 / characteristic_length
        )
        energy_scale = (
            rho_ref * cp * temperature_ref / duration
            + rho_ref * cp * velocity_ref * temperature_ref / characteristic_length
        )
        smoke_scale = (
            rho_ref * smoke_ref / duration
            + rho_ref * velocity_ref * smoke_ref / characteristic_length
        )

        mass_n = mass / mass_scale
        momentum_n = [component / momentum_scale for component in momentum]
        energy_n = energy / energy_scale
        smoke_n = smoke_transport / smoke_scale
        residual_vector = torch.cat((mass_n, *momentum_n, energy_n, smoke_n), dim=1)
        scalar = torch.sqrt(torch.mean(residual_vector.square(), dim=1, keepdim=True) + eps)

        material_temperature = temperature_t + thermal_convection
        low_mach_divergence = divergence - material_temperature / temperature_safe
        acceleration_scale = velocity_ref / duration + velocity_ref**2 / characteristic_length
        buoyancy_indicator = torch.abs(
            float(self.constants["buoyancy_beta"])
            * (temperature - float(self.constants["ambient_temperature"]))
            * float(self.constants["gravity"])
        ) / acceleration_scale
        momentum_indicator = torch.sqrt(
            sum(component.square() for component in momentum_n) + eps
        )
        feature = torch.cat(
            (
                low_mach_divergence / (velocity_ref / characteristic_length),
                momentum_indicator,
                buoyancy_indicator,
            ),
            dim=1,
        )

        return {
            "vector": residual_vector,
            "scalar": scalar,
            "physics_features": torch.tanh(feature),
            "density": rho,
            "eddy_viscosity": mu_t,
            "source": source,
        }

    def pde_loss(self, residual: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(residual["vector"].square())

    def initial_loss(self, prediction: dict[str, torch.Tensor]) -> torch.Tensor:
        velocity_scale = float(self.scales["velocity"])
        pressure_scale = float(self.constants["ambient_density"]) * velocity_scale**2
        temperature_scale = float(self.scales["temperature"])
        smoke_scale = float(self.scales["smoke"])
        velocity_loss = sum(
            _mean_square(prediction[name] / velocity_scale) for name in self.velocity_names
        )
        return (
            velocity_loss
            + _mean_square(prediction["pressure"] / pressure_scale)
            + _mean_square(
                (prediction["temperature"] - float(self.constants["ambient_temperature"]))
                / temperature_scale
            )
            + _mean_square(prediction["smoke"] / smoke_scale)
        )

    def boundary_loss(
        self,
        prediction: dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        boundary_type: torch.Tensor,
    ) -> torch.Tensor:
        lengths = self.spatial_lengths
        velocity_scale = float(self.scales["velocity"])
        temperature_scale = float(self.scales["temperature"])
        smoke_scale = float(self.scales["smoke"])
        pressure_scale = float(self.constants["ambient_density"]) * velocity_scale**2
        ambient = float(self.constants["ambient_temperature"])

        velocity = [prediction[name] for name in self.velocity_names]
        pressure = prediction["pressure"]
        temperature = prediction["temperature"]
        smoke = prediction["smoke"]
        velocity_gradients = [_gradient(component, coordinates) for component in velocity]
        temperature_gradient = _gradient(temperature, coordinates)
        smoke_gradient = _gradient(smoke, coordinates)

        inlet = boundary_type == 0
        outlet = boundary_type == 1
        wall = boundary_type >= 2
        inlet_loss = sum(
            _mean_square(component[inlet] / velocity_scale) for component in velocity
        )
        inlet_loss = (
            inlet_loss
            + _mean_square((temperature[inlet] - ambient) / temperature_scale)
            + _mean_square(smoke[inlet] / smoke_scale)
        )
        outlet_loss = _mean_square(pressure[outlet] / pressure_scale)
        outlet_loss = outlet_loss + sum(
            _mean_square(gradient[outlet, 1:2] / velocity_scale)
            for gradient in velocity_gradients
        )
        outlet_loss = (
            outlet_loss
            + _mean_square(temperature_gradient[outlet, 1:2] / temperature_scale)
            + _mean_square(smoke_gradient[outlet, 1:2] / smoke_scale)
        )

        temperature_normal = torch.zeros_like(temperature)
        smoke_normal = torch.zeros_like(smoke)
        if self.spatial_dimensions == 2:
            normal_specs = ((2, 2, -1.0, lengths[1]), (3, 2, 1.0, lengths[1]))
        else:
            normal_specs = (
                (2, 2, -1.0, lengths[1]),
                (3, 2, 1.0, lengths[1]),
                (4, 3, -1.0, lengths[2]),
                (5, 3, 1.0, lengths[2]),
            )
        for tag, coordinate_axis, sign, length in normal_specs:
            mask = boundary_type == tag
            temperature_normal[mask] = (
                sign * temperature_gradient[mask, coordinate_axis : coordinate_axis + 1] / length
            )
            smoke_normal[mask] = (
                sign * smoke_gradient[mask, coordinate_axis : coordinate_axis + 1] / length
            )
        wall_flux = (
            -float(self.constants["thermal_conductivity"]) * temperature_normal
            - float(self.constants["wall_heat_transfer"]) * (temperature - ambient)
        )
        shortest_wall_scale = min(lengths[-2:]) if self.spatial_dimensions == 3 else lengths[-1]
        wall_flux_scale = (
            float(self.constants["thermal_conductivity"])
            * temperature_scale
            / shortest_wall_scale
            + float(self.constants["wall_heat_transfer"]) * temperature_scale
        )
        wall_loss = sum(
            _mean_square(component[wall] / velocity_scale) for component in velocity
        )
        wall_loss = (
            wall_loss
            + _mean_square(wall_flux[wall] / wall_flux_scale)
            + _mean_square(smoke_normal[wall] / (smoke_scale / shortest_wall_scale))
        )
        return inlet_loss + outlet_loss + wall_loss
