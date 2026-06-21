from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apr_pit.config import load_config, smoke_test_config  # noqa: E402
from apr_pit.model import APRPiT  # noqa: E402
from apr_pit.metrics import field_metrics  # noqa: E402
from apr_pit.physics import TunnelFirePhysics  # noqa: E402
from apr_pit.sampling import APRController, CollocationPool, sample_boundary  # noqa: E402


class APRPiTCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = smoke_test_config(load_config(PROJECT_ROOT / "configs" / "tunnel_2d.yaml"))

    def test_model_shapes_and_coordinate_gradients(self) -> None:
        model = APRPiT(self.config)
        coordinates = torch.rand(10, 3, requires_grad=True)
        prediction = model(coordinates, torch.rand(10, 1), torch.rand(10, 3))
        self.assertEqual(prediction["normalized_output"].shape, (10, 5))
        self.assertEqual(prediction["auxiliary_physics"].shape, (10, 3))
        gradient = torch.autograd.grad(
            prediction["temperature"].sum(), coordinates, create_graph=True
        )[0]
        self.assertEqual(gradient.shape, coordinates.shape)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_physics_residual_and_backward(self) -> None:
        model = APRPiT(self.config)
        physics = TunnelFirePhysics(self.config)
        coordinates = torch.rand(4, 3, requires_grad=True)
        prediction = model(coordinates)
        residual = physics.residuals(prediction, coordinates)
        self.assertEqual(residual["vector"].shape, (4, 5))
        self.assertEqual(residual["physics_features"].shape, (4, 3))
        loss = physics.pde_loss(residual)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_boundary_sampling_has_all_types(self) -> None:
        generator = torch.Generator().manual_seed(1)
        points, tags = sample_boundary(20, generator)
        self.assertEqual(points.shape, (20, 3))
        self.assertEqual(set(tags.tolist()), {0, 1, 2, 3})

    def test_apr_respects_budget(self) -> None:
        generator = torch.Generator().manual_seed(2)
        pool = CollocationPool.initialize(100, generator)
        pool.residual_scores = torch.linspace(0.0, 1.0, 100).reshape(-1, 1)
        controller = APRController(self.config)
        statistics = controller.refine(pool, generator)
        self.assertLessEqual(len(pool), self.config["sampling"]["max_points"])
        self.assertGreater(len(pool), 0)
        self.assertEqual(statistics["points_after"], len(pool))

    def test_reference_metrics(self) -> None:
        reference = np.array([[1.0, 2.0], [3.0, np.nan]])
        prediction = np.array([[1.0, 1.0], [4.0, 100.0]])
        metrics = field_metrics(prediction, reference)
        self.assertEqual(metrics["valid_points"], 3)
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.0 / 3.0))


class APRPiT3DCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = smoke_test_config(
            load_config(PROJECT_ROOT / "configs" / "tunnel_3d.yaml")
        )

    def test_3d_model_and_residual_shapes(self) -> None:
        model = APRPiT(self.config)
        physics = TunnelFirePhysics(self.config)
        coordinates = torch.rand(3, 4, requires_grad=True)
        prediction = model(coordinates)
        self.assertEqual(prediction["normalized_output"].shape, (3, 6))
        self.assertEqual(prediction["v"].shape, (3, 1))
        residual = physics.residuals(prediction, coordinates)
        self.assertEqual(residual["vector"].shape, (3, 6))
        loss = physics.pde_loss(residual)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_3d_boundary_sampling_has_six_faces(self) -> None:
        generator = torch.Generator().manual_seed(3)
        points, tags = sample_boundary(24, generator, dimensions=4)
        self.assertEqual(points.shape, (24, 4))
        self.assertEqual(set(tags.tolist()), {0, 1, 2, 3, 4, 5})

    def test_3d_boundary_loss_is_finite(self) -> None:
        generator = torch.Generator().manual_seed(4)
        coordinates, tags = sample_boundary(12, generator, dimensions=4)
        coordinates.requires_grad_(True)
        model = APRPiT(self.config)
        physics = TunnelFirePhysics(self.config)
        loss = physics.boundary_loss(model(coordinates), coordinates, tags)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
