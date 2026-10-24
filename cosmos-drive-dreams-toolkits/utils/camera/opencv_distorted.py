# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OpenCV Brown-Conrady camera model used by nuPlan cameras."""

import cv2
import numpy as np
import torch

from utils.camera.pinhole import PinholeCamera


class OpenCVDistortedCamera(PinholeCamera):
    """Pinhole camera with OpenCV ``k1, k2, p1, p2, k3`` distortion."""

    def __init__(self, fx, fy, cx, cy, w, h, distortion, dtype=torch.float32, device=None):
        self.distortion = np.asarray(distortion, dtype=np.float32).reshape(5)
        super().__init__(fx, fy, cx, cy, w, h, dtype=dtype, device=device)
        self._update_serialized_intrinsics()
        self._update_valid_radius()

    @classmethod
    def from_numpy(cls, intrinsics: np.ndarray, device=None):
        if len(intrinsics) != 11:
            raise ValueError(
                "OpenCV distorted intrinsics must contain "
                "[fx, fy, cx, cy, width, height, k1, k2, p1, p2, k3]"
            )
        return cls(*intrinsics[:6], intrinsics[6:11], device=device)

    def _update_serialized_intrinsics(self):
        self._intrinsics = np.asarray(
            [self.fx, self.fy, self.cx, self.cy, self.w, self.h, *self.distortion],
            dtype=np.float32,
        )

    def _update_valid_radius(self):
        corners = np.asarray(
            [[0, 0], [self.w - 1, 0], [0, self.h - 1], [self.w - 1, self.h - 1]],
            dtype=np.float32,
        )
        undistorted = cv2.undistortPoints(
            corners.reshape(-1, 1, 2), self.intrinsics_matrix_np, self.distortion
        ).reshape(-1, 2)
        self._max_valid_radius = float(np.linalg.norm(undistorted, axis=1).max() * 1.01)

    def rescale(self, ratio_h: float, ratio_w: float = None):
        super().rescale(ratio_h, ratio_w)
        self._update_serialized_intrinsics()
        self._update_valid_radius()
        if hasattr(self, "rays_cached"):
            del self.rays_cached

    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        rays = np.asarray(rays)
        if rays.ndim == 1:
            rays = rays.reshape(1, -1)
        x = rays[:, 0] / rays[:, 2]
        y = rays[:, 1] / rays[:, 2]
        k1, k2, p1, p2, k3 = self.distortion
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        pixel_coords = np.column_stack(
            [self.fx * x_distorted + self.cx, self.fy * y_distorted + self.cy]
        )
        # Brown-Conrady polynomials can fold back outside their calibrated
        # image domain. Keep such rays outside the canvas instead of allowing
        # distant geometry to reappear near the optical center.
        radius = np.sqrt(r2)
        invalid = radius > self._max_valid_radius
        if np.any(invalid):
            direction = np.column_stack([x[invalid], y[invalid]]) / radius[invalid, None]
            pixel_coords[invalid] = np.asarray([self.cx, self.cy]) + direction * max(self.w, self.h) * 4
        return pixel_coords

    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        if rays.ndim == 1:
            rays = rays.unsqueeze(0)
        x = rays[:, 0] / rays[:, 2]
        y = rays[:, 1] / rays[:, 2]
        distortion = torch.as_tensor(self.distortion, dtype=rays.dtype, device=rays.device)
        k1, k2, p1, p2, k3 = distortion
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2.square() + k3 * r2.pow(3)
        x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
        y_distorted = y * radial + p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y
        pixel_coords = torch.stack(
            [self.fx * x_distorted + self.cx, self.fy * y_distorted + self.cy], dim=-1
        )
        radius = torch.sqrt(r2)
        invalid = radius > self._max_valid_radius
        if torch.any(invalid):
            direction = torch.stack([x[invalid], y[invalid]], dim=-1) / radius[invalid, None]
            center = torch.as_tensor([self.cx, self.cy], dtype=rays.dtype, device=rays.device)
            pixel_coords[invalid] = center + direction * max(self.w, self.h) * 4
        return pixel_coords

    def pixel2ray_np(self, pixel_coords: np.ndarray) -> np.ndarray:
        pixel_coords = np.asarray(pixel_coords, dtype=np.float32)
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, -1)
        normalized = cv2.undistortPoints(
            pixel_coords.reshape(-1, 1, 2),
            self.intrinsics_matrix_np,
            self.distortion,
        ).reshape(-1, 2)
        return np.column_stack([normalized, np.ones(len(normalized), dtype=np.float32)])

    def pixel2ray_torch(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        rays = self.pixel2ray_np(pixel_coords.detach().cpu().numpy())
        return torch.as_tensor(rays, dtype=pixel_coords.dtype, device=pixel_coords.device)

    def _get_rays_impl(self) -> torch.Tensor:
        u, v = np.meshgrid(np.arange(self.w), np.arange(self.h), indexing="xy")
        pixels = np.column_stack([u.reshape(-1), v.reshape(-1)])
        rays = self.pixel2ray_np(pixels)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return torch.as_tensor(rays.reshape(self.h, self.w, 3), dtype=self.dtype, device=self.device)
