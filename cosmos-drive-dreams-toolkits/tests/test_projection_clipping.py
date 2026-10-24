import numpy as np

from utils.camera.opencv_distorted import OpenCVDistortedCamera
from utils.camera.pinhole import PinholeCamera
from utils.bbox_utils import clip_triangle_to_z_near
from utils.projection_clipping import (
    project_and_clip_segment,
    project_and_clip_segments,
    project_and_clip_triangle,
)


def _assert_inside_image(vertices, camera):
    assert np.isfinite(vertices).all()
    assert np.all(vertices[..., 0] >= 0)
    assert np.all(vertices[..., 0] <= camera.width - 1)
    assert np.all(vertices[..., 1] >= 0)
    assert np.all(vertices[..., 1] <= camera.height - 1)
    assert np.all(vertices[..., 2] >= 0.1)


def test_triangle_crossing_near_plane_is_clipped_and_triangulated():
    camera = PinholeCamera(100, 100, 50, 50, 100, 100, device="cpu")
    triangle = np.array([
        [-0.02, -0.02, -1.0],
        [0.25, -0.15, 1.0],
        [0.0, 0.25, 1.0],
    ])
    colors = np.eye(3)

    triangles, triangle_colors = project_and_clip_triangle(
        camera, triangle, attributes=colors
    )

    assert triangles.shape[0] >= 2
    assert triangles.shape[1:] == (3, 3)
    assert triangle_colors.shape == triangles.shape
    _assert_inside_image(triangles, camera)


def test_triangle_fully_behind_camera_is_rejected():
    camera = PinholeCamera(100, 100, 50, 50, 100, 100, device="cpu")
    triangle = np.array([
        [-1.0, -1.0, -2.0],
        [1.0, -1.0, -2.0],
        [0.0, 1.0, -2.0],
    ])

    triangles, _ = project_and_clip_triangle(camera, triangle)

    assert triangles.shape == (0, 3, 3)


def test_triangle_is_clipped_to_image_rectangle():
    camera = PinholeCamera(100, 100, 50, 50, 100, 100, device="cpu")
    triangle = np.array([
        [-1.0, -0.2, 1.0],
        [0.2, -0.2, 1.0],
        [0.2, 0.2, 1.0],
    ])

    triangles, _ = project_and_clip_triangle(camera, triangle)

    assert len(triangles) > 0
    _assert_inside_image(triangles, camera)


def test_opencv_ray_outside_calibrated_domain_is_rejected():
    camera = OpenCVDistortedCamera(
        100, 100, 50, 50, 100, 100, [0, 0, 0, 0, 0], device="cpu"
    )
    triangle = np.array([
        [5.0, 0.0, 1.0],
        [5.2, 0.1, 1.0],
        [5.0, 0.2, 1.0],
    ])

    triangles, _ = project_and_clip_triangle(camera, triangle)

    assert triangles.shape == (0, 3, 3)


def test_segment_crossing_near_plane_and_image_edge_is_clipped():
    camera = PinholeCamera(100, 100, 50, 50, 100, 100, device="cpu")
    segment = np.array([
        [-0.02, 0.0, -1.0],
        [0.2, 0.0, 1.0],
    ])

    projected = project_and_clip_segment(camera, segment)

    assert projected.shape == (2, 3)
    _assert_inside_image(projected, camera)


def test_vectorized_segments_match_individual_clipping():
    camera = PinholeCamera(100, 100, 50, 50, 100, 100, device="cpu")
    segments = np.array(
        [
            [[-0.02, 0.0, -1.0], [0.2, 0.0, 1.0]],
            [[-1.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
            [[-1.0, 0.0, -2.0], [1.0, 0.0, -2.0]],
        ]
    )
    original = segments.copy()
    expected = [
        projected
        for segment in segments
        if (projected := project_and_clip_segment(camera, segment)) is not None
    ]

    projected = project_and_clip_segments(camera, segments)

    assert np.allclose(segments, original)
    assert np.allclose(projected, np.asarray(expected))
    _assert_inside_image(projected, camera)


def test_pr46_bbox_near_plane_is_preserved_during_triangle_clipping():
    near = 5e-2
    triangle = np.array([
        [0.0, 0.0, -1.0],
        [0.02, -0.02, 1.0],
        [-0.02, 0.02, 1.0],
    ])
    colors = np.eye(3)

    clipped = clip_triangle_to_z_near(
        triangle,
        colors,
        z_near=near,
    )

    assert len(clipped) == 2
    assert all(np.all(points[:, 2] >= near) for points, _ in clipped)
    assert all(points.shape == vertex_colors.shape == (3, 3) for points, vertex_colors in clipped)
