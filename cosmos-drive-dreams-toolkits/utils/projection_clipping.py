"""Camera-space and image-space clipping for projected filled geometry."""

import numpy as np


DEFAULT_NEAR_PLANE = 0.1


def _interpolate(a, b, t):
    return a + t * (b - a)


def clip_polygon_near_plane(points_in_camera, attributes=None, near=DEFAULT_NEAR_PLANE):
    """Clip a convex camera-space polygon against ``z >= near``."""
    points = np.array(points_in_camera, dtype=np.float64, copy=True)
    attrs = None if attributes is None else np.asarray(attributes, dtype=np.float64)
    if len(points) == 0:
        return points, attrs

    output_points = []
    output_attrs = [] if attrs is not None else None
    previous = points[-1]
    previous_attr = attrs[-1] if attrs is not None else None
    previous_inside = previous[2] >= near

    for index, current in enumerate(points):
        current_attr = attrs[index] if attrs is not None else None
        current_inside = current[2] >= near
        if current_inside != previous_inside:
            denominator = current[2] - previous[2]
            t = (near - previous[2]) / denominator
            output_points.append(_interpolate(previous, current, t))
            if output_attrs is not None:
                output_attrs.append(_interpolate(previous_attr, current_attr, t))
        if current_inside:
            output_points.append(current)
            if output_attrs is not None:
                output_attrs.append(current_attr)
        previous = current
        previous_attr = current_attr
        previous_inside = current_inside

    clipped_points = np.asarray(output_points, dtype=np.float64).reshape(-1, 3)
    if output_attrs is None:
        return clipped_points, None
    return clipped_points, np.asarray(output_attrs, dtype=np.float64)


def projection_domain_mask(camera_model, points_in_camera):
    """Return rays that lie inside the calibrated forward-projection domain."""
    points = np.asarray(points_in_camera, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    positive = points[:, 2] > 0
    mask = finite & positive

    # Brown-Conrady is calibrated over a bounded normalized-image radius.
    if hasattr(camera_model, "_max_valid_radius"):
        normalized_radius = np.linalg.norm(points[:, :2] / points[:, 2:3], axis=1)
        mask &= normalized_radius <= float(camera_model._max_valid_radius)

    # F-theta extrapolates beyond this angle, which is useful for isolated
    # points but unsafe as a vertex of a filled face.
    if hasattr(camera_model, "_max_ray_angle") and camera_model._max_ray_angle is not None:
        norm = np.linalg.norm(points, axis=1)
        cosine = np.divide(points[:, 2], norm, out=np.zeros_like(norm), where=norm > 0)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        mask &= angle <= float(np.asarray(camera_model._max_ray_angle).reshape(-1)[0])
    return mask


def _clip_projected_polygon_axis(vertices, attributes, axis, boundary, keep_greater):
    if len(vertices) == 0:
        return vertices, attributes

    output_vertices = []
    output_attrs = [] if attributes is not None else None
    previous = vertices[-1]
    previous_attr = attributes[-1] if attributes is not None else None

    def inside(vertex):
        return vertex[axis] >= boundary if keep_greater else vertex[axis] <= boundary

    previous_inside = inside(previous)
    for index, current in enumerate(vertices):
        current_attr = attributes[index] if attributes is not None else None
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            t = (boundary - previous[axis]) / denominator
            output_vertices.append(_interpolate(previous, current, t))
            if output_attrs is not None:
                output_attrs.append(_interpolate(previous_attr, current_attr, t))
        if current_inside:
            output_vertices.append(current)
            if output_attrs is not None:
                output_attrs.append(current_attr)
        previous = current
        previous_attr = current_attr
        previous_inside = current_inside

    clipped_vertices = np.asarray(output_vertices, dtype=np.float64).reshape(-1, 3)
    if output_attrs is None:
        return clipped_vertices, None
    return clipped_vertices, np.asarray(output_attrs, dtype=np.float64)


def clip_projected_polygon_to_image(vertices, attributes, width, height):
    """Clip a convex ``[x, y, depth]`` polygon to the image rectangle."""
    clipped_vertices = np.asarray(vertices, dtype=np.float64)
    clipped_attrs = None if attributes is None else np.asarray(attributes, dtype=np.float64)
    for axis, boundary, keep_greater in (
        (0, 0.0, True),
        (0, float(width - 1), False),
        (1, 0.0, True),
        (1, float(height - 1), False),
    ):
        clipped_vertices, clipped_attrs = _clip_projected_polygon_axis(
            clipped_vertices, clipped_attrs, axis, boundary, keep_greater
        )
        if len(clipped_vertices) < 3:
            break
    return clipped_vertices, clipped_attrs


def triangulate_convex_polygon(vertices, attributes=None):
    """Triangulate a convex polygon while preserving optional vertex attributes."""
    vertices = np.asarray(vertices)
    if len(vertices) < 3:
        empty = np.empty((0, 3, 3), dtype=np.float64)
        return empty, None if attributes is None else empty.copy()
    triangles = np.asarray(
        [[vertices[0], vertices[index], vertices[index + 1]] for index in range(1, len(vertices) - 1)]
    )
    if attributes is None:
        return triangles, None
    attributes = np.asarray(attributes)
    triangle_attributes = np.asarray(
        [[attributes[0], attributes[index], attributes[index + 1]] for index in range(1, len(attributes) - 1)]
    )
    return triangles, triangle_attributes


def project_and_clip_triangle(
    camera_model,
    points_in_camera,
    attributes=None,
    near=DEFAULT_NEAR_PLANE,
):
    """Near-clip, project, image-clip, and triangulate one camera-space triangle."""
    camera_polygon, clipped_attrs = clip_polygon_near_plane(
        points_in_camera, attributes=attributes, near=near
    )
    if len(camera_polygon) < 3:
        return np.empty((0, 3, 3)), None

    # Do not connect a calibrated ray to the artificial far-away coordinates
    # produced by extrapolation outside a camera model's valid domain.
    if not np.all(projection_domain_mask(camera_model, camera_polygon)):
        return np.empty((0, 3, 3)), None

    pixels = np.asarray(camera_model.ray2pixel(camera_polygon), dtype=np.float64)
    if not np.isfinite(pixels).all():
        return np.empty((0, 3, 3)), None
    projected = np.column_stack([pixels, camera_polygon[:, 2]])
    projected, clipped_attrs = clip_projected_polygon_to_image(
        projected,
        clipped_attrs,
        camera_model.width,
        camera_model.height,
    )
    return triangulate_convex_polygon(projected, clipped_attrs)


def project_and_clip_segment(camera_model, points_in_camera, near=DEFAULT_NEAR_PLANE):
    """Clip and project one 3D line segment, returning ``[2, 3]`` or ``None``."""
    points = np.array(points_in_camera, dtype=np.float64, copy=True)
    inside = points[:, 2] >= near
    if not inside.any():
        return None
    if inside.sum() == 1:
        outside_index = int(np.flatnonzero(~inside)[0])
        inside_index = 1 - outside_index
        outside_point = points[outside_index]
        inside_point = points[inside_index]
        t = (near - outside_point[2]) / (inside_point[2] - outside_point[2])
        points[outside_index] = _interpolate(outside_point, inside_point, t)

    if not np.all(projection_domain_mask(camera_model, points)):
        return None
    pixels = np.asarray(camera_model.ray2pixel(points), dtype=np.float64)
    if not np.isfinite(pixels).all():
        return None
    projected = np.column_stack([pixels, points[:, 2]])

    # Liang-Barsky clipping in pixel space, interpolating depth at intersections.
    delta = projected[1] - projected[0]
    t_min, t_max = 0.0, 1.0
    for p, q in (
        (-delta[0], projected[0, 0]),
        (delta[0], camera_model.width - 1 - projected[0, 0]),
        (-delta[1], projected[0, 1]),
        (delta[1], camera_model.height - 1 - projected[0, 1]),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            t_min = max(t_min, ratio)
        else:
            t_max = min(t_max, ratio)
        if t_min > t_max:
            return None
    start = _interpolate(projected[0], projected[1], t_min)
    end = _interpolate(projected[0], projected[1], t_max)
    return np.stack([start, end])


def project_and_clip_segments(camera_model, segments_in_camera, near=DEFAULT_NEAR_PLANE):
    """Vectorized near/domain/image clipping for camera-space line segments."""
    segments = np.array(segments_in_camera, dtype=np.float64, copy=True).reshape(-1, 2, 3)
    if len(segments) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    finite = np.isfinite(segments).all(axis=(1, 2))
    inside = segments[:, :, 2] >= near
    keep = finite & inside.any(axis=1)
    segments = segments[keep]
    inside = inside[keep]
    if len(segments) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    crossing = inside[:, 0] != inside[:, 1]
    crossing_indices = np.flatnonzero(crossing)
    if len(crossing_indices):
        behind_indices = (~inside[crossing]).argmax(axis=1)
        front_indices = 1 - behind_indices
        behind = segments[crossing_indices, behind_indices]
        front = segments[crossing_indices, front_indices]
        fraction = (near - behind[:, 2]) / (front[:, 2] - behind[:, 2])
        segments[crossing_indices, behind_indices] = behind + fraction[:, None] * (front - behind)

    domain_valid = projection_domain_mask(camera_model, segments.reshape(-1, 3)).reshape(-1, 2)
    segments = segments[domain_valid.all(axis=1)]
    if len(segments) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    pixels = np.asarray(camera_model.ray2pixel(segments.reshape(-1, 3)), dtype=np.float64).reshape(-1, 2, 2)
    projected = np.concatenate([pixels, segments[:, :, 2:3]], axis=2)
    projected = projected[np.isfinite(projected).all(axis=(1, 2))]
    if len(projected) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)

    delta = projected[:, 1] - projected[:, 0]
    t_min = np.zeros(len(projected), dtype=np.float64)
    t_max = np.ones(len(projected), dtype=np.float64)
    valid = np.ones(len(projected), dtype=bool)
    constraints = (
        (-delta[:, 0], projected[:, 0, 0]),
        (delta[:, 0], camera_model.width - 1 - projected[:, 0, 0]),
        (-delta[:, 1], projected[:, 0, 1]),
        (delta[:, 1], camera_model.height - 1 - projected[:, 0, 1]),
    )
    for p, q in constraints:
        parallel = np.abs(p) < 1e-12
        valid &= ~(parallel & (q < 0))
        nonparallel = ~parallel
        ratio = np.zeros_like(p)
        ratio[nonparallel] = q[nonparallel] / p[nonparallel]
        entering = nonparallel & (p < 0)
        leaving = nonparallel & (p > 0)
        t_min[entering] = np.maximum(t_min[entering], ratio[entering])
        t_max[leaving] = np.minimum(t_max[leaving], ratio[leaving])
        valid &= t_min <= t_max

    projected = projected[valid]
    delta = delta[valid]
    t_min = t_min[valid]
    t_max = t_max[valid]
    if len(projected) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    start = projected[:, 0] + t_min[:, None] * delta
    end = projected[:, 0] + t_max[:, None] * delta
    return np.stack([start, end], axis=1)
