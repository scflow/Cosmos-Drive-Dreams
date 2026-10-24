import numpy as np

from convert_nuplan_to_rds_hq import build_oriented_cuboid, traffic_light_samples


def test_build_oriented_cuboid_preserves_dimensions():
    vertices = build_oriented_cuboid(
        center=np.array([3.0, -2.0, 5.0]),
        yaw=np.pi / 2,
        lwh=(0.35, 0.25, 0.75),
    )

    assert vertices.shape == (8, 3)
    assert np.allclose(vertices.mean(axis=0), [3.0, -2.0, 5.0])
    assert np.allclose(np.sort(np.ptp(vertices, axis=0)), np.sort([0.35, 0.25, 0.75]))


def test_traffic_light_samples_are_dense_and_resolve_conflicts_to_unknown():
    heads = [
        {
            "feature_id": "101",
            "connector_ids": ["1", "2"],
            "light_face_type_fid": 1,
            "vertices": np.zeros((8, 3)).tolist(),
        }
    ]
    statuses = {
        b"a": {"1": "red", "2": "red"},
        b"b": {"1": "red", "2": "green"},
    }

    positions, states = traffic_light_samples(
        "clip", heads, statuses, [b"a", b"b"], pose_count=6
    )

    assert len(positions["traffic_lights.json"]["labels"]) == 1
    signal = states["aggregated_states.json"]["traffic_light_states"]["0"]
    assert signal["feature_id"] == "101"
    assert signal["state"] == ["red", "red", "red", "unknown", "unknown", "unknown"]


def test_traffic_light_samples_use_rds_empty_sentinel():
    positions, states = traffic_light_samples(
        "clip", [], {}, [], pose_count=3
    )

    labels = positions["traffic_lights.json"]["labels"]
    assert labels == [{"labelData": {"emptyLabel": True}}]
    assert states["aggregated_states.json"]["traffic_light_states"] == {}
