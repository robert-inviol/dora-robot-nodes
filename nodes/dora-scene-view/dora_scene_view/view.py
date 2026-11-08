"""The data the debug page draws, built from the latest keyframe, segments and scene."""

import base64
import io

import numpy as np
from PIL import Image

from dora_rig_messages.perception import NO_SEGMENT, Keyframe, Scene, Segments

POINT_STRIDE_PX = 2
MIN_RANGE_M, MAX_RANGE_M = 0.4, 8.0
# A point this close to a plane is drawn on it.
PLANE_DISTANCE_M = 0.10
NO_PLANE = -1
# Fewer points than this give a centroid too noisy to label or fly to.
MIN_POINTS_TO_LOCATE = 20


def floor_axes(gravity_reaction: tuple[float, float, float]) -> np.ndarray:
    """Rows are (right, forward, up) in camera coordinates, with right and forward level."""
    up = np.asarray(gravity_reaction, dtype=np.float64)
    up /= np.linalg.norm(up)
    forward = np.array([0.0, 0.0, 1.0]) - up[2] * up
    forward /= np.linalg.norm(forward)
    return np.stack([np.cross(forward, up), forward, up])


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def page_data(keyframe: Keyframe, segments: Segments | None, scene: Scene | None, version: str) -> dict:
    """Segments and scene are drawn only if they belong to this keyframe."""
    segments = segments if segments is not None and segments.frame_id == keyframe.frame_id else None
    scene = scene if scene is not None and scene.frame_id == keyframe.frame_id else None
    k = keyframe.intrinsics
    v, u = np.mgrid[0 : keyframe.height : POINT_STRIDE_PX, 0 : keyframe.width : POINT_STRIDE_PX]
    z = keyframe.depth_mm[v, u].astype(np.float32) * 0.001
    valid = (z > MIN_RANGE_M) & (z < MAX_RANGE_M)
    u, v, z = u[valid], v[valid], z[valid]
    camera_points = np.stack([(u - k.ppx) * z / k.fx, (v - k.ppy) * z / k.fy, z], axis=1)
    axes = floor_axes(keyframe.gravity_reaction)
    points = (camera_points @ axes.T).astype(np.float32)
    color = np.asarray(Image.open(io.BytesIO(keyframe.color_jpeg)).convert("RGB"))
    point_segments = segments.segment_ids[v, u] if segments else np.full(len(points), NO_SEGMENT, np.int16)

    point_planes = np.full(len(points), NO_PLANE, np.int16)
    flattened = points.copy()
    planes = scene.planes if scene else ()
    for index, plane in enumerate(planes):
        normal = np.asarray(plane.normal, np.float32)
        distance = points @ normal + plane.offset_m
        on = (np.abs(distance) < PLANE_DISTANCE_M) & (point_planes == NO_PLANE)
        point_planes[on] = index
        flattened[on] -= distance[on, None] * normal

    boxes = {o.segment_index: o for o in scene.objects} if scene else {}
    objects = []
    for index, (label, score) in enumerate(zip(segments.labels, segments.scores) if segments else ()):
        own = points[point_segments == index]
        located = len(own) >= MIN_POINTS_TO_LOCATE
        box = boxes.get(index)
        objects.append({
            "label": label,
            "score": round(score, 2),
            "point_count": int(len(own)),
            "centroid": own.mean(axis=0).tolist() if located else None,
            "top": float(np.percentile(own[:, 2], 98)) if located else None,
            "box": None if box is None else {
                "distance_m": box.distance_m,
                "size_m": [box.footprint_long_m, box.footprint_short_m, box.height_m],
                "corners": [list(c) for c in box.corners_m],
                "bottom": box.top_m - box.height_m,
                "top": box.top_m,
            },
        })

    room = scene.room if scene else None
    return {
        "version": version,
        "name": f"frame {keyframe.frame_id}",
        "positions_b64": _b64(points),
        "flattened_b64": _b64(flattened),
        "colors_b64": _b64(color[v, u].astype(np.uint8)),
        "segments_b64": _b64(point_segments.astype(np.int16)),
        "planes_b64": _b64(point_planes),
        "no_segment": NO_SEGMENT,
        "no_plane": NO_PLANE,
        "camera": {
            "forward": (axes @ np.array([0.0, 0.0, 1.0])).tolist(),
            "up": (axes @ np.array([0.0, -1.0, 0.0])).tolist(),
            "vertical_fov_deg": float(np.degrees(2 * np.arctan(keyframe.height / 2 / k.fy))),
            "median_depth_m": float(np.median(z)) if len(z) else 1.0,
        },
        "planes": [{"kind": p.kind.value, "area_m2": round(p.area_m2, 1), "outline": [list(c) for c in p.outline_m]} for p in planes],
        "room": None if room is None else {
            "floor": room.floor_m, "ceiling": room.ceiling_m,
            "footprint": [list(c) for c in room.footprint_m], "edge_seen": list(room.edge_seen),
        },
        "room_problem": scene.room_problem if scene else "waiting for the scene",
        "objects": objects,
    }
