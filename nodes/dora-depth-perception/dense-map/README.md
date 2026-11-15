# dense-map

The dense map of the house, built from posed keyframes and posed depth frames, with no dora dependency. The map frame is z up,
gravity aligned, metres; a keyframe pose (x, y, z, yaw) places that keyframe's gravity-aligned floor frame from
`depth-perception` in it.

| Stage | What it does |
|---|---|
| `BlockTsdf` (`Tsdf`, `BlockField`) | A TSDF at 5 cm in 8x8x8-voxel blocks in a hash map, each block shared with snapshots of the field until it is fused into again. Each keyframe casts rays through every fourth pixel to find the blocks it sees, then projects every voxel of those blocks into its depth image in parallel and folds the signed distance into a weighted average. Readings behind a surface lose weight towards the truncation distance. The `Tsdf` trait (fuse, voxel and point queries, surface points, bounds) is what the rest uses, so another implementation can replace it. `BlockField` adds what a partial rebuild needs: a frame's footprint (the blocks it touches), fusing a frame into some blocks only, replacing blocks with another field's, and a cheap snapshot. |
| `levels` | The floor is the lowest level of upward-facing surface holding a good share of the most crowded level's points; ceiling points face down well above it. |
| `slice_field` | A floor-plan grid at the voxel size through a height band above the floor (0.05 m to 0.60 m): occupied where any voxel of the band is near or behind a surface, free where every voxel of the band was seen clear, unknown otherwise. |
| `squared_distance_transform` | Felzenszwalb and Huttenlocher's exact Euclidean distance transform; `occupancy_grid` gives each free cell its distance to the nearest occupied cell. |
| `TopologyBuilder` | Cuts free space into rooms at doorways: pinch points of the clearance ridge narrower than a door (1.1 m) where clearance rises on both sides. Rooms under 1.5 m2 merge into the neighbour behind their widest doorway. Places go along the ridge 0.75 m apart, with doorway middles first; edges join places that see each other within 1.5 m, thinned to the relative neighbourhood graph. Room ids follow the largest overlap with the previous cut; doorway and place ids follow the nearest previous one close enough. |
| `KeepOutZones` | One zone per object whose label is on the list (glass door, glass, window, mirror, glass table): its footprint and a margin. |
| `KeptKeyframes` | The keyframes kept to rebuild from, each with its footprint. Keyframe poses fall in pose cells (0.3 m along x, y and z, 20 degrees in yaw); a cell keeps only its newest keyframe, and past 500 kept keyframes the oldest with a kept neighbour one cell away goes, or the oldest when none has one. A keyframe let go is remembered by its footprint (up to 20000 of them), so that when it moves, the blocks it left behind are rebuilt without it. |
| `DenseMapper` | Posed depth frames are fused as they arrive, at their odometry pose carried into the map by the latest `map_to_odom` of their odom frame; frames before the first correction or of another odom frame are not fused. They are fused once and not kept. Keyframes wait for their poses (or poses for their keyframes); a lost pose is not fused. A restarted keyframe is fused at its pose like a tracked one: it is a map keyframe, and the pose graph updates that tie its run to the map move it like any other. When the localizer moves a keyframe by at least 1 cm or 0.25 degrees, the blocks of its footprint at its old and new poses are rebuilt on a background thread, from the kept keyframes that touch them, while the map keeps serving; moves during a rebuild wait for it to land, and nothing else starts one. Posed depth fused into a rebuilt block is gone from it, since it carried the uncorrected pose. |
| `DerivedMaps` | Meshing and slicing, run on a snapshot of the field on a background thread so input is never held up: the surface of the changed blocks at most once per `mesh.period_s` (0.2 s), and the occupancy grid and topology at most once per `occupancy_grid.period_s` (1 s), topology sent only when it changes. The mapper keeps every mesh block's surface as last sent, so `mesh_snapshot` can send the whole surface as a snapshot, numbered next in the run's sequence, to a receiver that missed a message. |

Topology treats unknown cells as walls, so the edge of what has been seen bounds a room; small unknown pockets
bordered mostly by free cells (the floor right under where the camera stood) count as free. Place and edge
clearances are distances to the nearest cell that is not known free.

Every setting is in `DenseMapSettings`; `../dora-dense-map/dense-map-settings.toml` lists them at their defaults.

## Running

```
cargo test --release -p dense-map
cargo run --release -p dense-map --example profile_house
```

A rebuilt block equals the kept keyframes fused into it from scratch (`tests/fusion.rs` checks every voxel).

`profile_house` renders a walk through a 15 m x 15 m house of six rooms at 848x480 and times fusion and
slicing; `SHOW_PLAN=1` prints the rooms it cut, `OPEN_PLAN=1` leaves out the inner walls.
