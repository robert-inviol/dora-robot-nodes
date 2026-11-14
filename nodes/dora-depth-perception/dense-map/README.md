# dense-map

The dense map of the house, built from posed depth keyframes, with no dora dependency. The map frame is z up,
gravity aligned, metres; a keyframe pose (x, y, z, yaw) places that keyframe's gravity-aligned floor frame from
`depth-perception` in it.

| Stage | What it does |
|---|---|
| `BlockTsdf` (`DistanceField`) | A TSDF at 5 cm in 8x8x8-voxel blocks in a hash map. Each keyframe casts rays through every fourth pixel to find the blocks it sees, then projects every voxel of those blocks into its depth image in parallel and folds the signed distance into a weighted average. Readings behind a surface lose weight towards the truncation distance. The `DistanceField` trait (integrate, voxel and point queries, surface points, bounds) is what the rest uses, so another implementation can replace it. |
| `levels` | The floor is the lowest level of upward-facing surface holding a good share of the most crowded level's points; ceiling points face down well above it. |
| `slice_field` | A floor-plan grid at the voxel size through a height band above the floor (0.05 m to 0.60 m): occupied where any voxel of the band is near or behind a surface, free where every voxel of the band was seen clear, unknown otherwise. |
| `squared_distance_transform` | Felzenszwalb and Huttenlocher's exact Euclidean distance transform; `free_space` gives each free cell its distance to the nearest occupied cell. |
| `Topologist` | Cuts free space into rooms at doorways: pinch points of the clearance ridge narrower than a door (1.1 m) where clearance rises on both sides. Rooms under 1.5 m2 merge into the neighbour behind their widest doorway. Places go along the ridge 0.75 m apart, with doorway middles first; edges join places that see each other within 1.5 m, thinned to the relative neighbourhood graph. Room ids follow the largest overlap with the previous cut; doorway and place ids follow the nearest previous one close enough. |
| `KeepOutZones` | One zone per object whose label is on the list (glass door, glass, window, mirror, glass table): its footprint and a margin. |
| `DenseMapper` | Keyframes wait for their poses (or poses for their keyframes); a lost pose is not integrated. Every integrated keyframe is kept, so when the localizer moves keyframes or raises the map version the map is rebuilt from scratch on a background thread while the old one keeps serving. Free space and topology are sliced at most once a period, and topology is sent only when it changes. |

Topology treats unknown cells as walls, so the edge of what has been seen bounds a room; small unknown pockets
bordered mostly by free cells (the floor right under where the camera stood) count as free. Place and edge
clearances are distances to the nearest cell that is not known free.

Every setting is in `DenseMapSettings`; `../dora-dense-map/dense-map-settings.toml` lists them at their defaults.

## Running

```
cargo test --release -p dense-map
cargo run --release -p dense-map --example profile_house
```

`profile_house` renders a walk through a 15 m x 15 m house of six rooms at 848x480 and times integration and
slicing; `SHOW_PLAN=1` prints the rooms it cut, `OPEN_PLAN=1` leaves out the inner walls.
