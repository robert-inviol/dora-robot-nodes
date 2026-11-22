# object-map

Persistent objects from SAM 3 masks, in the map frame (z up, gravity aligned, metres).

Each keyframe arrives with its instance masks and its pose from the localizer (`ObservedFrame`). For every mask:

1. The mask's pixels with depth are lifted into the keyframe's floor frame and cleaned in 3D (`cleaning`, settings
   `detection.cleaning`). Points are counted in `cell_m` (3 cm) cells; a point whose cell and the 26 around it hold
   fewer than `min_neighbourhood_points` (8) is a flying pixel or noise and is dropped. The cells left are joined into
   clusters: two cells join with up to `max_gap_cells` (2) empty cells between them across the line of sight, so chair
   legs, lamp stems and plants are not cut apart, and up to `max_depth_gap_cells` (5) along it, where the front of an
   object hides the join to a part behind it, such as a lamp stem under the front edge of its shade. Only the cluster
   with the most points is kept: a mask that spills onto the side of a neighbouring object at a similar depth, or
   covers background seen through the object, loses the spill. A dropped cluster holding at least
   `large_cluster_share` (0.2) of the mask's points is logged at info with its size: usually two objects under one
   mask, which is kept as one detection of the larger. Then the kept points at least `erosion_px` inside the mask
   (the mask's edge can lie on the surface the object stands on or leans against, which cleaning cannot separate
   where the two touch; `depth-perception`'s erosion) are moved into the map with the keyframe pose and counted in
   5 cm voxels (`detection`). The mask's pixels less those whose points were dropped are the mask the object's view is
   cut with. The 3D clustering takes the place of `depth-perception`'s depth band and footprint rectangle, which the
   object map does not use.
2. The detection belongs to the object it shares the most space with: at least `association.min_overlap` of its
   voxels touch the object's (one voxel of slack, `voxels`) and, unless their labels agree (`label`: same words,
   ignoring case and spacing), as much of the object's touch its own. So a "throw pillow" where a "cushion" is known
   is that cushion, a part of a sofa seen as "sofa" is that sofa, and a cushion on a sofa is not the sofa. Otherwise
   it starts a candidate. An object's label is the one with the most detection score behind it (`LabelVotes`).

Then, per keyframe, in order (`map`):

- **Absences.** Each confirmed object not observed is projected into the keyframe. Voxels where the measured depth is
  within `absence.surface_margin_m` see the surface; farther behind sees past it; nearer or no reading is hidden.
  Enough surface resets the misses; mostly seen-past counts a miss; `misses_to_disappear` misses in a row and the
  object `disappeared` (`evidence`).
- **Confirmation.** A candidate observed in `confirmation.observations` keyframes within `window_s` `appeared`. If an
  object with an agreeing label disappeared within `absence.moved_window_s`, the candidate takes its id and the
  object `moved`.
- **Merging**, every `merge.interval_s` of capture time: confirmed objects where the smaller one's voxels touch the
  larger by `merge.min_overlap` and, unless their labels agree, the larger one's touch the smaller as much, merge
  into the older id (`merged` with `merged_into`); their label votes add up.
- **Updates** for every confirmed object observed or merged into.

The best view of each object is the mask with the highest score times area. The view is cut from the colour JPEG
(`crop`) with the cleaned mask: its bounding box grown on every side by `crop.context_margin` of its width and height, the mask's
own pixels as they are and everything else as `crop.background` says: `dimmed` (the default) keeps the surroundings
at 0.35 of their brightness, so what the object stands on and how big it is stay readable; `blank` puts the object on
neutral grey. It goes out on `appeared` and `moved`, and on `updated` when a view `crop.min_improvement` times better
replaces it. The spatial agent describes and embeds each view it receives (see `nodes/dora-world-agent`); the object
map makes no network calls.

Keyframes, poses and instance masks are joined by frame id (`join`). Only `tracking` and `relocalized` keyframes are
observed: a `lost` one has no map pose, and a `restarted` one has a dead-reckoned pose that later pose graph updates
move, which the object map does not follow, so both are skipped.

Every threshold is in `ObjectMapSettings` (`settings`). Every decision is logged with `tracing`: appearances,
misses, disappearances, moves and merges at info; matches, candidates, dropped masks and absence votes at debug.

`synthetic` renders labelled boxes in a box room with perfect masks, for tests.
