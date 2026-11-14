# object-map

Persistent objects from SAM 3 masks, in the map frame (z up, gravity aligned, metres).

Each keyframe arrives with its segments and its pose from the localizer (`ObservedFrame`). For every mask:

1. `depth-perception`'s `object_footprint` erodes the mask, keeps the depth band of the object's own surface and
   fits the footprint rectangle and height band; the mask's points inside that rectangle and band are moved into
   the map with the keyframe pose and counted in 5 cm voxels (`detection`).
2. The detection belongs to the object whose label agrees (`label`: same words, ignoring case and spacing) and whose
   voxels it touches most, if at least `association.min_overlap` of its voxels touch (one voxel of slack, `voxels`).
   Otherwise it starts a candidate.

Then, per keyframe, in order (`map`):

- **Absences.** Each confirmed object not sighted is projected into the keyframe. Voxels where the measured depth is
  within `absence.surface_margin_m` see the surface; farther behind sees past it; nearer or no reading is hidden.
  Enough surface resets the misses; mostly seen-past counts a miss; `misses_to_disappear` misses in a row and the
  object `disappeared` (`evidence`).
- **Confirmation.** A candidate sighted in `confirmation.sightings` keyframes within `window_s` `appeared`. If an
  object with an agreeing label disappeared within `absence.moved_window_s`, the candidate takes its id and the
  object `moved`.
- **Merging**, every `merge.interval_s` of capture time: confirmed objects whose labels agree, where the smaller one's
  voxels touch the larger by `merge.min_overlap`, merge into the older id (`merged` with `merged_into`).
- **Updates** for every confirmed object sighted or merged into.

The best view of each object is the mask with the highest score times area; its bounding box is cut from the colour
JPEG (`crop`). It goes out on `appeared` and `moved`, and on `updated` when a view `crop.min_improvement` times
better replaces it.

Keyframes, poses and segments are joined by frame id (`join`); frames the localizer lost are skipped.

Every threshold is in `ObjectMapSettings` (`settings`). Every decision is logged with `tracing`: appearances,
misses, disappearances, moves and merges at info; matches, candidates, dropped masks and absence tallies at debug.

`synthetic` renders labelled boxes in a box room with perfect masks, for tests.
