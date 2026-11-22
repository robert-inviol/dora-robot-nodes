# object-map

Persistent objects from SAM 3 masks, in the map frame (z up, gravity aligned, metres).

Each keyframe arrives with its instance masks and its pose from the localizer (`ObservedFrame`). For every mask:

1. `depth-perception`'s `object_footprint` erodes the mask, keeps the depth band of the object's own surface and
   fits the footprint rectangle and height band; the mask's points inside that rectangle and band are moved into
   the map with the keyframe pose and counted in 5 cm voxels (`detection`).
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

The best view of each object is the mask with the highest score times area; its bounding box is cut from the colour
JPEG (`crop`). It goes out on `appeared` and `moved`, and on `updated` when a view `crop.min_improvement` times
better replaces it.

Keyframes, poses and instance masks are joined by frame id (`join`). Only `tracking` and `relocalized` keyframes are
observed: a `lost` one has no map pose, and a `restarted` one has a dead-reckoned pose that later pose graph updates
move, which the object map does not follow, so both are skipped.

Every threshold is in `ObjectMapSettings` (`settings`). Every decision is logged with `tracing`: appearances,
misses, disappearances, moves and merges at info; matches, candidates, dropped masks and absence votes at debug.

`synthetic` renders labelled boxes in a box room with perfect masks, for tests.
