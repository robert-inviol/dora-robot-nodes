# dora-scan-view

A [dora](https://dora-rs.ai) node that shows a planar lidar's scan as a top-down map in a browser,
for a phone or a laptop: range rings, bearings clockwise from the lidar's zero mark, neighbouring
returns on one surface joined by a line, scale buttons, and a range and bearing readout under the
pointer.

Raw scans are too much and too jittery to stream to a page, so the node steadies them first. It
keeps the nearest return per degree for the last eight revolutions and shows a bin only when at
least three of them saw something there, at the median range. The page gets that twice a second,
about two kilobytes a time, and says so when the lidar has gone quiet for a second.

The drawing code is one script, `/map.js`, which other pages can load to embed the map:
`LidarMap.attach(container, { socketUrl, onStatus })`. `dora-teleop` does this to show the map
beside its video. Below thumbnail size the map drops its labels and controls.

| Input | What it carries |
|---|---|
| `scan` | A scan from `dora-rig-messages`: one turn of the lidar's head |

The node has no outputs.

| Variable | Meaning |
|---|---|
| `SCANVIEW_PORT` | Port of the map page, `/map.js` and the `/plane` socket |

```yaml
- id: scan-view
  path: dora-scan-view
  env:
    SCANVIEW_PORT: "8082"
  inputs:
    scan: lidar/scan
```

`/plane` is a WebSocket that sends one JSON object per refresh: `live`, `spin_rev_per_s`,
`bin_deg`, and `ranges_mm`, one entry per bin starting at bearing 0 and running clockwise, null
where the lidar does not steadily see anything.
