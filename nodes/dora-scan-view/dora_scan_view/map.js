// The lidar map: draws the scan plane that the scan-view node feeds into any positioned container.
// Pages load this from the scan-view server and call LidarMap.attach.
(function () {
"use strict";

const RECONNECT_AFTER_MS = 1000;
const MINIMUM_RANGE_M = 0.15;
// Half-widths the map can be set to, in metres. FIT picks the smallest that holds nearly every return.
const SCALES_M = [1, 2, 4, 8, 16];
const FIT = "FIT";
const FIT_SHARE_OF_RETURNS = 0.98;
// Neighbouring bins this close in range are the same surface, so a line joins them.
const SAME_SURFACE_SHARE = 0.06;
const SAME_SURFACE_SLACK_M = 0.03;
// A map smaller than this is a thumbnail: no labels, no scale buttons, no pointer readout.
const COMPACT_BELOW_PX = 260;

const STYLE = `
  .lidar-map { position:absolute; inset:0; background:#0b0e13; overflow:hidden;
               --map-grid:#232c38; --map-dim:#8b97a7; --map-text:#e8edf3; --map-points:#3987e5;
               --map-panel:#1b222c; --map-line:#2c3644; }
  .lidar-map canvas { position:absolute; inset:0; width:100%; height:100%; }
  .lidar-map-scales { position:absolute; left:8px; top:8px; display:flex; gap:4px; }
  .lidar-map-scales button { font:700 12px system-ui, sans-serif; color:var(--map-dim); min-width:44px; min-height:30px;
               letter-spacing:0; padding:0 6px; border:1px solid var(--map-line); border-radius:8px;
               background:var(--map-panel); cursor:pointer; }
  .lidar-map-scales button.on { color:var(--map-text); border-color:var(--map-dim); }
  .lidar-map-probe { position:absolute; left:10px; bottom:8px; color:var(--map-dim); pointer-events:none;
               font:12px system-ui, sans-serif; font-variant-numeric:tabular-nums; }
  .lidar-map.compact .lidar-map-scales, .lidar-map.compact .lidar-map-probe { display:none; }
`;

function attach(container, { socketUrl, onStatus = () => {} }) {
  if (!document.getElementById("lidar-map-style")) {
    const style = document.createElement("style");
    style.id = "lidar-map-style";
    style.textContent = STYLE;
    document.head.append(style);
  }
  const root = document.createElement("div");
  root.className = "lidar-map";
  const canvas = document.createElement("canvas");
  const scales = document.createElement("div");
  scales.className = "lidar-map-scales";
  const probe = document.createElement("div");
  probe.className = "lidar-map-probe";
  root.append(canvas, scales, probe);
  container.append(root);

  const context = canvas.getContext("2d");
  const colour = name => getComputedStyle(root).getPropertyValue(`--map-${name}`).trim();
  const colours = { grid: colour("grid"), dim: colour("dim"), text: colour("text"), points: colour("points") };

  let plane = { live: false, spin_rev_per_s: null, bin_deg: 1, ranges_mm: [] };
  let chosenScale = FIT;
  let pointer = null;

  function halfWidthM() {
    if (chosenScale !== FIT) return chosenScale;
    const ranges = plane.ranges_mm.filter(range => range !== null).sort((a, b) => a - b);
    if (ranges.length === 0) return SCALES_M[2];
    const reach = ranges[Math.floor((ranges.length - 1) * FIT_SHARE_OF_RETURNS)] / 1000;
    return SCALES_M.find(scale => scale >= reach) ?? SCALES_M.at(-1);
  }

  function ringStepM(halfWidth) {
    return halfWidth <= 1 ? 0.25 : halfWidth <= 2 ? 0.5 : halfWidth <= 8 ? 1 : 2;
  }

  function draw() {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (width === 0 || height === 0) return;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const compact = Math.min(width, height) < COMPACT_BELOW_PX;
    root.classList.toggle("compact", compact);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const halfWidth = halfWidthM();
    const centreX = width / 2, centreY = height / 2;
    const margin = compact ? 4 : 14;
    const pixelsPerMetre = (Math.min(width, height) / 2 - margin) / halfWidth;
    const place = (bearingDeg, rangeM) => {
      const bearing = bearingDeg * Math.PI / 180;
      return [centreX + Math.sin(bearing) * rangeM * pixelsPerMetre, centreY - Math.cos(bearing) * rangeM * pixelsPerMetre];
    };
    const frame = { place, halfWidth, centreX, centreY, pixelsPerMetre, compact,
                    cornerM: Math.hypot(width, height) / 2 / pixelsPerMetre };

    drawGrid(frame);
    drawPlane(frame);
    drawLidar(frame);
    describePointer(frame);
  }

  function drawGrid({ place, halfWidth, cornerM, centreX, centreY, pixelsPerMetre, compact }) {
    const step = ringStepM(halfWidth);
    context.lineWidth = 1;
    context.strokeStyle = colours.grid;
    for (let ring = step; ring <= cornerM; ring += step) {
      context.beginPath();
      context.arc(centreX, centreY, ring * pixelsPerMetre, 0, Math.PI * 2);
      context.stroke();
    }
    for (let bearing = 0; bearing < 360; bearing += 45) {
      context.beginPath();
      context.moveTo(centreX, centreY);
      context.lineTo(...place(bearing, cornerM));
      context.stroke();
    }
    if (compact) return;
    context.fillStyle = colours.dim;
    context.font = "11px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    // Ring labels run down the 202.5 degree line, clear of the spokes.
    for (let ring = step; ring <= halfWidth; ring += step) {
      const [x, y] = place(202.5, ring);
      context.fillText(`${ring} m`, x, y);
    }
    for (let bearing = 0; bearing < 360; bearing += 45) {
      const [x, y] = place(bearing, halfWidth * 0.97);
      context.fillText(`${bearing}°`, x, y);
    }
  }

  function drawPlane({ place, compact }) {
    const ranges = plane.ranges_mm;
    const spots = ranges.map((range, bin) => range === null ? null : place((bin + 0.5) * plane.bin_deg, range / 1000));

    context.strokeStyle = colours.points;
    context.lineWidth = compact ? 1.5 : 2;
    context.lineCap = "round";
    context.beginPath();
    spots.forEach((spot, bin) => {
      const next = (bin + 1) % spots.length;
      if (spot === null || spots[next] === null) return;
      const nearer = Math.min(ranges[bin], ranges[next]) / 1000;
      const apart = Math.abs(ranges[bin] - ranges[next]) / 1000;
      if (apart > nearer * SAME_SURFACE_SHARE + SAME_SURFACE_SLACK_M) return;
      context.moveTo(...spot);
      context.lineTo(...spots[next]);
    });
    context.stroke();

    context.fillStyle = colours.points;
    for (const spot of spots) {
      if (spot === null) continue;
      context.beginPath();
      context.arc(spot[0], spot[1], compact ? 1.2 : 2, 0, Math.PI * 2);
      context.fill();
    }
  }

  function drawLidar({ centreX, centreY, pixelsPerMetre, compact }) {
    context.strokeStyle = colours.dim;
    context.lineWidth = 1;
    context.setLineDash([3, 3]);
    context.beginPath();
    context.arc(centreX, centreY, MINIMUM_RANGE_M * pixelsPerMetre, 0, Math.PI * 2);
    context.stroke();
    context.setLineDash([]);
    // The arrowhead points along the lidar's zero bearing.
    const size = compact ? 0.6 : 1;
    context.fillStyle = colours.text;
    context.beginPath();
    context.moveTo(centreX, centreY - 9 * size);
    context.lineTo(centreX + 6 * size, centreY + 6 * size);
    context.lineTo(centreX - 6 * size, centreY + 6 * size);
    context.closePath();
    context.fill();
  }

  function describePointer({ centreX, centreY, pixelsPerMetre }) {
    if (pointer === null) { probe.textContent = ""; return; }
    const east = (pointer.x - centreX) / pixelsPerMetre, north = (centreY - pointer.y) / pixelsPerMetre;
    const bearing = (Math.atan2(east, north) * 180 / Math.PI + 360) % 360;
    const seen = plane.ranges_mm[Math.floor(bearing / plane.bin_deg)];
    const sees = seen == null ? "nothing seen that way" : `lidar sees ${(seen / 1000).toFixed(2)} m that way`;
    probe.textContent = `pointer ${Math.hypot(east, north).toFixed(2)} m at ${bearing.toFixed(0)}°, ${sees}`;
  }

  function report(linked) {
    onStatus({ linked, live: linked && plane.live, spinRevPerS: plane.spin_rev_per_s });
  }

  function connect() {
    const socket = new WebSocket(socketUrl);
    socket.addEventListener("message", message => {
      plane = JSON.parse(message.data);
      report(true);
      draw();
    });
    socket.addEventListener("close", () => {
      plane = { ...plane, live: false, ranges_mm: [] };
      report(false);
      draw();
      setTimeout(connect, RECONNECT_AFTER_MS);
    });
  }

  for (const scale of [FIT, ...SCALES_M]) {
    const button = document.createElement("button");
    button.textContent = scale === FIT ? FIT : `${scale} m`;
    button.classList.toggle("on", scale === chosenScale);
    button.addEventListener("click", event => {
      // The click is for the map's scale, not for whatever the page does with clicks on the container.
      event.stopPropagation();
      chosenScale = scale;
      for (const other of scales.children) other.classList.toggle("on", other === button);
      draw();
    });
    scales.append(button);
  }
  canvas.addEventListener("pointermove", event => { pointer = { x: event.offsetX, y: event.offsetY }; draw(); });
  canvas.addEventListener("pointerleave", () => { pointer = null; draw(); });
  new ResizeObserver(draw).observe(canvas);
  report(false);
  connect();
}

window.LidarMap = { attach };
})();
