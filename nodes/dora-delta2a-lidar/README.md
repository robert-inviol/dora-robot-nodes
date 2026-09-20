# dora-delta2a-lidar

A [dora](https://dora-rs.ai) node, in Rust, for the 3irobotics Delta-2A: a low-cost 2D lidar that
measures by triangulation, 0.15 m to 8 m, about 600 points a revolution at 4 to 10 revolutions a
second. It publishes every turn of the head as one scan.

The workspace holds two crates. `delta2a` decodes the lidar's serial protocol and knows nothing
about serial ports or dora: frames found in the byte stream however it is chopped up, checksums
verified, sectors collected into revolutions, with bearings, ranges and the spin rate as typed
quantities. `dora-delta2a-lidar` is the node around it.

The lidar only talks (230400 baud, 8N1), usually through a CP2102 USB adapter. The adapter board
has two USB sockets and the lidar needs both: one is the serial data, the other its 5 V supply
(0.6 A while the head spins up). The node keeps trying to reopen the port when the adapter drops
off the bus, and logs when the lidar reports that its head is not up to speed. While the lidar has
no power the adapter stops answering USB control requests, so opening and closing the port each
take the kernel five seconds.

| Input | What it carries |
|---|---|
| `tick` | A timer. Each tick reads the bytes that have arrived, so it must be much shorter than a turn of the head |

| Output | What it carries |
|---|---|
| `scan` | A scan from `dora-rig-messages`: the spin rate, and bearings and ranges in step, with a null range where nothing came back. Bearings run clockwise from the zero mark on the lidar's body, seen from above |

| Variable | Meaning |
|---|---|
| `LIDAR_SERIAL_DEVICE` | The serial port, best given by its `/dev/serial/by-id/` path |

```yaml
- id: lidar
  path: nodes/dora-delta2a-lidar/dist/dora-delta2a-lidar
  env:
    LIDAR_SERIAL_DEVICE: /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0
  inputs:
    tick: dora/timer/millis/20
  outputs:
    - scan
```

`build-for-pi.sh` cross-compiles a static aarch64 binary into `dist/`, so nothing has to be
compiled on the robot. `cargo test` checks the decoder against a capture from a real unit, and the
node's scan against the golden Arrow file in `dora-rig-messages`.
