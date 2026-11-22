# dora-serial-lidar

A [dora](https://dora-rs.ai) node, in Rust, for a low-cost 2D lidar on a serial port. It publishes every turn of the
head as one scan. Two lidars are supported, chosen by `LIDAR_MODEL`:

| Model | Lidar | Measures | Turn | Intensity | Reading times |
|---|---|---|---|---|---|
| `ld06` | LDRobot LD06 | time of flight, 0.02 m to 12 m | about 455 readings at 10 revolutions a second | 0 to 255 per reading | the lidar's own stamps, placed on the host's clock |
| `delta2a` | 3irobotics Delta-2A | triangulation, 0.15 m to 8 m | about 600 readings at 4 to 10 revolutions a second | none | when the bytes reached the host |

The workspace holds three crates. `ld06` and `delta2a` each decode one lidar's serial protocol and know nothing about
serial ports or dora: packets found in the byte stream however it is chopped up, checksums verified, readings
collected into turns, with bearings, ranges and the spin rate as typed quantities. `dora-serial-lidar` is the node
around them.

Both lidars only talk, at 230400 baud, 8N1. The node keeps trying to reopen the port when its USB adapter drops off
the bus.

## Wiring

The LD06 has four wires: red 5 V, black ground, green data out at 3.3 V, yellow motor speed in. Leave yellow
unconnected and the lidar holds its own 10 revolutions a second. Through a CP2102 USB adapter: red to `5V`, black to
`GND`, green to `RXD`. On a Raspberry Pi's header instead: red to pin 2, black to pin 6, green to pin 10 (`GPIO15`),
with `dtparam=uart0=on` in `config.txt`, giving `/dev/ttyAMA0` on a Pi 5.

The Delta-2A's adapter board has two USB sockets and the lidar needs both: one is the serial data, the other its 5 V
supply (0.6 A while the head spins up). While the lidar has no power the adapter stops answering USB control
requests, so opening and closing the port each take the kernel five seconds.

Cheap CP2102 adapters all report the serial number `0001`, so two of them share one `/dev/serial/by-id/` name. Give
the port by its `/dev/serial/by-path/` name, which follows the USB socket.

## Reading times

The LD06 stamps each packet of twelve readings with its own millisecond clock. A packet reaches the host some time
after its stamp, never before, so the node places stamps on the host's clock with the smallest gap between arrival
and stamp seen over the last ten seconds. The first turns after the port opens can be placed up to a tick late while
that gap settles. When the stamps break off, the node logs it and places the clock again from the next packet.

The Delta-2A stamps nothing, so its readings are timed by the poll that read their bytes, which is late by the
serial line and up to a tick.

| Input | What it carries |
|---|---|
| `tick` | A timer. Each tick reads the bytes that have arrived, so it must be much shorter than a turn of the head |

| Output | What it carries |
|---|---|
| `scan` | A scan from `dora-rig-messages`: the spin rate; when the turn's first and last reading were measured, in unix nanoseconds on the host's clock; bearings, ranges and intensities in step, with a null range where nothing came back and a null intensity list for a lidar without one. Bearings run clockwise from the zero mark on the lidar's body, seen from above, in the order measured. The first turn after the port opens is partial |

| Variable | Meaning |
|---|---|
| `LIDAR_MODEL` | `ld06` or `delta2a` |
| `LIDAR_SERIAL_DEVICE` | The serial port, best given by its `/dev/serial/by-path/` name |

```yaml
- id: lidar
  path: nodes/dora-serial-lidar/dist/dora-serial-lidar
  env:
    LIDAR_MODEL: ld06
    LIDAR_SERIAL_DEVICE: /dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0
  inputs:
    tick: dora/timer/millis/20
  outputs:
    - scan
```

`build-for-pi.sh` cross-compiles a static aarch64 binary into `dist/`, so nothing has to be compiled on the robot.
`cargo test` checks each decoder against a capture from a real unit, and the node's scans against the golden Arrow
files in `dora-rig-messages`.
