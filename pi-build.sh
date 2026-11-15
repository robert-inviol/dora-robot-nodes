#!/usr/bin/env bash
# Builds the camera machine's Rust nodes for the Raspberry Pi on strixen and copies them, with rsync, to a tree laid
# out like the repo, at the paths the Pi's dataflows name (each workspace's target/release). Run it on strixen from
# the repo root, into a staging directory, then copy that over the Pi's checkout:
#
#   ./pi-build.sh ~/pi-out
#   rsync -a strixen:pi-out/ raspai:src/dora-robot-nodes-dev/      (from a machine that reaches both)
#
# Needs, as the user: rustup's aarch64-unknown-linux-gnu target (for the default toolchain and for 1.90.0, which
# dora-zenoh-bridge pins), cargo-zigbuild, zig (`uv tool install ziglang`), pkgconf (`uv tool install pkgconf`), and
# the Pi's librealsense in ~/pi-sysroot/usr/local (lib/librealsense2.so*, include/librealsense2 and a
# lib/pkgconfig/realsense2.pc whose prefix is that directory). Binaries target the Pi's glibc, 2.36.
set -euo pipefail

destination="${1:?give where the binaries go, e.g. ~/pi-out}"
target=aarch64-unknown-linux-gnu
glibc=2.36

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_ZIGBUILD_ZIG_PATH="$HOME/.local/bin/python-zig"
export PKG_CONFIG="$HOME/.local/bin/pkgconf"
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH="$HOME/pi-sysroot/usr/local/lib/pkgconfig"
# librealsense2.so's own libraries (libusb, libudev, libstdc++) are on the Pi, not here.
export RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined"

# workspace directory, then the packages built there
builds=(
  "nodes/dora-realsense dora-realsense dora-keyframe-selector dora-posed-depth"
  "nodes/dora-depth-perception dora-depth-inertial-odometry dora-odometry-input-recorder"
  "nodes/dora-people-3d dora-people-3d"
  "nodes/dora-zenoh-bridge dora-zenoh-bridge"
)

for build in "${builds[@]}"; do
  read -r workspace packages <<<"$build"
  package_flags=()
  for package in $packages; do package_flags+=(-p "$package"); done
  echo "== $workspace: $packages"
  (cd "$workspace" && cargo zigbuild --release --target "$target.$glibc" "${package_flags[@]}")
  mkdir -p "$destination/$workspace/target/release"
  for package in $packages; do
    rsync -a "$workspace/target/$target/release/$package" "$destination/$workspace/target/release/"
  done
done
echo "copied to $destination"
