#!/usr/bin/env bash
# Cross-compiles the lidar node into a static binary for the Pi and puts it where dataflow.yaml expects it.
set -euo pipefail
cd "$(dirname "$0")"

PI_TARGET=aarch64-unknown-linux-musl

rustup target add "$PI_TARGET"
cargo build --release --package dora-delta2a-lidar --target "$PI_TARGET"
install -D "target/$PI_TARGET/release/dora-delta2a-lidar" dist/dora-delta2a-lidar
