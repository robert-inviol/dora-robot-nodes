"""A dora node that runs another node's letterboxed `model_input` frames through YOLOv8 pose on a Hailo-8 with
`hailotracker`, and publishes every frame's people as `person_poses`.

Frames enter a GStreamer pipeline through `appsrc` and leave it through `appsink`, which the main loop empties;
Hailo's own post-process decodes the keypoints and its tracker numbers the people. A frame's id rides through the
pipeline as its buffer timestamp, one millisecond apart per frame. Nothing in the pipeline drops a frame: the node
gives it at most `FRAMES_IN_FLIGHT` at a time and skips a frame that arrives while that many are in flight.

The node stops the process when inference stops: when the oldest frame in flight has waited `INFERENCE_STALL_S`
for its result, and, from a native timer, when the main loop itself has not come round for `MAIN_LOOP_BACKSTOP_S`.
"""

import faulthandler
import os
import sys
import time

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # noqa: E402, F401  (GstApp gives appsink its pull methods)

import hailo  # noqa: E402
from dora import Node  # noqa: E402

from .in_flight import FramesInFlight, InferenceStalled  # noqa: E402
from .node_loop import EventKind, MainLoopBackstop, classify  # noqa: E402
from .poses import HailoPerson, ModelInputFrame, person_poses  # noqa: E402

DEFAULT_HEF = "/usr/share/hailo-models/yolov8s_pose_h8.hef"
DEFAULT_POST_PROCESS = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so"
MODEL_INPUT = "model_input"
PERSON_POSES = "person_poses"
# How long the main loop waits for an event before collecting what the pipeline has finished.
EVENT_WAIT_S = 0.005
# Buffer timestamps only identify frames; the pipeline does not pace on them.
PTS_PER_FRAME_NS = 1_000_000
FRAMES_IN_FLIGHT = 3
# Covers the Hailo's configuration on the first frame as well as one frame's inference.
INFERENCE_STALL_S = 5.0
MAIN_LOOP_BACKSTOP_S = INFERENCE_STALL_S + 2.0
MAIN_LOOP_REARM_EVERY_S = 1.0
STATUS_EVERY_S = 30.0
EXIT_INFERENCE_STALLED = 1


def read_people(buffer: Gst.Buffer) -> list[HailoPerson]:
    roi = hailo.get_roi_from_buffer(buffer)
    people = []
    for detection in hailo.get_hailo_detections(roi):
        box = detection.get_bbox()
        ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        points = tuple((p.x(), p.y(), p.confidence()) for p in landmarks[0].get_points())
        people.append(
            HailoPerson(
                track_id=ids[0].get_id() if ids else None,
                score=float(detection.get_confidence()),
                xmin=box.xmin(),
                ymin=box.ymin(),
                width=box.width(),
                height=box.height(),
                keypoints=points,
            )
        )
    return people


class HailoPosePipeline:
    def __init__(self, side: int, hef: str, post_process: str):
        Gst.init(None)
        self.side = side
        queue = f"queue max-size-buffers={FRAMES_IN_FLIGHT} max-size-bytes=0 max-size-time=0"
        self.pipeline = Gst.parse_launch(
            f"appsrc name=frames is-live=true format=time do-timestamp=false block=false max-bytes=0 "
            f"caps=video/x-raw,format=RGB,width={side},height={side},framerate=30/1 ! {queue} ! "
            # Python keeps a reference to every buffer it hands appsrc, so hailonet has to be told it may write anyway.
            f"hailonet hef-path={hef} batch-size=1 force-writable=true ! {queue} ! "
            f"hailofilter so-path={post_process} function-name=filter qos=false ! "
            "hailotracker name=tracker class-id=-1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.7 "
            "keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=false qos=false ! "
            f"appsink name=people emit-signals=false sync=false drop=false max-buffers={FRAMES_IN_FLIGHT}"
        )
        self.source = self.pipeline.get_by_name("frames")
        self.sink = self.pipeline.get_by_name("people")
        self.bus = self.pipeline.get_bus()

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("the Hailo pose pipeline would not start")

    def push(self, pts: int, pixels: np.ndarray) -> None:
        # hailonet writes into its input buffer, so the frame is copied into one GStreamer owns.
        buffer = Gst.Buffer.new_allocate(None, pixels.nbytes, None)
        buffer.fill(0, pixels.tobytes())
        buffer.pts = pts
        if self.source.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
            raise RuntimeError(f"the Hailo pose pipeline refused frame pts {pts}")

    def pull_finished(self) -> Gst.Buffer | None:
        """The next frame the pipeline has finished, or None when it has none ready."""
        sample = self.sink.try_pull_sample(0)
        return None if sample is None else sample.get_buffer()

    def raise_if_failed(self) -> None:
        message = self.bus.pop_filtered(Gst.MessageType.ERROR)
        if message is not None:
            error, debug = message.parse_error()
            raise RuntimeError(f"the Hailo pose pipeline failed: {error.message} ({debug})")

    def stop(self) -> None:
        self.source.emit("end-of-stream")
        self.pipeline.set_state(Gst.State.NULL)


class PoseNode:
    def __init__(self, node: Node, hef: str, post_process: str):
        self.node = node
        self.hef = hef
        self.post_process = post_process
        self.pipeline: HailoPosePipeline | None = None
        self.in_flight = FramesInFlight(FRAMES_IN_FLIGHT)
        self.frames_returned = 0
        self.frames_skipped = 0
        self.started_at_s = time.monotonic()
        self.status_at_s = self.started_at_s

    def give(self, frame: ModelInputFrame, pixels: np.ndarray, now_s: float) -> None:
        if self.pipeline is None:
            self.pipeline = HailoPosePipeline(frame.side, self.hef, self.post_process)
            self.pipeline.start()
            print(f"[hailo-pose] running {self.hef} on {frame.side}x{frame.side} model input", flush=True)
        elif frame.side != self.pipeline.side:
            raise ValueError(f"model input changed from {self.pipeline.side} to {frame.side} pixels square")
        if not self.in_flight.has_room():
            self.frames_skipped += 1
            return
        pts = frame.frame_id * PTS_PER_FRAME_NS
        self.in_flight.give(pts, frame, now_s)
        self.pipeline.push(pts, pixels)

    def send_finished(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.raise_if_failed()
        while (buffer := self.pipeline.pull_finished()) is not None:
            frame = self.in_flight.take(buffer.pts)
            people = read_people(buffer)
            self.node.send_output(PERSON_POSES, person_poses(frame, people).to_arrow(), {"people": len(people)})
            self.frames_returned += 1
            if self.frames_returned == 1:
                print(f"[hailo-pose] first result {time.monotonic() - self.started_at_s:.1f} s after start", flush=True)

    def report(self, now_s: float) -> None:
        if now_s - self.status_at_s >= STATUS_EVERY_S:
            self.status_at_s = now_s
            print(self.status(now_s), flush=True)

    def status(self, now_s: float) -> str:
        return (
            f"[hailo-pose] {self.frames_returned} frames returned, {self.frames_skipped} skipped with "
            f"{FRAMES_IN_FLIGHT} in flight, after {now_s - self.started_at_s:.0f} s"
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


def run(pose: PoseNode, backstop: MainLoopBackstop) -> None:
    while True:
        now_s = time.monotonic()
        backstop.keep_alive(now_s)
        event = pose.node.next(timeout=EVENT_WAIT_S)
        kind = classify(event, MODEL_INPUT)
        if kind is EventKind.END:
            return
        if kind is EventKind.FRAME:
            frame = ModelInputFrame.from_metadata(event["metadata"])
            pose.give(frame, event["value"].to_numpy(zero_copy_only=False), time.monotonic())
        pose.send_finished()
        now_s = time.monotonic()
        pose.in_flight.raise_if_stalled(now_s, INFERENCE_STALL_S)
        pose.report(now_s)


def main():
    pose = PoseNode(
        Node(),
        hef=os.environ.get("POSE_HEF", DEFAULT_HEF),
        post_process=os.environ.get("POSE_POST_PROCESS", DEFAULT_POST_PROCESS),
    )
    backstop = MainLoopBackstop(MAIN_LOOP_BACKSTOP_S, MAIN_LOOP_REARM_EVERY_S)
    try:
        run(pose, backstop)
    except InferenceStalled as stalled:
        # Threads of a stalled pipeline can be stuck in the kernel, so stopping it could hang: the process ends here.
        print(f"[hailo-pose] inference stalled: {stalled}; every thread's stack follows", file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        os._exit(EXIT_INFERENCE_STALLED)
    finally:
        print(pose.status(time.monotonic()), flush=True)
        # The backstop stays armed until the pipeline has stopped, so a teardown that hangs still ends the process.
        pose.stop()
        backstop.disarm()


if __name__ == "__main__":
    main()
