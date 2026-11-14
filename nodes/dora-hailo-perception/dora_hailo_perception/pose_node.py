"""A dora node that runs another node's letterboxed `model_input` frames through YOLOv8 pose on a Hailo-8 with
`hailotracker`, and publishes every frame's people as `person_poses`.

Frames enter a GStreamer pipeline through `appsrc`; Hailo's own post-process decodes the keypoints and its tracker
numbers the people. A frame's id rides through the pipeline as its buffer timestamp, one millisecond apart per frame.
"""

import os
import queue
import threading

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from dora import Node  # noqa: E402

from .poses import HailoPerson, ModelInputFrame, person_poses  # noqa: E402

DEFAULT_HEF = "/usr/share/hailo-models/yolov8s_pose_h8.hef"
DEFAULT_POST_PROCESS = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so"
MODEL_INPUT = "model_input"
PERSON_POSES = "person_poses"
# How long the main loop waits for an event before sending what the pipeline has finished.
EVENT_WAIT_S = 0.005
# Buffer timestamps only identify frames; the pipeline does not pace on them.
PTS_PER_FRAME_NS = 1_000_000
# Frames still in the pipeline; older ones were dropped by its leaky queues.
FRAMES_IN_FLIGHT = 30


def read_people(buffer) -> list[HailoPerson]:
    import hailo

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
    def __init__(self, side: int, hef: str, post_process: str, on_people):
        Gst.init(None)
        self._on_people = on_people
        self.pipeline = Gst.parse_launch(
            f"appsrc name=frames is-live=true format=time do-timestamp=false block=false "
            f"caps=video/x-raw,format=RGB,width={side},height={side},framerate=30/1 ! "
            "queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            # Python keeps a reference to every buffer it hands appsrc, so hailonet has to be told it may write anyway.
            f"hailonet hef-path={hef} batch-size=1 force-writable=true ! "
            "queue leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            f"hailofilter so-path={post_process} function-name=filter qos=false ! "
            "hailotracker name=tracker class-id=-1 kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.7 "
            "keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=false qos=false ! "
            "identity name=tap signal-handoffs=true ! fakesink sync=false"
        )
        self.source = self.pipeline.get_by_name("frames")
        self.pipeline.get_by_name("tap").connect("handoff", self._handoff)
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._error)
        self.failure: str | None = None

    def _handoff(self, _identity, buffer):
        self._on_people(buffer.pts, read_people(buffer))

    def _error(self, _bus, message):
        error, debug = message.parse_error()
        self.failure = f"{error.message} ({debug})"
        self.loop.quit()

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("the Hailo pose pipeline would not start")
        threading.Thread(target=self.loop.run, daemon=True).start()

    def push(self, pts: int, pixels: np.ndarray) -> None:
        # hailonet writes into its input buffer, so the frame is copied into one GStreamer owns.
        buffer = Gst.Buffer.new_allocate(None, pixels.nbytes, None)
        buffer.fill(0, pixels.tobytes())
        buffer.pts = pts
        self.source.emit("push-buffer", buffer)

    def stop(self) -> None:
        self.source.emit("end-of-stream")
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()


def main():
    node = Node()
    finished: queue.Queue = queue.Queue()
    in_flight: dict[int, ModelInputFrame] = {}
    pipeline: HailoPosePipeline | None = None
    hef = os.environ.get("POSE_HEF", DEFAULT_HEF)
    post_process = os.environ.get("POSE_POST_PROCESS", DEFAULT_POST_PROCESS)
    try:
        while True:
            event = node.next(timeout=EVENT_WAIT_S)
            if event is not None:
                if event["type"] == "STOP":
                    break
                if event["type"] == "INPUT" and event["id"] == MODEL_INPUT:
                    frame = ModelInputFrame.from_metadata(event["metadata"])
                    if pipeline is None:
                        pipeline = HailoPosePipeline(frame.side, hef, post_process, lambda pts, people: finished.put((pts, people)))
                        pipeline.start()
                        print(f"[hailo-pose] running {hef} on {frame.side}x{frame.side} model input", flush=True)
                    pts = frame.frame_id * PTS_PER_FRAME_NS
                    in_flight[pts] = frame
                    for stale in sorted(in_flight)[:-FRAMES_IN_FLIGHT]:
                        del in_flight[stale]
                    pipeline.push(pts, event["value"].to_numpy(zero_copy_only=False))
            if pipeline is not None and pipeline.failure:
                raise RuntimeError(f"the Hailo pose pipeline failed: {pipeline.failure}")
            while not finished.empty():
                pts, people = finished.get_nowait()
                frame = in_flight.pop(pts, None)
                if frame is None:
                    raise RuntimeError(f"the Hailo pose pipeline returned a frame it was never given (pts {pts})")
                node.send_output(PERSON_POSES, person_poses(frame, people).to_arrow(), {"people": len(people)})
    finally:
        if pipeline is not None:
            pipeline.stop()


if __name__ == "__main__":
    main()
