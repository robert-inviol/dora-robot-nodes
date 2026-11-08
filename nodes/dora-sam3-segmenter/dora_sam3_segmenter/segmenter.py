"""SAM 3 on one image at a time: the image is encoded once, then each object type is found in it."""

import io
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from dora_rig_messages.perception import NO_SEGMENT

MODEL_ID = "facebook/sam3"


@dataclass(frozen=True)
class SegmenterSettings:
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    # Two masks overlapping this much are the same object named twice; the higher score is kept.
    duplicate_iou: float = 0.7
    # Object types searched for in one pass; each extra one holds a copy of the image features on the GPU.
    batch_size: int = 1


@dataclass
class Detection:
    label: str
    score: float
    # Boolean mask at image size, on the GPU.
    mask: torch.Tensor


@dataclass
class EncodedImage:
    frame_id: int
    vision_embeds: object
    size_hw: tuple[int, int]
    # Object types already searched for in this image, with what each found.
    found: dict[str, list[Detection]] = field(default_factory=dict)


class Sam3Segmenter:
    def __init__(self, device: str, settings: SegmenterSettings) -> None:
        self.device = device
        self.settings = settings
        self.processor = Sam3Processor.from_pretrained(MODEL_ID)
        self.model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
        self._text_features: dict[str, tuple[object, torch.Tensor]] = {}

    @torch.inference_mode()
    def encode(self, frame_id: int, color_jpeg: bytes) -> EncodedImage:
        image = Image.open(io.BytesIO(color_jpeg)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        embeds = self.model.get_vision_features(pixel_values=inputs.pixel_values.to(torch.bfloat16))
        return EncodedImage(frame_id, embeds, (image.height, image.width))

    def _text(self, concept: str) -> tuple[object, torch.Tensor]:
        if concept not in self._text_features:
            tokens = self.processor(text=concept, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, return_dict=True)
            self._text_features[concept] = (features, tokens.attention_mask)
        return self._text_features[concept]

    @staticmethod
    def _repeated(embeds: Sam3VisionEncoderOutput, count: int) -> Sam3VisionEncoderOutput:
        if count == 1:
            return embeds
        return Sam3VisionEncoderOutput(
            last_hidden_state=embeds.last_hidden_state.expand(count, -1, -1),
            fpn_hidden_states=tuple(x.expand(count, -1, -1, -1) for x in embeds.fpn_hidden_states),
            fpn_position_encoding=tuple(x.expand(count, -1, -1, -1) for x in embeds.fpn_position_encoding),
        )

    @torch.inference_mode()
    def search(self, image: EncodedImage, concepts: list[str]) -> list[str]:
        """Searches for each concept not yet searched for in this image; returns the ones it searched for."""
        new = [c for c in concepts if c not in image.found]
        for start in range(0, len(new), self.settings.batch_size):
            batch = new[start : start + self.settings.batch_size]
            texts = [self._text(c) for c in batch]
            text_embeds = BaseModelOutputWithPooling(pooler_output=torch.cat([features.pooler_output for features, _ in texts]))
            outputs = self.model(
                vision_embeds=self._repeated(image.vision_embeds, len(batch)),
                text_embeds=text_embeds,
                attention_mask=torch.cat([mask for _, mask in texts]),
            )
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.settings.score_threshold,
                mask_threshold=self.settings.mask_threshold,
                target_sizes=[image.size_hw] * len(batch),
            )
            for concept, result in zip(batch, results):
                image.found[concept] = [
                    Detection(concept, float(score), mask.bool()) for score, mask in zip(result["scores"], result["masks"])
                ]
        return new

    @torch.inference_mode()
    def segment_ids(self, image: EncodedImage) -> tuple[list[str], list[float], np.ndarray]:
        """Every detection so far as one segment map: duplicates dropped, smaller masks painted over larger ones."""
        detections = sorted((d for found in image.found.values() for d in found), key=lambda d: -d.score)
        height, width = image.size_hw
        if not detections:
            return [], [], np.full((height, width), NO_SEGMENT, np.int16)
        masks = torch.stack([d.mask for d in detections]).reshape(len(detections), -1)
        flat = masks.to(torch.float16)
        areas = flat.sum(dim=1)
        overlap = flat @ flat.T
        iou = overlap / (areas[:, None] + areas[None, :] - overlap).clamp(min=1)
        kept: list[int] = []
        for index in range(len(detections)):
            if areas[index] > 0 and all(iou[index, k] < self.settings.duplicate_iou for k in kept):
                kept.append(index)
        ids = torch.full((height * width,), NO_SEGMENT, dtype=torch.int16, device=masks.device)
        for position in sorted(range(len(kept)), key=lambda p: -float(areas[kept[p]])):
            ids[masks[kept[position]]] = position
        return (
            [detections[k].label for k in kept],
            [detections[k].score for k in kept],
            ids.reshape(height, width).cpu().numpy(),
        )


def elapsed_ms(started: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e3
