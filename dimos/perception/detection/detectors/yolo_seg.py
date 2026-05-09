# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from ultralytics import YOLO  # type: ignore[attr-defined, import-not-found]

from dimos.perception.detection.detectors.types import Detector
from dimos.perception.detection.type import ImageDetections2D
from dimos.utils.data import get_data
from dimos.utils.gpu_utils import is_cuda_available
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.msgs.sensor_msgs import Image

logger = setup_logger()


class YoloSeg2DDetector(Detector):
    """YOLO segmentation detector returning Detection2DSeg masks when available."""

    def __init__(
        self,
        model_path: str = "models_yolo",
        model_name: str = "yolo11n-seg.pt",
        device: str | None = None,
        allow_download: bool = True,
    ) -> None:
        local_model = get_data(model_path) / model_name
        model_ref = local_model if local_model.exists() else model_name

        if not local_model.exists() and not allow_download:
            raise FileNotFoundError(f"YOLO segmentation model not found: {local_model}")
        if not local_model.exists():
            logger.warning(
                "YOLO segmentation model not found locally; ultralytics may try to download it",
                model=model_name,
            )

        self.model = YOLO(model_ref, task="segment")

        if device:
            self.device = device
            return

        if is_cuda_available():  # type: ignore[no-untyped-call]
            self.device = "cuda"
            logger.debug("Using CUDA for YOLO segmentation detector")
        else:
            self.device = "cpu"
            logger.debug("Using CPU for YOLO segmentation detector")

    def process_image(self, image: Image) -> ImageDetections2D:
        results = self.model.predict(
            source=image.to_opencv(),
            device=self.device,
            conf=0.5,
            iou=0.6,
            verbose=False,
        )

        return ImageDetections2D.from_ultralytics_result(image, results)

    def stop(self) -> None:
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None


__all__ = ["YoloSeg2DDetector"]
