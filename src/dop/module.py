from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Type

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.ocr_utils import (
    map_tesseract_script,
    parse_tesseract_orientation,
    tesseract_box_to_bounding_rectangle,
)
from docling.utils.profiling import TimeRecorder
from pydantic import ConfigDict
from typing import ClassVar, Literal

_log = logging.getLogger(__name__)


class DeepSeekOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: DeepSeekOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: DeepSeekOcrOptions
        self._is_auto: bool = "auto" in self.options.lang
        self.scale = 3  # multiplier for 72 dpi == 216 dpi.
        self.reader = None
        self.script_readers: dict[str, tesserocr.PyTessBaseAPI] = {}

        if self.enabled:
            install_errmsg = (
                "deepseek-ocr is not correctly installed. "
                "Note that deepseek-ocr might have to be manually compiled for working with "
                "your DeepSeek OCR installation. The Docling documentation provides examples for it. "
                "Alternatively, Docling has support for other OCR engines. See the documentation: "
                "https://docling-project.github.io/docling/installation/"
            )

            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                import os
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'
                model_name = 'deepseek-ai/DeepSeek-OCR'
            except ImportError:
                raise ImportError(install_errmsg)

            # Initialize the DeepSeek OCR model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
            self.model = model.eval().cuda().to(torch.bfloat16)

            # prompt = "<image>\nFree OCR. "
            self.prompt = "<image>\n<|grounding|>Convert the document to markdown. "


    def __del__(self):
        pass
    
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page_i, page in enumerate(page_batch):
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):

                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        text = model.infer(high_res_image, prompt=self.prompt)
                        # del high_res_image
                        all_ocr_cells.append(TextCell(
                            index=0,
                            text=text,
                            orig=text,
                            from_ocr=True,
                            confidence=1.0,
                            rect=ocr_rect,
                        ))

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return OcrDeepSeekOcrOptions

class OcrDeepSeekOcrOptions(OcrOptions):
    """Options for the DeepSeek OCR engine."""

    kind: ClassVar[Literal["deepseek-ocr"]] = "deepseek-ocr"
    lang: list[str] = []

    model_config = ConfigDict(
        extra="forbid",
    )

# Factory registration
def ocr_engines():
    return {
        "ocr_engines": [
            DeepSeekOcrModel,
        ]
    }
    
def deepseek_ocr():
    return {
        "deepseek-ocr": [
            DeepSeekOcrModel     
        ],
    }