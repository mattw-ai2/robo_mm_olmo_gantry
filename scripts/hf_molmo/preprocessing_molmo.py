"""
Processor class for Molmo.
"""
from typing import List, Optional, Union, Any, Tuple, Dict

import PIL
from PIL import ImageFile, ImageOps, Image

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import torch

from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    TextKwargs,
    ProcessingKwargs,
    ProcessorMixin,
)

from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

from transformers import AutoTokenizer
from .image_preprocessing_molmo import MolmoImagesKwargs, MolmoImageProcessor


logger = logging.get_logger(__name__)


# Special tokens, these should be present in any tokenizer we use since the preprocessor uses them
IMAGE_PATCH_TOKEN = f"<im_patch>"  # Where to insert high-res tokens
IMAGE_LOW_RES_TOKEN = f"<im_low>"  # Where to insert low-res tokens
IM_START_TOKEN = f"<im_start>"
IM_END_TOKEN = f"<im_end>"
IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"

EXTRA_TOKENS = (IM_START_TOKEN, IM_END_TOKEN, IMAGE_PATCH_TOKEN,
                IM_COL_TOKEN, IMAGE_PROMPT, IMAGE_LOW_RES_TOKEN)


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_special_token_ids(tokenizer: AutoTokenizer) -> Dict[str, int]:
    ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


def load_image(image: Union[PIL.Image.Image, np.ndarray]) -> np.ndarray:
    """Load image"""
    setup_pil()
    if isinstance(image, PIL.Image.Image):
        image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)
        return np.array(image)
    elif isinstance(image, np.ndarray):
        assert len(image.shape) == 3, "Image should have 3 dimensions"
        assert image.shape[2] == 3, "Image should have 3 channels"
        assert image.dtype == np.uint8, "Image should have uint8 type"
        return image
    else:
        raise ValueError("Image should be PIL.Image or np.ndarray")


class MolmoProcessorKwargs(ProcessingKwargs, total=False):
    """Molmo processor kwargs"""
    images_kwargs: MolmoImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class MolmoProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    optional_attributes = ["chat_template", "prompt_templates", "message_format", "system_prompt", "always_start_with_space", "default_inference_len"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor: MolmoImageProcessor=None,
        tokenizer: AutoTokenizer=None,
        prompt_templates: Optional[str] = "uber_model",
        message_format: Optional[str] = "role",
        system_prompt: Optional[str] = "demo_or_style",
        always_start_with_space: Optional[bool] = False,
        default_inference_len: Optional[int] = 65,
        **kwargs
    ) -> None:
        super().__init__(
            image_processor,
            tokenizer,
            prompt_templates=prompt_templates,
            message_format=message_format,
            system_prompt=system_prompt,
            always_start_with_space=always_start_with_space,
            default_inference_len=default_inference_len,
        )
        self._special_tokens = None
        # Add audio_tokenizer attribute for compatibility with newer transformers library
        self.audio_tokenizer = None

    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens
    
    def get_user_prompt(self, text: TextInput) -> str:
        """Get user prompt"""
        if self.prompt_templates == "none":
            return ""
        elif self.prompt_templates == "uber_model":
            return text
        else:
            raise NotImplementedError(self.prompt_templates)
    
    def get_prefix(self) -> str:
        """Get prefix"""
        if self.system_prompt == "style_and_length":  # captioner
            style = "long_caption"
            n = None if self.default_inference_len is None else str(self.default_inference_len)
            if n is not None and len(n) > 0:  # allow empty string to signal unconditioned
                prefix = style + " " + n + ":"
            else:
                prefix = style + ":"
        elif self.system_prompt == "demo_or_style":  # demo model
            style = "demo"
            prefix = ""
        else:
            raise NotImplementedError(self.system_prompt)
        return prefix
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt"""
        if self.message_format == "none":
            pass
        elif self.message_format == "role":
            prompt = "User: " + prompt + " Assistant:"
        else:
            raise NotImplementedError(self.message_format)
        
        if self.always_start_with_space:
            prompt = " " + prompt
        
        return prompt
    
    def get_prompt(self, text: TextInput) -> str:
        prompt = self.get_user_prompt(text)
        prefix = self.get_prefix()
        if len(prefix) > 0 and len(prompt) > 0:
            prompt = prefix + " " + prompt
        elif len(prefix) > 0:
            prompt = prefix
        prompt = self.format_prompt(prompt)
        return prompt          

    def tokenize_prompt(self, prompt: str) -> np.ndarray:
        """Tokenize prompt"""
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        text_token_ids = [bos] + self.tokenizer.encode(prompt, add_special_tokens=False)
        return text_token_ids
    
    def tokenize_and_interleave(
        self,
        text_token_ids: np.ndarray,
        multi_model_tokens: List[np.ndarray],
    ) -> np.ndarray:
        """Tokenize and interleave text"""
        image_prompt_token_id = self.special_token_ids[IMAGE_PROMPT]

        mm_idx = [1] * len(multi_model_tokens)

        mm_tokens = []
        on = 0
        for i, token_ix in enumerate(mm_idx):
            mm_tokens.append(text_token_ids[on:token_ix])
            vision_tokens = multi_model_tokens[i]
            mm_tokens.append(vision_tokens)
            if text_token_ids[token_ix] == image_prompt_token_id:
                on = token_ix + 1  # Skip over the image prompt token
            else:
                on = token_ix
        
        mm_tokens.append(text_token_ids[on:])
        mm_tokens = np.concatenate(mm_tokens)

        return mm_tokens

    def process(
        self,
        text: TextInput = None,
        images: ImageInput = None,
        *,
        tokens: Optional[PreTokenizedInput] = None,
        **kwargs: Unpack[MolmoProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            MolmoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if tokens is None:
            prompt = self.get_prompt(text)
            tokens = self.tokenize_prompt(prompt)

        tokens = np.array(tokens)
        
        if images is None or (
            isinstance(images, (list, tuple)) and len(images) == 0
        ):
            input_ids = self.tokenize_and_interleave(tokens, [])
            out = {"input_ids": input_ids}
        
        else:
            if not isinstance(images, (list, tuple)):
                images = [images]
            images = [load_image(image) for image in images]

            image_patch_token_id = self.special_token_ids[IMAGE_PATCH_TOKEN]
            image_col_token_id = self.special_token_ids[IM_COL_TOKEN]
            image_start_token_id = self.special_token_ids[IM_START_TOKEN]
            image_end_token_id = self.special_token_ids[IM_END_TOKEN]

            all_image_tokens = []
            all_crop_masks = []
            all_crops = []
            pooled_patches_idx = []
            for idx, image in enumerate(images):
                image_tokens, crops, img_mask, pooled_idx = self.image_processor.preprocess(
                    image,
                    image_patch_token_id,
                    image_col_token_id,
                    image_start_token_id,
                    image_end_token_id,
                    **output_kwargs["images_kwargs"],
                )
                pooled_patches_idx.append(pooled_idx + sum(np.prod(x.shape[:2]) for x in all_crops))
                all_crops.append(crops)
                if len(images) > 1:
                    # Add prefix to the image tokens if there are multiple images
                    prefix = f"Image {idx + 1}"
                    image_tokens = np.concatenate([self.tokenizer.encode(prefix), image_tokens], 0)
                all_image_tokens.append(image_tokens)
                all_crop_masks.append(img_mask)
        
            input_ids = self.tokenize_and_interleave(tokens, all_image_tokens)
            out = {
                "input_ids": input_ids,
                "images": np.concatenate(all_crops),
                "pooled_patches_idx": np.concatenate(pooled_patches_idx)
            }
            image_padding_mask = self.image_processor.image_padding_mask
            if "image_padding_mask" in output_kwargs["images_kwargs"]:
                image_padding_mask = output_kwargs["images_kwargs"]["image_padding_mask"]
            if image_padding_mask:
                out["image_masks"] = np.concatenate(all_crop_masks)
        
        for k, v in out.items():
            out[k] = torch.from_numpy(v)

        return out


MolmoProcessor.register_for_auto_class()
