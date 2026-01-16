"""Image processor class for Molmo"""
from typing import Tuple, List, Optional

import numpy as np
import einops
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    is_valid_image,
)
from transformers.processing_utils import ImagesKwargs
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import logging


logger = logging.get_logger(__name__)


def normalize_image(image: np.ndarray, normalize_mode: str) -> np.ndarray:
    if normalize_mode == "openai":
        image -= np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
        image /= np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
    elif normalize_mode == "siglip":
        image = np.asarray(-1.0, dtype=np.float32) + image * np.asarray(2.0, dtype=np.float32)
    elif normalize_mode == "dino":
        image -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
        image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    else:
        raise NotImplementedError(normalize_mode)
    return image


def resize_and_pad(
    image,
    desired_output_size,
    resize_method="torch-bilinear",
    pad_value=0,
):
    """Resize an image while padding to preserve uts aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method in ["torch-bilinear"]:
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        mode = InterpolationMode.BILINEAR
        image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask


def metaclip_resize(image, desired_output_size):
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    if torch.is_floating_point(image):
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
    else:
        assert image.dtype == torch.uint8, "Expected float images or uint8 images, but got {}".format(image.dtype)
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 255)
        image = image / 255.0
    resized = torch.permute(image, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    return resized, image_mask


def siglip_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
        image_mask = None
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()
        image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def dino_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
    else:
        assert image.dtype == torch.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0, 255).to(torch.float32)
        resized = resized / 255.0

    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def resize_image(
    image: np.ndarray,
    resize_mode: str,
    output_size: Tuple[int, int],
    pad_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if resize_mode == "siglip":
        return siglip_resize_and_pad(image, output_size)
    elif resize_mode == "dino":
        return dino_resize_and_pad(image, output_size)
    elif resize_mode == "metaclip":
        return metaclip_resize(image, output_size)
    else:
        resize = "torch-bilinear" if resize_mode == "default" else resize_mode
        return resize_and_pad(
            image, output_size, resize_method=resize, pad_value=pad_value,
        )


def select_tiling(h, w, patch_size, max_num_crops):
    """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]

    # The original size can be zero in rare cases if the image is smaller than the margin
    # In those cases letting the scale become infinite means the tiling is based on the
    # other side, or falls back to the smallest tiling
    with np.errstate(divide='ignore'):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size,
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def build_resized_image(
    image: np.ndarray,
    resize_mode: str,
    normalized_mode: str,
    base_image_input_size: List[int],
    pad_value: float,
    image_patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized, resized_mask = resize_image(
        image, resize_mode, base_image_input_size, pad_value,
    )
    resized = normalize_image(resized, normalized_mode)
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, 0)
    resized_mask = np.expand_dims(resized_mask, 0)
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
    return resized, resized_mask, resize_idx


def build_overlapping_crops(
    image: np.ndarray,
    resize_mode: str,
    normalize_mode: str,
    max_crops: int,
    overlap_margins: List[int],
    base_image_input_size: List[int],
    pad_value: float,
    image_patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose an image into a set of overlapping crops

    :return crop_arr: [n_crops, h, w, 3] The crops
    :return mask_arr: [n_crops, h, w] The padding masks
    :return patch_idx: [overlap_patch_h, overlap_patch_w] For each patch in the resized image
                        the crops were extracted from, what patch in `crop_arr` it corresponds to
    """
    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]
    assert base_image_input_size[0] == base_image_input_size[1]

    left_margin, right_margin = overlap_margins
    total_margin_pixels = image_patch_size * (right_margin + left_margin)  # pixels removed per dim
    crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
    crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
    crop_window_size = crop_window_patches * image_patch_size
    crop_patch_w = base_image_input_size[1] // image_patch_size
    crop_patch_h = base_image_input_size[0] // image_patch_size
    original_image_h, original_image_w = image.shape[:2]
    crop_size = base_image_input_size[0]

    # Decide how to tile the image, to account for the overlap margins we compute the tiling
    # as if we had an image without the margins and were using a crop size without the margins
    tiling = select_tiling(
        original_image_h - total_margin_pixels,
        original_image_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    src, img_mask = resize_image(
        image,
        resize_mode,
        [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
        pad_value,
    )
    src = normalize_image(src, normalize_mode)

    # Now we have to split the image into crops, and track what patches came from
    # where in `patch_idx_arr`
    n_crops = tiling[0] * tiling[1]
    crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
    mask_arr = np.zeros([n_crops, crop_size, crop_size], dtype=img_mask.dtype)
    patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
    on = 0
    on_crop = 0
    for i in range(tiling[0]):
        # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
        # which results in overlapping crop windows
        y0 = i*crop_window_size
        for j in range(tiling[1]):
            x0 = j*crop_window_size
            crop_arr[on_crop] = src[y0:y0+crop_size, x0:x0+crop_size]
            mask_arr[on_crop] = img_mask[y0:y0+crop_size, x0:x0+crop_size]
            patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
            patch_idx += on_crop * crop_patch_h * crop_patch_w

            # Mask out idx that are in the overlap region
            if i != 0:
                patch_idx[:left_margin, :] = -1
            if j != 0:
                patch_idx[:, :left_margin] = -1
            if i != tiling[0]-1:
                patch_idx[-right_margin:, :] = -1
            if j != tiling[1]-1:
                patch_idx[:, -right_margin:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
    # so it is ordered left-to-right order
    patch_idx_arr = np.reshape(
        patch_idx_arr,
        [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
    )
    patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
    patch_idx_arr = np.reshape(patch_idx_arr, [-1])

    # Now get the parts not in the overlap region, so it should map each patch in `src`
    # to the correct patch it should come from in `crop_arr`
    patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
        src.shape[0]//image_patch_size,
        src.shape[1]//image_patch_size,
    )
    return crop_arr, mask_arr, patch_idx_arr


def batch_pixels_to_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def arange_for_pooling(
    idx_arr: np.ndarray,
    pool_h: int,
    pool_w: int,
) -> np.ndarray:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


def image_to_patches_and_tokens(
    image: ImageInput,
    crop_mode: str,
    use_col_tokens: bool,
    resize_mode: str,
    normalize_mode: str,
    max_crops: int,
    overlap_margins: List[int],
    base_image_input_size: List[int],
    pad_value: float,
    image_patch_size: int,
    image_pooling_w: int,
    image_pooling_h: int,
    image_patch_token_id: int,
    image_col_token_id: int,
    image_start_token_id: int,
    image_end_token_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :return image_tokens, the token IDS for this image, including special tokens
    :return crops, the image crops to processes with the ViT
    :return mask, the padding mask for each crop
    :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                patches in `crops` to pool for that token, masked with -1
    """

    if isinstance(base_image_input_size, int):
        base_image_input_size = (base_image_input_size, base_image_input_size)
    
    base_image_input_d = image_patch_size
    pooling_w = image_pooling_w
    pooling_h = image_pooling_h
    patch_id = image_patch_token_id
    col_id = image_col_token_id
    start_id = image_start_token_id
    end_id = image_end_token_id
    crop_patch_w = base_image_input_size[1] // base_image_input_d
    crop_patch_h = base_image_input_size[0] // base_image_input_d

    if crop_mode == "resize":
        resized, resized_mask, resize_idx = build_resized_image(
            image,
            resize_mode,
            normalize_mode,
            base_image_input_size,
            pad_value,
            image_patch_size
        )
        pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
        h, w = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
        per_row = np.full(
            (w,),
            patch_id,
            dtype=np.int32
        )
        if use_col_tokens:
            per_row = np.concatenate([per_row, [col_id]], 0)
        extra_tokens = np.tile(per_row, [h])
        joint = [
            [start_id],
            extra_tokens,
            [end_id],
        ]
        return (
            np.concatenate(joint, 0),
            batch_pixels_to_patches(resized, image_patch_size),
            batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1),
            pooling_idx,
        )
    
    if crop_mode in ["overlap-and-resize-c2", "overlap-and-resize"]:
        crop_arr, mask_arr, patch_idx_arr = build_overlapping_crops(
            image,
            resize_mode,
            normalize_mode,
            max_crops,
            overlap_margins,
            base_image_input_size,
            pad_value,
            image_patch_size,
        )
        pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
        h, w = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])

        # Now build the output tokens
        per_row = np.full(w, patch_id, dtype=np.int32)
        if use_col_tokens:
            per_row = np.concatenate([per_row, [col_id]], 0)
        joint = np.tile(per_row, [h])
        joint = [
            [start_id],
            joint,
            [end_id]
        ]

        if crop_mode == "overlap-and-resize":
            crop_arr = batch_pixels_to_patches(crop_arr, image_patch_size)
            mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
            return np.concatenate(joint, 0), crop_arr, mask_arr, pooling_idx
        
        # Finally do the same for the global image
        resized, resized_mask, resize_idx = build_resized_image(
            image,
            resize_mode,
            normalize_mode,
            base_image_input_size,
            pad_value,
            image_patch_size
        )
        crop_arr = np.concatenate([resized, crop_arr], 0)

        mask_arr = np.concatenate([resized_mask, mask_arr], 0)

        resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
        h, w = resize_idx.shape[:2]
        resize_idx = resize_idx.reshape([-1, pooling_h*pooling_w])

        # Global image goes first, so the order of patches in previous crops gets increased
        pooling_idx = np.where(
            pooling_idx >= 0,
            pooling_idx + crop_patch_h*crop_patch_w,
            -1
        )
        pooling_idx = np.concatenate([resize_idx, pooling_idx])

        per_row = np.full(
            (w,),
            patch_id,
            dtype=np.int32
        )
        if use_col_tokens:
            per_row = np.concatenate([per_row, [col_id]], 0)
        extra_tokens = np.tile(per_row, [h])
        joint = [
            [start_id],
            extra_tokens,
            [end_id],
        ] + joint
        mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
        return (
            np.concatenate(joint, 0),
            batch_pixels_to_patches(crop_arr, image_patch_size),
            mask_arr,
            pooling_idx
        )
    else:
        raise NotImplementedError(crop_mode)


class MolmoImagesKwargs(ImagesKwargs, total=False):
    crop_mode: Optional[str]
    use_col_tokens: Optional[bool]
    resize_mode: Optional[str]
    normalize_mode: Optional[str]
    max_crops: Optional[int]
    overlap_margins: Optional[List[int]]
    base_image_input_size: Optional[List[int]]
    pad_value: Optional[float]
    image_patch_size: Optional[int]
    image_pooling_w: Optional[int]
    image_pooling_h: Optional[int]
    image_padding_mask: Optional[bool]


class MolmoImageProcessor(BaseImageProcessor):

    def __init__(
        self,
        crop_mode: str = "overlap-and-resize-c2",
        use_col_tokens: bool = True,
        resize_mode: str = "siglip",
        normalize_mode: str = "siglip",
        max_crops: int = 8,
        overlap_margins: List[int] = [4, 4],
        base_image_input_size: List[int] = (378, 378),
        pad_value: float = 0.0,
        image_patch_size: int = 14,
        image_pooling_w: int = 2,
        image_pooling_h: int = 2,
        image_padding_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.crop_mode = crop_mode
        self.use_col_tokens = use_col_tokens
        self.resize_mode = resize_mode
        self.normalize_mode = normalize_mode
        self.overlap_margins = overlap_margins
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.base_image_input_size = base_image_input_size
        self.pad_value = pad_value
        self.image_patch_size = image_patch_size
        self.image_pooling_w = image_pooling_w
        self.image_pooling_h = image_pooling_h
        self.image_padding_mask = image_padding_mask
    
    def preprocess(
        self,
        image: ImageInput,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        crop_mode: Optional[str] = None,
        use_col_tokens: Optional[bool] = None,
        resize_mode: Optional[str] = None,
        normalize_mode: Optional[str] = None,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[List[int]] = None,
        base_image_input_size: Optional[List[int]] = None,
        pad_value: Optional[float] = None,
        image_patch_size: Optional[int] = None,
        image_pooling_w: Optional[int] = None,
        image_pooling_h: Optional[int] = None,
        image_padding_mask: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess an image for the model.
        Args:
            image: The image to preprocess.
            image_patch_token_id: The token ID for the image patch.
            image_col_token_id: The token ID for the image column.
            image_start_token_id: The start token ID for the image.
            image_end_token_id: The end token ID for the image.
            crop_mode: The crop mode to use. If None, use the default crop mode.
            use_col_tokens: Whether to use column tokens. If None, use the default value.
            resize_mode: The resize mode to use. If None, use the default resize mode.
            normalize_mode: The normalization mode to use. If None, use the default normalization mode.
            max_crops: The maximum number of crops to use. If None, use the default value.
            overlap_margins: The overlap margins to use. If None, use the default values.
            base_image_input_size: The base image input size to use. If None, use the default size.
            pad_value: The padding value to use. If None, use the default value.
            image_patch_size: The size of the image patches. If None, use the default size.
            image_pooling_h: The height of the image pooling. If None, use the default height.
            image_pooling_w: The width of the image pooling. If None, use the default width.
            image_padding_mask: Whether to use image padding masks. If None, use the default value.

        Returns:
            A tuple containing:
                - The image tokens
                - The preprocessed images
                - The padding masks
                - The pooling indices
        """
        if not is_valid_image(image):
            raise ValueError("Invalid image input")
        
        crop_mode = crop_mode or self.crop_mode
        use_col_tokens = use_col_tokens or self.use_col_tokens
        normalize_mode = normalize_mode or self.normalize_mode
        resize_mode = resize_mode or self.resize_mode
        normalize_mode = normalize_mode or self.normalize_mode
        use_col_tokens = use_col_tokens or self.use_col_tokens
        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        base_image_input_size = base_image_input_size or self.base_image_input_size
        pad_value = pad_value or self.pad_value
        image_patch_size = image_patch_size or self.image_patch_size
        image_pooling_w = image_pooling_w or self.image_pooling_w
        image_pooling_h = image_pooling_h or self.image_pooling_h
        image_padding_mask = image_padding_mask or self.image_padding_mask

        image_tokens, crops, img_mask, pooled_idx = image_to_patches_and_tokens(
            image,
            crop_mode,
            use_col_tokens,
            resize_mode,
            normalize_mode,
            max_crops,
            overlap_margins,
            base_image_input_size,
            pad_value,
            image_patch_size,
            image_pooling_w,
            image_pooling_h,
            image_patch_token_id,
            image_col_token_id,
            image_start_token_id,
            image_end_token_id,
        )

        return image_tokens, crops, img_mask, pooled_idx


MolmoImageProcessor.register_for_auto_class()
