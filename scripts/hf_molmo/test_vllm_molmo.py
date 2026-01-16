import argparse
from typing import Optional
import numpy as np
import requests
import warnings
import logging
from io import BytesIO
import base64
import PIL
from PIL import Image, ImageFile, ImageOps
import os
import json
import math
from tqdm import tqdm

import torch
from vllm import LLM, ModelRegistry
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.sampling_params import SamplingParams
from scripts.hf_molmo.vllm_molmo import MolmoForCausalLM
ModelRegistry.register_model("MolmoForCausalLM", MolmoForCausalLM)
_MULTIMODAL_MODELS["MolmoForCausalLM"] = ("molmo", "MolmoForCausalLM")

from olmo.util import prepare_cli_environment
from olmo.data.image_preprocessor import load_image
from olmo.html_utils import postprocess_prompt
from olmo.data.dataset import DATA_HOME

BATCH_SIZE = 4


def vllm_inference(model_dir: str, data_path: str, output_path: str):

    sampling_params = SamplingParams(
        max_tokens=448,
        temperature=0
    )


    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        dtype="bfloat16",
    )

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    def _iter():
        for i in range(0, len(data), BATCH_SIZE):
            subset = data[i:i+BATCH_SIZE]
            images = [load_image(os.path.join(DATA_HOME, "pixmo_images", ex["image"])) for ex in subset]
            out = [{"prompt": "Describe this image", "multi_modal_data": {"image": img, "image_url": ex["url"]}} for img, ex in zip(images, subset)]
            yield out
    
    pbar = tqdm(total=math.ceil(len(data) / BATCH_SIZE), desc="Captioning images")

    logging.info(f"Captioning dense caption dataset with {len(data)} images, model: {model_dir}")

    predictions = []
    for inputs in _iter():
        urls = []
        for ex in inputs:
            urls.append(ex["multi_modal_data"].pop("image_url"))
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        for url, output in zip(urls, outputs):
            generated_text = output.outputs[0].text
            prompt = postprocess_prompt(output.prompt)
            predictions.append(
                dict(
                    prediction=generated_text,
                    prompt=prompt,
                    image_url=url,
                )
            )
        pbar.update(1)
    pbar.close()

    with open(output_path, "w") as f:
        json.dump(predictions, f)


def main():
    parser = argparse.ArgumentParser(
        description="vllm implementation test"
    )
    parser.add_argument(
        "model_dir",
        help="Location of Molmo HF-compatible checkpoint.",
    )
    parser.add_argument(
        "data_path",
        default="/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/dense-caption-eval/test.jsonl",
        help="Path to dense caption dataset.",
    )
    parser.add_argument(
        "output_path",
        help="Path to save predictions.",
    )
    args = parser.parse_args()
    prepare_cli_environment()
    vllm_inference(args.model_dir, args.data_path, args.output_path)


if __name__ == "__main__":
    main()

