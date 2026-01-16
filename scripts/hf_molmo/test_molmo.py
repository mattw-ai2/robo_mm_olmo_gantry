from PIL import Image
import requests

import torch

from .modeling_molmo import MolmoForCausalLM
from .configuration_molmo import MolmoConfig
from .image_preprocessing_molmo import MolmoImagesKwargs, MolmoImageProcessor
from .preprocessing_molmo import MolmoProcessorKwargs, MolmoProcessor

from transformers import AutoTokenizer, AutoProcessor, GenerationConfig

#ckpt_path = "/net/nfs/prior/sanghol/molmo/models/dense-cap-v1/captioner-qwen2.5_7b-siglip2-norm-fix/hf_ckpt"
ckpt_path = "/weka/oe-training-default/piperw/checkpoints/webolmo_checkpoints/molmo_uground_1m_hf"
#ckpt_path = "/weka/oe-training-default/webolmo/checkpoints/molmo/molmo_uground_1m/step30000-hf"

processor = AutoProcessor.from_pretrained(
    ckpt_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

model = MolmoForCausalLM.from_pretrained(
    ckpt_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image.",
)


def cast_float_dtype(t: torch.Tensor):
    if torch.is_floating_point(t):
        t = t.to(torch.bfloat16)
    return t

inputs = {k: cast_float_dtype(v.to(model.device)).unsqueeze(0) for k, v in inputs.items()}

with torch.inference_mode():
    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=448, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

generated_tokens = output[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

import pdb; pdb.set_trace()
