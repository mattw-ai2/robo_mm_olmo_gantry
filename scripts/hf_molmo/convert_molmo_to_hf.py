import argparse
import os
import logging

import torch

from olmo.models.molmo.molmo import Molmo, MolmoConfig as ModelConfig
from olmo.train.checkpointer import load_model_state
from olmo.util import (
    prepare_cli_environment,
    resource_path
)

from .configuration_molmo import MolmoConfig
from .modeling_molmo import MolmoForCausalLM
from .preprocessing_molmo import MolmoProcessor
from .image_preprocessing_molmo import MolmoImageProcessor


logger = logging.getLogger(__name__)

demo_chat_template = """{% for message in messages -%}
        {%- if (loop.index % 2 == 1 and message['role'] != 'user') or 
          (loop.index % 2 == 0 and message['role'].lower() != 'assistant') -%}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {%- endif -%}
        {{ message['role'].capitalize() + ': ' + message['content'] }}
        {%- if not loop.last -%}
        {{ ' ' }}
        {%- endif %}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
        {{ ' Assistant:' }}
        {%- endif %}"""


def convert_config(model_config: ModelConfig) -> MolmoConfig:
    """Convert config to HF-compatible config"""

    kwargs = {
        "torch_dtype": "bfloat16"
    }
    config = MolmoConfig(config=model_config, use_cache=True, **kwargs)
    return config


def convert_model(checkpoint_dir: str, model_config: ModelConfig, hf_config: MolmoConfig) -> MolmoForCausalLM:
    """Convert model to HF-compatible model"""
    with torch.device("meta"):
        model: Molmo = model_config.build_model()
    model.to_empty(device=torch.device("cpu"))
    load_model_state(checkpoint_dir, model)
    model.eval()
    model = model.to(torch.bfloat16)
    
    hf_model = MolmoForCausalLM(hf_config, model=model)
    return hf_model


def convert_checkpoint(checkpoint_dir: str, output_dir: str) -> None:
    logger.info(f"Loading model config from {checkpoint_dir}")
    config_path = resource_path(checkpoint_dir, "config.yaml")
    model_config: ModelConfig = ModelConfig.load(config_path, key="model", validate_paths=False)
    logger.info(f"Save model config and checkpoint to {output_dir}")
    hf_config = convert_config(model_config)

    # add max_sequence_length to model_config's outmost layer, otherwise vllm will not be able to read it and will set it to default
    hf_config.max_sequence_length = model_config.llm.max_sequence_length

    hf_model = convert_model(checkpoint_dir, model_config, hf_config)
    hf_model.save_pretrained(output_dir)

    logger.info(f"Save preprocessor config to {output_dir}")
    if "qwen" not in model_config.llm.tokenizer.identifier.lower():
        MolmoProcessor.tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast")

    tokenizer = model_config.build_tokenizer().tokenizer
    data_formatter = model_config.data_formatter
    if data_formatter.message_format == "role":
        tokenizer.chat_template = demo_chat_template
    mm_cfg = model_config.mm_preprocessor
    vit_cfg = model_config.vision_backbone.vit

    preprocessor = MolmoProcessor(
        MolmoImageProcessor(
            crop_mode=mm_cfg.crop_mode,
            use_col_tokens=mm_cfg.use_col_tokens,
            resize_mode=vit_cfg.resize_mode,
            normalize_mode=vit_cfg.normalize,
            max_crops=mm_cfg.max_crops,
            overlap_margins=mm_cfg.overlap_margins,
            base_image_input_size=vit_cfg.image_default_input_size,
            pad_value=vit_cfg.pad_value,
            image_patch_size=vit_cfg.image_patch_size,
            image_pooling_w=mm_cfg.pooling_w,
            image_pooling_h=mm_cfg.pooling_h,
            image_padding_mask=model_config.vision_backbone.image_padding_embed is not None,
        ),
        tokenizer,
        prompt_templates=data_formatter.prompt_templates,
        message_format=data_formatter.message_format,
        system_prompt=data_formatter.system_prompt,
        always_start_with_space=data_formatter.always_start_with_space,
        default_inference_len=data_formatter.default_inference_len
    )
    preprocessor.save_pretrained(output_dir)

    if "qwen" not in model_config.llm.tokenizer.identifier.lower():
        logging.warning("Hacking processor file to use the GPT2Toeknizer tokenizer class")
        preprocessing_file = os.path.join(output_dir, "preprocessing_molmo.py")
        logging.warning("Fixing tokenizer")
        with open(preprocessing_file, "r") as f:
            data = f.read()
        to_replace = '("Qwen2Tokenizer", "Qwen2TokenizerFast")'
        data = data.replace(to_replace, "(\"GPT2Tokenizer\", \"GPT2TokenizerFast\")")
        with open(preprocessing_file, "w") as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser(
        description="Adds a config.json to the checkpoint directory, creates pytorch_model.bin, and save the toeknizer,"
        "making it easier to load weights as HF models."
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Location of Molmo checkpoint.",
    )
    parser.add_argument(
        "output_dir",
        help="Location to save the converted checkpoint.",
    )
    args = parser.parse_args()
    prepare_cli_environment()
    convert_checkpoint(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
