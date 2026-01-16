import logging
from dataclasses import fields
from typing import List, Optional, Tuple, Union, Dict, Any, Sequence

import torch
import transformers
from packaging import version
from transformers import PreTrainedModel, GenerationConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.auto import AutoModelForCausalLM
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput

from olmo.tokenizer import TokenizerConfig
from olmo.nn.llm import LlmConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.nn.image_vit import VitConfig
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.model_preprocessor import MolmoPreprocessorConfig
from olmo.models.molmo.molmo import Molmo, MolmoConfig as ModelConfig

from .configuration_molmo import MolmoConfig

log = logging.getLogger(__name__)


def create_model_config_from_pretrained_config(config: MolmoConfig) -> ModelConfig:

    """
    Utility function
    """

    key2config = {
        "llm": LlmConfig,
        "vision_backbone": MolmoVisionBackboneConfig,
        "data_formatter": DataFormatter,
        "mm_preprocessor": MolmoPreprocessorConfig,
        "vit": VitConfig,
        "tokenizer": TokenizerConfig,
    }

    config_dict = config.to_dict()
    del config_dict["model_name"]

    def _parse_dict(d):
        if not isinstance(d, dict):
            return d
        result = {}
        for key, val in d.items():
            if key in key2config:
                val = _parse_dict(val)
                result[key] = key2config[key](**val)
            else:
                result[key] = val
        return result
    
    config_dict = _parse_dict(config_dict)

    kwargs = {}
    for field in fields(ModelConfig):
        if field.name in config_dict:
            kwargs[field.name] = config_dict[field.name]

    model_config = ModelConfig(**kwargs)
    return model_config


class MolmoForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Extremely barebones HF model wrapper.
    """
    config_class = MolmoConfig
    base_model_prefix = "model"
    _no_split_modules = ["OLMoBlock"]

    def __init__(self, config: MolmoConfig, model: Optional[Molmo] = None):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            self.model: Molmo = model_config.build_model(torch.device("cpu"))
        else:
            self.model = model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        last_logits_only: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in Molmo")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            images=images,
            image_masks=image_masks,
            pooled_patches_idx=pooled_patches_idx,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_hidden_states=output_hidden_states,
            append_last_valid_logits=append_last_valid_logits,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            assert loss_masks is not None
            loss_masks = loss_masks * (loss_masks > 0)
            batch_size_in_tokens = max(loss_masks.sum().item(), 1)
            labels = labels.long()
            labels.masked_fill_(~(loss_masks > 0), -100)
            labels = labels.view(-1)
            logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1)) # for numerical stability
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(logits_for_loss, labels)
            loss = (loss*loss_masks.view(loss.shape)).sum()
            loss = loss / batch_size_in_tokens
            use_zloss = getattr(self.config, "softmax_auxiliary_loss", False)
            if use_zloss:
                z_squared = logits_for_loss.logsumexp(-1).pow(2)
                z_loss = self.config.softmax_auxiliary_loss_scale * z_squared
                z_loss = (z_loss*loss_masks.view(z_loss.shape)).sum()
                z_loss = z_loss / batch_size_in_tokens
                loss += z_loss
        
        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte
    
    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.transformer.wte = value
    
    def get_output_embeddings(self) -> torch.nn.Module:
        if self.config.llm["weight_tying"]:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out
    
    def set_output_embeddings(self, value: torch.nn.Module) -> None:
        if self.config.llm["weight_tying"]:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def set_decoder(self, decoder: torch.nn.Module) -> None:
        self.model.transformer = decoder

    def get_decoder(self) -> torch.nn.Module:
        return self.model.transformer
    
    @torch.no_grad()
    def generate_from_batch(
        self,
        batch: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if generation_config is not None:
            assert generation_config.use_cache
        
        images = batch.get("images")
        image_masks = batch.get("image_masks")
        pooled_patches_idx = batch.get("pooled_patches_idx")

        # Validate inputs.
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        attention_mask = batch.get("attention_mask", None)
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        mask_len = seq_len + max_new_tokens if self.config.llm["use_position_ids"] else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.config.llm["use_position_ids"] and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        
        out = super().generate(
            input_ids,
            generation_config,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            pooled_patches_idx=pooled_patches_idx,
            position_ids=position_ids,
            append_last_valid_logits=append_last_valid_logits,
            **kwargs,
        )

        return out

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        
        if self.config.llm["use_position_ids"]:
            attention_mask = kwargs.get("attention_mask")
            images = kwargs.get("images")
            image_masks = kwargs.get("image_masks")
            pooled_patches_idx = kwargs.get("pooled_patches_idx")
            position_ids = kwargs.get("position_ids")
            append_last_valid_logits = kwargs.get("append_last_valid_logits")
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "last_logits_only": True,
            }
            if past_key_values is None:
                model_inputs["images"] = images
                model_inputs["image_masks"] = image_masks
                model_inputs["pooled_patches_idx"] = pooled_patches_idx
                model_inputs["append_last_valid_logits"] = append_last_valid_logits
        else:
            model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
            model_inputs.update(kwargs)
            model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        attention_mask: torch.Tensor = None
        if self.config.llm["use_position_ids"]:
            model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
            if "append_last_valid_logits" in model_kwargs:
                del model_kwargs["append_last_valid_logits"]
            if "images" in model_kwargs:
                del model_kwargs["images"]
                del model_kwargs["image_masks"]
                del model_kwargs["pooled_patches_idx"]
            attention_mask = model_kwargs.pop("attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, num_new_tokens)
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def tie_weights(self):
        """
        This function is intentionally left as a no-op.

        Weight tying is handled as follows:
        - When the model is initialized, the `ff_out` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.

        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, mean_resizing: bool = True,
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.embedding_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        Note:
            This method differs from the base class implementation by resizing the `embedding_size` attribute of the
            model configuration instead of the `vocab_size`. It also includes a warning if the resized `embedding_size`
            is less than the `vocab_size`. In OLMo, `embedding_size` refers to the dimensionality of the model's token
            embeddings, while `vocab_size` refers to the number of unique tokens in the vocabulary.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.llm["embedding_size"] = model_embeds.weight.shape[0]
        self.model.config.llm.embedding_size = model_embeds.weight.shape[0]

        # Check if the embedding size is less than the vocab size
        if self.config.llm["embedding_size"] < self.config.llm["vocab_size"]:
            warning_message = (
                f"Resizing token embeddings to size {self.config.llm['embedding_size']}, which is less than the vocab size "
                f"{self.config.llm['vocab_size']} defined in the model configuration. Make sure your tokenizer's vocabulary "
                "size is less than or equal to the new token embedding size."
            )
            log.warning(warning_message)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

# Always register for multi-modal features
AutoModelForCausalLM.register(MolmoConfig, MolmoForCausalLM)
