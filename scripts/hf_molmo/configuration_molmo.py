"""
Molmo configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from olmo.models.molmo.molmo import MolmoConfig as ModelConfig

logger = logging.get_logger(__name__)


class MolmoConfig(PretrainedConfig):
    model_type = "molmo"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, config: ModelConfig = None, use_cache: bool = False, **kwargs):
        model_config = config if config is not None else ModelConfig()
        all_kwargs = model_config.asdict()
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {"architectures": all_kwargs.get("architectures", ["MolmoForCausalLM"]) or ["MolmoForCausalLM"]}
        )
        if all_kwargs["llm"]["head_dim"] is None:
            all_kwargs["llm"]["head_dim"] = all_kwargs["llm"]["d_model"] // all_kwargs["llm"]["n_heads"]
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.llm["n_heads"]

    @property
    def num_key_value_heads(self):
        if self.llm["n_kv_heads"] is None:
            return self.llm["n_heads"]
        else:
            return self.llm["n_kv_heads"]
    
    @property
    def head_dim(self):
        if self.llm["head_dim"] is None:
            return self.llm["d_model"] // self.llm["n_heads"]
        else:
            return self.llm["head_dim"]

    @property
    def num_hidden_layers(self):
        return self.llm["n_layers"]

    @property
    def hidden_size(self):
        return self.llm["d_model"]
    
    @property
    def vocab_size(self):
        return self.llm["embedding_size"] or self.llm["vocab_size"]


AutoConfig.register("molmo", MolmoConfig)
