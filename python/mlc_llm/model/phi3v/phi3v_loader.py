"""
This file specifies how MLC's Phi parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .phi3v_model import Phi3VConfig, Phi3VForCausalLM


def huggingface(model_config: Phi3VConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of Phi-1/Phi-1.5 HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : PhiConfig
        The configuration of the Phi model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Phi3VForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=W0632:unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    def _add(mlc_name, hf_name):
        mapping.add_mapping(
            mlc_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[mlc_name].dtype,
            ),
        )

    _add("lm_head.weight", "lm_head.weight")
    _add("transformer.norm.weight", "model.norm.weight")
    _add("transformer.embd.weight", "model.embed_tokens.weight")
    _add("vision_embed_tokens.wte.weight","model.embed_tokens.weight")

    prefix = "transformer.h"
    hf_prefix = "model.layers"
    for i in range(model_config.num_hidden_layers):
        _add(f"{prefix}.{i}.ln.weight", f"{hf_prefix}.{i}.input_layernorm.weight")
        _add(f"{prefix}.{i}.mlp.down_proj.weight", f"{hf_prefix}.{i}.mlp.down_proj.weight")
        _add(f"{prefix}.{i}.mlp.gate_up_proj.weight", f"{hf_prefix}.{i}.mlp.gate_up_proj.weight")
        _add(
            f"{prefix}.{i}.post_attention_layernorm.weight",
            f"{hf_prefix}.{i}.post_attention_layernorm.weight",
        )
        _add(f"{prefix}.{i}.mixer.out_proj.weight", f"{hf_prefix}.{i}.self_attn.o_proj.weight")
        _add(f"{prefix}.{i}.mixer.qkv_proj.weight", f"{hf_prefix}.{i}.self_attn.qkv_proj.weight")

    _add("vision_embed_tokens.glb_GN", "model.vision_embed_tokens.glb_GN")
    _add("vision_embed_tokens.sub_GN", "model.vision_embed_tokens.sub_GN")


    _add(f"vision_embed_tokens.img_processor.vision_model.embeddings.class_embedding", \
            f"model.vision_embed_tokens.img_processor.vision_model.embeddings.class_embedding")
    _add(f"vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight", \
            f"model.vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight")
    _add(f"vision_embed_tokens.img_processor.vision_model.embeddings.position_embedding.weight", \
            f"model.vision_embed_tokens.img_processor.vision_model.embeddings.position_embedding.weight")

    _add("vision_embed_tokens.img_processor.vision_model.post_layernorm.bias","model.vision_embed_tokens.img_processor.vision_model.post_layernorm.bias")
    _add("vision_embed_tokens.img_processor.vision_model.post_layernorm.weight","model.vision_embed_tokens.img_processor.vision_model.post_layernorm.weight")
    _add("vision_embed_tokens.img_processor.vision_model.pre_layrnorm.bias","model.vision_embed_tokens.img_processor.vision_model.pre_layrnorm.bias")
    _add("vision_embed_tokens.img_processor.vision_model.pre_layrnorm.weight","model.vision_embed_tokens.img_processor.vision_model.pre_layrnorm.weight")

    prefix = "vision_embed_tokens.img_processor.vision_model.encoder.layers"
    hf_prefix = "model.vision_embed_tokens.img_processor.vision_model.encoder.layers"
    for i in range(model_config.vision_config.num_hidden_layers):
        _add(f"{prefix}.{i}.layer_norm1.bias", f"{hf_prefix}.{i}.layer_norm1.bias")
        _add(f"{prefix}.{i}.layer_norm1.weight", f"{hf_prefix}.{i}.layer_norm1.weight")
        _add(f"{prefix}.{i}.layer_norm2.bias", f"{hf_prefix}.{i}.layer_norm2.bias")
        _add(f"{prefix}.{i}.layer_norm2.weight", f"{hf_prefix}.{i}.layer_norm2.weight")

        _add(f"{prefix}.{i}.mlp.fc1.bias", f"{hf_prefix}.{i}.mlp.fc1.bias")
        _add(f"{prefix}.{i}.mlp.fc1.weight", f"{hf_prefix}.{i}.mlp.fc1.weight")
        _add(f"{prefix}.{i}.mlp.fc2.bias", f"{hf_prefix}.{i}.mlp.fc2.bias")
        _add(f"{prefix}.{i}.mlp.fc2.weight", f"{hf_prefix}.{i}.mlp.fc2.weight")

        _add(f"{prefix}.{i}.self_attn.k_proj.bias", f"{hf_prefix}.{i}.self_attn.k_proj.bias")
        _add(f"{prefix}.{i}.self_attn.k_proj.weight", f"{hf_prefix}.{i}.self_attn.k_proj.weight")
        _add(f"{prefix}.{i}.self_attn.out_proj.bias", f"{hf_prefix}.{i}.self_attn.out_proj.bias")
        _add(f"{prefix}.{i}.self_attn.out_proj.weight", f"{hf_prefix}.{i}.self_attn.out_proj.weight")
        _add(f"{prefix}.{i}.self_attn.q_proj.bias", f"{hf_prefix}.{i}.self_attn.q_proj.bias")
        _add(f"{prefix}.{i}.self_attn.q_proj.weight", f"{hf_prefix}.{i}.self_attn.q_proj.weight")
        _add(f"{prefix}.{i}.self_attn.v_proj.bias", f"{hf_prefix}.{i}.self_attn.v_proj.bias")
        _add(f"{prefix}.{i}.self_attn.v_proj.weight", f"{hf_prefix}.{i}.self_attn.v_proj.weight")

    _add("vision_embed_tokens.img_projection.linear_1.bias","model.vision_embed_tokens.img_projection.0.bias")
    _add("vision_embed_tokens.img_projection.linear_1.weight","model.vision_embed_tokens.img_projection.0.weight")
    _add("vision_embed_tokens.img_projection.linear_2.bias","model.vision_embed_tokens.img_projection.2.bias")
    _add("vision_embed_tokens.img_projection.linear_2.weight","model.vision_embed_tokens.img_projection.2.weight")

    return mapping
