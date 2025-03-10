import torch
from transformers import AutoModelForCausalLM, AutoConfig
from vllm.model_executor.models.apertus import LlamaModel
from vllm.config import ModelConfig, VllmConfig, CacheConfig
import os
import json

def print_param_names(hf_checkpoint_path, vllm_model_path=None):
    """
    Print all parameter names from a Hugging Face checkpoint and VLLM model.

    Args:
        hf_checkpoint_path: Path to the Hugging Face checkpoint
        vllm_model_path: Path to the VLLM model (optional)
    """
    print(f"Loading Hugging Face model from {hf_checkpoint_path}")

    # Load the Hugging Face config and model
    config = AutoConfig.from_pretrained(hf_checkpoint_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Get HF parameter names
    hf_param_names = sorted([name for name, _ in hf_model.named_parameters()])
    print(f"\nFound {len(hf_param_names)} parameters in Hugging Face model")
    for name in hf_param_names:
        print(name)
    
    if vllm_model_path:
        print(f"\nLoading VLLM model from {vllm_model_path}")
        vllm_config = ModelConfig(model=vllm_model_path)
        vllm_model = LlamaModel(vllm_config)

        # Get VLLM parameter names
        vllm_param_names = sorted([name for name, _ in vllm_model.named_parameters()])
        print(f"\nFound {len(vllm_param_names)} parameters in VLLM model")
        for name in vllm_param_names:
            print(name)
