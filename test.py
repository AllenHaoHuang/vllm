import torch
from transformers import AutoModelForCausalLM, AutoConfig
from vllm.model_executor.models.apertus import LlamaForCausalLM
from dataclasses import dataclass

# Define a custom ModelConfig class
@dataclass
class ModelConfig:
    hf_config: dict  # Hugging Face config as a dictionary

# Define a custom VllmConfig class
@dataclass
class VllmConfig:
    model_config: ModelConfig  # Contains the Hugging Face config
    cache_config: dict = None  # Add cache_config (required by LlamaModel)
    quant_config: dict = None  # Quantization config (optional)
    lora_config: dict = None  # LoRA config (optional)

def print_param_names(hf_checkpoint_path):
    """
    Print all parameter names from a Hugging Face checkpoint and a custom VLLM model.
    Args:
        hf_checkpoint_path: Path to the Hugging Face checkpoint
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
    
    # Load your custom vLLM model
    print("\nLoading Custom VLLM model")
    
    # Create a custom VllmConfig
    hf_config = config.to_dict()  # Convert Hugging Face config to dict
    vllm_config = VllmConfig(
        model_config=ModelConfig(hf_config=hf_config),  # Wrap in ModelConfig
        cache_config={},  # Add an empty cache_config (required)
        quant_config=None,  # Add quantization config if applicable
        lora_config=None,  # Add LoRA config if applicable
    )
    
    # Initialize the model with the custom vllm_config
    vllm_model = LlamaForCausalLM(vllm_config=vllm_config)
    
    # Get VLLM parameter names
    vllm_param_names = sorted([name for name, _ in vllm_model.named_parameters()])
    print(f"\nFound {len(vllm_param_names)} parameters in Custom VLLM model")
    for name in vllm_param_names:
        print(name)

print_param_names("/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k")
