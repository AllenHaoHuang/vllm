import torch
from transformers import AutoModelForCausalLM, AutoConfig
from vllm.model_executor.models.apertus import LlamaForCausalLM
from vllm.config import ModelConfig

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
    
    # Create a properly configured ModelConfig with required parameters
    vllm_config = ModelConfig(
        model=hf_checkpoint_path,
        task="generate",  # This was missing and is required
        tokenizer=hf_checkpoint_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=42
    )
    
    # Initialize the model with just the vllm_config
    vllm_model = LlamaForCausalLM(vllm_config=vllm_config)
    
    # Get VLLM parameter names
    vllm_param_names = sorted([name for name, _ in vllm_model.named_parameters()])
    print(f"\nFound {len(vllm_param_names)} parameters in Custom VLLM model")
    for name in vllm_param_names:
        print(name)

print_param_names("/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k")
