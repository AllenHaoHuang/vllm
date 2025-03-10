import torch
from transformers import AutoModelForCausalLM, AutoConfig
from vllm.model_executor.models.apertus import LlamaForCausalLM
from vllm.config import ModelConfig

# Set the default device to CPU
torch.device("cpu")

def print_param_names(hf_checkpoint_path):
    # Load Hugging Face model on CPU
    config = AutoConfig.from_pretrained(hf_checkpoint_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint_path,
        config=config,
        torch_dtype=torch.float32,  # Use float32 instead of float16 for CPU
        low_cpu_mem_usage=True
    ).to("cpu")  # Move model to CPU

    # Print Hugging Face parameter names
    hf_param_names = sorted([name for name, _ in hf_model.named_parameters()])
    print(f"\nFound {len(hf_param_names)} parameters in Hugging Face model")
    for name in hf_param_names:
        print(name)
    
    # Load custom vLLM model on CPU
    print("\nLoading Custom VLLM model")
    vllm_config = ModelConfig(
        model=hf_checkpoint_path,
        task="generate",
        tokenizer=hf_checkpoint_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",  # Use float32 instead of float16 for CPU
        seed=42
    )
    vllm_model = LlamaForCausalLM(vllm_config=vllm_config).to("cpu")  # Move model to CPU
    
    # Print VLLM parameter names
    vllm_param_names = sorted([name for name, _ in vllm_model.named_parameters()])
    print(f"\nFound {len(vllm_param_names)} parameters in Custom VLLM model")
    for name in vllm_param_names:
        print(name)

print_param_names("/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k")
