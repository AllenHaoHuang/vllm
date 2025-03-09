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
    
    # Create VLLM configs and model
    model_config = ModelConfig(hf_config=config)
    cache_config = CacheConfig()
    vllm_config = VllmConfig(model_config=model_config, cache_config=cache_config)
    vllm_model = LlamaModel(vllm_config=vllm_config)
    
    # Get VLLM parameter names
    vllm_param_names = sorted([name for name, _ in vllm_model.named_parameters()])
    print(f"Found {len(vllm_param_names)} parameters in VLLM model")
    
    # Print detailed parameter names
    print("\n=== Hugging Face Parameters ===")
    for name in hf_param_names[:20]:  # Print first 20 for brevity
        print(f"  {name}")
    if len(hf_param_names) > 20:
        print(f"  ... and {len(hf_param_names) - 20} more")
    
    print("\n=== VLLM Parameters ===")
    for name in vllm_param_names[:20]:  # Print first 20 for brevity
        print(f"  {name}")
    if len(vllm_param_names) > 20:
        print(f"  ... and {len(vllm_param_names) - 20} more")
    
    # Find parameter name differences
    hf_set = set(hf_param_names)
    vllm_set = set(vllm_param_names)
    
    only_in_hf = hf_set - vllm_set
    only_in_vllm = vllm_set - hf_set
    
    print(f"\nParameters only in Hugging Face: {len(only_in_hf)}")
    for name in sorted(only_in_hf)[:20]:  # Print first 20 for brevity
        print(f"  {name}")
    if len(only_in_hf) > 20:
        print(f"  ... and {len(only_in_hf) - 20} more")
    
    print(f"\nParameters only in VLLM: {len(only_in_vllm)}")
    for name in sorted(only_in_vllm)[:20]:  # Print first 20 for brevity
        print(f"  {name}")
    if len(only_in_vllm) > 20:
        print(f"  ... and {len(only_in_vllm) - 20} more")
    
    # If we have a VLLM model path, check its parameters as well
    if vllm_model_path and os.path.exists(vllm_model_path):
        print(f"\nLoading saved VLLM model from {vllm_model_path}")
        vllm_state_dict = torch.load(vllm_model_path, map_location="cpu")
        saved_vllm_param_names = sorted(vllm_state_dict.keys())
        
        print(f"Found {len(saved_vllm_param_names)} parameters in saved VLLM model")
        print("\n=== Saved VLLM Parameters ===")
        for name in saved_vllm_param_names[:20]:  # Print first 20 for brevity
            print(f"  {name}")
        if len(saved_vllm_param_names) > 20:
            print(f"  ... and {len(saved_vllm_param_names) - 20} more")
    
    # Create a mapping between HF and VLLM parameter names
    print("\nCreating parameter name mapping...")
    mapping = {}
    
    # Special mappings from the provided code
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }
    
    # Save the mappings to a file for reference
    with open("param_names.json", "w") as f:
        json.dump({
            "hf_params": hf_param_names,
            "vllm_params": vllm_param_names,
            "only_in_hf": list(only_in_hf),
            "only_in_vllm": list(only_in_vllm),
            "packed_modules_mapping": packed_modules_mapping,
            "mistral_mapping": mistral_mapping
        }, f, indent=2)
    
    print(f"Parameter names saved to param_names.json")

# Example usage
if __name__ == "__main__":
    hf_checkpoint = "/capstor/store/cscs/swissai/a06/main_run_megatron/hf-checkpoints/apertus3-1b-21n-600k"
    vllm_output = ""  # Optional
    
    print_param_names(hf_checkpoint, vllm_output)
