
from transformers import SwissAIConfig, SwissAIModel
from vllm.model_executor.models.swissai import SwissAIForCausalLM
from vllm.config import ModelConfig, VllmConfig
import torch.distributed as dist
from vllm.distributed.parallel_state import initialize_model_parallel

hf_checkpoint_path = '/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k'
# Load the model configuration
config = SwissAIConfig.from_pretrained(hf_checkpoint_path)
# Load the model weights
model = SwissAIModel.from_pretrained(hf_checkpoint_path, config=config)
# Print parameter names to debug
for name, param in model.named_parameters():
        print(name)

# Initialize the PyTorch distributed process group
dist.init_process_group(backend='nccl', init_method='env://')

# Now initialize the model parallel environment
initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

# Create a VLLM ModelConfig object without the 'model_type' parameter
vllm_model_config = ModelConfig(
        model=hf_checkpoint_path,
        task="generate",
        tokenizer=hf_checkpoint_path,  # Uses the HF checkpoint tokenizer
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float16",
        seed=42
)

# Now create VllmConfig with the appropriate ModelConfig object
vllm_config = VllmConfig(
    model_config=vllm_model_config,
    cache_config=None,
    quant_config=None
)

swissai_model = SwissAIForCausalLM(vllm_config=vllm_config)
