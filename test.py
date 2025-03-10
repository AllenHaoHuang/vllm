
from transformers import SwissAIConfig, SwissAIModel
from vllm.model_executor.models.swissai import SwissAIForCausalLM
from vllm.config import ModelConfig, VllmConfig

# Load the model configuration
config = SwissAIConfig.from_pretrained('/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k')
# Load the model weights
model = SwissAIModel.from_pretrained('/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k', config=config)
# Print parameter names to debug
for name, param in model.named_parameters():
        print(name)

# Create a VLLM ModelConfig object without the 'model_type' parameter
vllm_model_config = ModelConfig(
        model=config,
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
