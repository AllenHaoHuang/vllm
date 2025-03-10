
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
    hidden_size=config.hidden_size,
    num_hidden_layers=config.num_hidden_layers,
    num_attention_heads=config.num_attention_heads,
    intermediate_size=config.intermediate_size,
    max_position_embeddings=config.max_position_embeddings,
    vocab_size=config.vocab_size,
    # Add other necessary parameters from config
)

# Now create VllmConfig with the appropriate ModelConfig object
vllm_config = VllmConfig(
    model_config=vllm_model_config,
    cache_config=None,
    quant_config=None
)

swissai_model = SwissAIForCausalLM(vllm_config=vllm_config)
