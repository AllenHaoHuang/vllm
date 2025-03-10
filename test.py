
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

# Create a VLLM ModelConfig object using your HuggingFace config
vllm_model_config = ModelConfig(
    model_type="swissai",
    hidden_size=config.hidden_size,
    num_hidden_layers=config.num_hidden_layers,
    num_attention_heads=config.num_attention_heads,
    intermediate_size=config.intermediate_size,
    hidden_act=config.hidden_act,
    max_position_embeddings=config.max_position_embeddings,
    vocab_size=config.vocab_size,
    num_key_value_heads=config.num_key_value_heads,
    rope_theta=config.rope_theta,
    # Add other parameters as needed
)

# Now create VllmConfig with the appropriate ModelConfig object
vllm_config = VllmConfig(
    model_config=vllm_model_config,
    cache_config=None,
    quant_config=None
)

swissai_model = SwissAIForCausalLM(vllm_config=vllm_config)
