import torch
from vllm import LLM, SamplingParams

# Define the path to your Hugging Face checkpoint
checkpoint_path = "/iopsstor/scratch/cscs/ahuang/apertus3-1b-21n-600k"

# Initialize the vLLM model
# Replace "apertus" with the name of your custom model class in apertus.py
llm = LLM(model="apertus", checkpoint=checkpoint_path)

# Define sampling parameters for generation
sampling_params = SamplingParams(
    temperature=0.7,  # Controls randomness (lower = more deterministic)
    top_p=0.9,       # Nucleus sampling (top-p) parameter
    max_tokens=100,  # Maximum number of tokens to generate
)

# Define a prompt for testing
prompt = "Hello, how are you?"

# Generate text using the model
outputs = llm.generate(prompt, sampling_params)

# Print the generated text
for output in outputs:
    print(f"Prompt: {prompt}")
    print(f"Generated text: {output.outputs[0].text}")
