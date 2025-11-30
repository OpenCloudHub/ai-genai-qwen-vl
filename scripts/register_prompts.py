"""Register example radiology prompts to MLflow."""

import os

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# V1: Basic
PROMPT_V1 = "Describe this medical image."

# V2: Role-based
PROMPT_V2 = (
    "You are an expert radiographer. Describe accurately what you see in this image."
)

# V3: Detailed instructions
PROMPT_V3 = """You are an expert radiographer analyzing medical images.

Provide a detailed description including:
1. Type of imaging (X-ray, CT, MRI, etc.)
2. Anatomical region shown
3. Key findings and observations
4. Any notable abnormalities or normal variants

Be accurate and clinically relevant."""

print("📝 Registering radiology prompts...")

mlflow.genai.register_prompt(
    "radiology-vlm-prompt",
    PROMPT_V1,
    commit_message="V1: Basic - minimal instruction",
)
print("  ✓ V1 registered")

mlflow.genai.register_prompt(
    "radiology-vlm-prompt",
    PROMPT_V2,
    commit_message="V2: Role-based - expert radiographer framing",
)
print("  ✓ V2 registered")

mlflow.genai.register_prompt(
    "radiology-vlm-prompt",
    PROMPT_V3,
    commit_message="V3: Detailed - structured output guidance",
)
print("  ✓ V3 registered")

print("✅ All prompts registered!")
