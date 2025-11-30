"""
MLflow PyFunc Wrapper for Qwen2.5-VL

Wraps trained Qwen model for serving via MLflow.
Supports both full models and LoRA adapters.
"""

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path

import mlflow
import torch
from mlflow.pyfunc import PythonModelContext
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)


# =============================================================================
# Inference Parameters
# =============================================================================
class InferenceParams(BaseModel):
    """Parameters for model inference."""

    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    do_sample: bool = True


# =============================================================================
# Image Utilities
# =============================================================================
def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def to_pil_images(inputs: list[str]) -> list[Image.Image]:
    """Convert list of base64 strings to PIL Images."""
    return [base64_to_pil(img) for img in inputs]


# =============================================================================
# MLflow PyFunc Model
# =============================================================================
class QwenVLPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for Qwen2.5-VL models.

    Handles:
    - Loading model and processor from artifacts
    - LoRA adapter detection and loading
    - CPU/GPU device selection based on memory
    - Inference with configurable parameters
    """

    def load_context(self, context: PythonModelContext):
        """Load model, processor, and prompt from MLflow artifacts."""
        model_path = Path(context.artifacts["model"])
        logger.info(f"Loading model from: {model_path}")

        # Load processor (required)
        preprocessor_config = model_path / "preprocessor_config.json"
        if not preprocessor_config.exists():
            raise RuntimeError(f"Missing preprocessor_config.json in {model_path}")

        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        logger.info("Loaded processor")

        # Load prompt info (required)
        prompt_info_path = model_path / "prompt_info.json"
        if not prompt_info_path.exists():
            raise RuntimeError(f"Missing prompt_info.json in {model_path}")

        with open(prompt_info_path) as f:
            self.prompt_info = json.load(f)

        if "prompt_text" not in self.prompt_info:
            raise RuntimeError("prompt_info.json missing 'prompt_text' field")

        self.prompt = self.prompt_info["prompt_text"]
        logger.info(
            f"Loaded prompt: {self.prompt_info['prompt_name']} v{self.prompt_info['prompt_version']}"
        )

        # Determine model type
        has_lora = (model_path / "adapter_config.json").exists()
        has_full = (model_path / "config.json").exists()

        if not has_lora and not has_full:
            raise RuntimeError(
                f"Missing model files (need config.json or adapter_config.json) in {model_path}"
            )

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # Load model
        if has_lora:
            logger.info("Loading LoRA model")

            # Load base model
            base_model_name = self._get_base_model_from_adapter_config(
                model_path / "adapter_config.json"
            )

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                str(model_path),
                torch_dtype=torch.bfloat16,
            )
            logger.info(f"Loaded LoRA adapter from {model_path}")
        else:
            logger.info("Loading full model")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def _get_base_model_from_adapter_config(self, adapter_config_path: Path) -> str:
        """Extract base model name from adapter_config.json."""
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)

        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise RuntimeError("adapter_config.json missing 'base_model_name_or_path'")

        logger.info(f"Base model: {base_model}")
        return base_model

    def _should_use_cpu(self) -> bool:
        """Determine if model should load on CPU."""
        # Check for explicit override
        if os.getenv("MLFLOW_MODEL_DEVICE", "").lower() == "cpu":
            return True

        if not torch.cuda.is_available():
            return True

        try:
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3

            # Use CPU if less than 3GB free or less than 20% available
            if free_mem < 3.0 or (free_mem / total_mem) < 0.2:
                logger.warning(f"Low GPU memory: {free_mem:.1f}GB / {total_mem:.1f}GB")
                return True

            return False

        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            return True

    def predict(
        self,
        context,
        model_input: list[str],
        params: dict | None = None,
    ) -> list[str]:
        """
        Run inference on images.

        Args:
            context: MLflow context (unused)
            model_input: List of base64-encoded images
            params: Optional inference parameters

        Returns:
            List of model responses
        """
        # Convert inputs
        images = to_pil_images(model_input)

        # Get inference params
        if params is None:
            params = InferenceParams().model_dump()

        # Default prompt for radiology (TODO: make configurable)
        prompt = "You are an expert radiographer. Describe accurately what you see in this image."

        results = []
        for image in images:
            try:
                # Prepare conversation
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Apply chat template
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Process inputs
                inputs = self.processor(
                    text=[text_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **params)

                # Decode (skip input tokens)
                generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                results.append(response)

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                raise RuntimeError(f"Prediction failed: {e}") from e

        return results
