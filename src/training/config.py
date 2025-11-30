"""
Qwen VL Training Configuration

Two config types:
- Infrastructure: From environment variables only (MLflow, DVC, Ray)
- Training: From YAML only (model, lora, hyperparameters)
"""
# TODO: Integrate Hydra for dynamic config overrides?

import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field, model_validator
from transformers import TrainingArguments


# =============================================================================
# Infrastructure Config (env vars only - CI controls this)
# =============================================================================
class InfraConfig:
    """Infrastructure configuration from environment variables only."""

    # DVC
    dvc_repo: str = os.environ.get(
        "DVC_REPO", "https://github.com/OpenCloudHub/data-registry"
    )
    dvc_data_version_tag: str = os.environ[
        "DVC_DATA_VERSION_TAG"
    ]  # Required - fail if not set
    dvc_processed_path: str = os.environ.get(
        "DVC_PROCESSED_PATH", "data/radiology-mini/processed"
    )
    dvc_metadata_path: str = os.environ.get(
        "DVC_METADATA_PATH", "data/radiology-mini/metadata.json"
    )

    # MLflow
    mlflow_tracking_uri: str = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    mlflow_experiment_name: str = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "qwen-training"
    )
    mlflow_registered_model_name: str = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME", "dev.qwen-vl-radiology"
    )

    # Ray
    ray_storage_path: str = os.environ.get("RAY_STORAGE_PATH", "/tmp/ray_results")
    ray_num_workers: int = int(os.environ.get("RAY_NUM_WORKERS", "1"))

    # Tracking metadata
    argo_workflow_uid: str = os.environ.get("ARGO_WORKFLOW_UID", "local")
    docker_image_tag: str = os.environ.get("DOCKER_IMAGE_TAG", "dev")


def get_infra_config() -> InfraConfig:
    """Get infrastructure config. Fails if required env vars missing."""
    return InfraConfig()


# =============================================================================
# Enums
# =============================================================================
class QuantizationType(str, Enum):
    NF4 = "nf4"
    INT4 = "int4"
    INT8 = "int8"


class LoRATargetModules(str, Enum):
    ALL_LINEAR = "all-linear"
    QKV_PROJ = "qkv_proj"
    CUSTOM = "custom"


# =============================================================================
# Training Configs (YAML only - devs control this)
# =============================================================================
class DataConfig(BaseModel):
    """Data processing settings."""

    max_pixels: int = Field(default=451584)
    min_pixels: int = Field(default=12544)
    do_train: bool = True
    do_eval: bool = False
    sampling_percent: float = Field(default=1.0, ge=0.01, le=1.0)
    data_flatten: bool = Field(
        default=False, description="Pack sequences for flash attention"
    )

    # Set at runtime after DVC download
    dataset_path: str | None = None
    annotation_filename: str = "annotations.json"
    images_folder: str = "images"

    @property
    def train_annotation_path(self) -> Path:
        if self.dataset_path is None:
            raise RuntimeError("dataset_path not set - call load_data_from_dvc first")
        return Path(self.dataset_path) / "train" / self.annotation_filename

    @property
    def train_images_path(self) -> Path:
        if self.dataset_path is None:
            raise RuntimeError("dataset_path not set - call load_data_from_dvc first")
        return Path(self.dataset_path) / "train" / self.images_folder

    @property
    def eval_annotation_path(self) -> Path:
        if self.dataset_path is None:
            raise RuntimeError("dataset_path not set - call load_data_from_dvc first")
        return Path(self.dataset_path) / "test" / self.annotation_filename

    @property
    def eval_images_path(self) -> Path:
        if self.dataset_path is None:
            raise RuntimeError("dataset_path not set - call load_data_from_dvc first")
        return Path(self.dataset_path) / "test" / self.images_folder


class ModelConfig(BaseModel):
    """Model architecture settings."""

    name: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_vision: bool = Field(default=False)
    tune_mlp: bool = Field(default=True)
    tune_llm: bool = Field(default=False)

    @model_validator(mode="after")
    def at_least_one_trainable(self):
        if not any([self.tune_vision, self.tune_mlp, self.tune_llm]):
            raise ValueError("At least one component must be trainable")
        return self


class LoRAConfig(BaseModel):
    """LoRA settings."""

    enabled: bool = False
    r: int = Field(default=64, ge=1, le=512)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    target_modules: LoRATargetModules = LoRATargetModules.ALL_LINEAR
    custom_target_modules: Optional[list[str]] = None

    @model_validator(mode="after")
    def validate_custom_modules(self):
        if self.target_modules == LoRATargetModules.CUSTOM:
            if not self.custom_target_modules:
                raise ValueError(
                    "custom_target_modules required when target_modules='custom'"
                )
        return self


class QuantizationConfig(BaseModel):
    """Quantization settings."""

    enabled: bool = False
    type: QuantizationType = QuantizationType.NF4
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    double_quant: bool = True
    compute_dtype: str = "bfloat16"

    @model_validator(mode="after")
    def validate_bit_settings(self):
        if self.enabled:
            if self.load_in_4bit and self.load_in_8bit:
                raise ValueError("Cannot use both 4-bit and 8-bit quantization")
            if not self.load_in_4bit and not self.load_in_8bit:
                raise ValueError(
                    "Must enable either 4-bit or 8-bit when quantization is enabled"
                )
        return self


class OptimizationConfig(BaseModel):
    """Optimization settings."""

    flash_attention: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=True)
    bf16: bool = Field(default=True)
    fp16: bool = False

    @model_validator(mode="after")
    def validate_precision(self):
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")
        return self


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    output_dir: str = "checkpoints"
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    mm_projector_lr: Optional[float] = Field(default=2e-5)
    vision_lr: Optional[float] = Field(default=2e-6)
    weight_decay: float = 0.01
    lr_scheduler: str = "linear"
    warmup_steps: int = 2
    max_steps: int = 10
    save_steps: int = 5
    save_total_limit: int = 1
    logging_steps: int = 1
    max_length: int = Field(default=2048)

    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    def to_hf_training_args(self, run_name: Optional[str] = None) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        if run_name is None:
            run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return TrainingArguments(
            output_dir=self.output_dir,
            run_name=run_name,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            report_to="mlflow",
            bf16=self.optimization.bf16,
            fp16=self.optimization.fp16,
            gradient_checkpointing=self.optimization.gradient_checkpointing,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )


# =============================================================================
# Main Training Config (YAML only)
# =============================================================================
class TrainConfig(BaseModel):
    """Training configuration from YAML. Infrastructure comes from env vars."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load training config from YAML."""
        import yaml

        with open(path) as f:
            yaml_config = yaml.safe_load(f) or {}

        return cls.model_validate(yaml_config)

    @property
    def training_method_summary(self) -> str:
        """Human-readable summary."""
        methods = []

        if self.training.lora.enabled and self.training.quantization.enabled:
            methods.append("QLoRA")
        elif self.training.lora.enabled:
            methods.append("LoRA")
        elif self.training.quantization.enabled:
            methods.append("Quantized")
        else:
            methods.append("Full Fine-tuning")

        if self.training.optimization.flash_attention:
            methods.append("FlashAttn")

        if self.training.optimization.gradient_checkpointing:
            methods.append("GradCkpt")

        return " + ".join(methods)

    def get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype based on precision settings."""
        if self.training.optimization.bf16:
            return torch.bfloat16
        elif self.training.optimization.fp16:
            return torch.float16
        return torch.float32
