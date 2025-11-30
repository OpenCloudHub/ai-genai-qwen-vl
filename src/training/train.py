"""
Qwen VL Training Entry Point

Ray Train distributed training for Qwen2.5-VL models.

Usage:
    python train.py --config configs/qlora.yaml

Environment variables (CI controls):
    DVC_DATA_VERSION: Data version tag (required)
    DVC_REPO: DVC repository URL
    MLFLOW_TRACKING_URI: MLflow server
    MLFLOW_EXPERIMENT_NAME: Experiment name
    MLFLOW_REGISTERED_MODEL_NAME: Model registry name
    RAY_STORAGE_PATH: Ray checkpoint storage
    ARGO_WORKFLOW_UID: Workflow tracking
    DOCKER_IMAGE_TAG: Image tracking
"""

import argparse
import os
from importlib.metadata import version
from pathlib import Path

import mlflow
import ray
import ray.train.huggingface.transformers
import torch
from mlflow.models import infer_signature
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from ray.train import (
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
    get_checkpoint,
    get_context,
)
from ray.train.torch import TorchTrainer
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from src.training.callbacks import RayTrainReportCallback, load_metadata_from_dvc
from src.training.config import InfraConfig, TrainConfig
from src.training.data import create_dataset_and_collator
from src.training.dvc_loader import load_data_from_dvc
from src.training.log import (
    get_logger,
    get_ray_logging_config,
    log_error,
    log_info,
    log_key_value,
    log_results_summary,
    log_section,
    log_success,
    log_training_summary,
    log_warning,
)
from src.training.mlflow_wrapper import QwenVLPyFuncModel
from src.training.trainer import QwenTrainer

logger = get_logger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================
def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    import base64
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def check_mlflow_env():
    """Check MLflow S3 environment variables are set."""
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL"]
    missing = [var for var in required if var not in os.environ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")


# =============================================================================
# Model Loading
# =============================================================================
def load_model(config: TrainConfig):
    """Load Qwen2.5-VL model with optional quantization."""
    model_name = config.model.name
    training_cfg = config.training

    logger.info("Loading model", extra={"model": model_name})

    # Setup quantization if enabled
    quantization_config = None
    if training_cfg.quantization.enabled:
        logger.info(
            "Quantization enabled",
            extra={
                "type": training_cfg.quantization.type.value,
                "compute_dtype": training_cfg.quantization.compute_dtype,
            },
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_cfg.quantization.load_in_4bit,
            load_in_8bit=training_cfg.quantization.load_in_8bit,
            bnb_4bit_use_double_quant=training_cfg.quantization.double_quant,
            bnb_4bit_quant_type=training_cfg.quantization.type.value,
            bnb_4bit_compute_dtype=getattr(
                torch, training_cfg.quantization.compute_dtype
            ),
        )

    # Attention implementation
    attn_impl = (
        "flash_attention_2" if training_cfg.optimization.flash_attention else "eager"
    )

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        dtype=config.get_torch_dtype(),
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Enable gradient checkpointing
    if training_cfg.optimization.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=training_cfg.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    return model, tokenizer, processor


def apply_lora(model, config: TrainConfig):
    """Apply LoRA adapters to model if enabled."""
    lora_cfg = config.training.lora

    if not lora_cfg.enabled:
        return model

    logger.info("Applying LoRA", extra={"r": lora_cfg.r, "alpha": lora_cfg.alpha})

    # Determine target modules
    if lora_cfg.target_modules.value == "custom":
        target_modules = lora_cfg.custom_target_modules
    elif lora_cfg.target_modules.value == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = ["q_proj", "k_proj", "v_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# =============================================================================
# Training Functions
# =============================================================================
def train_worker(train_loop_config: dict):
    """Training function that runs on each Ray worker."""
    # Unpack configs
    config = TrainConfig.model_validate(train_loop_config["config_dict"])
    infra = InfraConfig()  # Always from env vars
    mlflow_run_id = train_loop_config["mlflow_run_id"]

    rank = get_context().get_world_rank()
    is_rank0 = rank == 0

    if is_rank0:
        log_section("Worker Initialized", "🔧")
        check_mlflow_env()
        mlflow.set_tracking_uri(infra.mlflow_tracking_uri)
        mlflow.set_experiment(infra.mlflow_experiment_name)
        mlflow.start_run(run_id=mlflow_run_id)
        logger.info("Connected to MLflow run", extra={"run_id": mlflow_run_id})

    # Verify GPU
    if torch.cuda.is_available():
        if is_rank0:
            log_info(f"GPU: {torch.cuda.get_device_name()}")
    else:
        raise RuntimeError("CUDA not available")

    # Download data from DVC
    if is_rank0:
        log_section("Loading Data from DVC", "📦")

    local_path, metadata = load_data_from_dvc(
        repo=infra.dvc_repo,
        version=infra.dvc_data_version_tag,
        processed_path=infra.dvc_processed_path,
        metadata_path=infra.dvc_metadata_path,
    )

    # Update config with downloaded path
    config.data.dataset_path = str(local_path)

    # Prompt info is required
    prompt_info = metadata.get("prompt")
    if prompt_info is None:
        raise RuntimeError(
            f"No prompt info in metadata for DVC version {infra.dvc_data_version_tag}"
        )

    if is_rank0:
        log_key_value("DVC Version", infra.dvc_data_version_tag)
        log_key_value(
            "Prompt", f"{prompt_info['prompt_name']} v{prompt_info['prompt_version']}"
        )

    # Load model
    if is_rank0:
        log_section("Loading Model", "🤖")
    model, tokenizer, processor = load_model(config)

    # Apply LoRA if enabled
    model = apply_lora(model, config)

    # Create datasets
    if is_rank0:
        log_section("Preparing Data", "📦")
    data_module = create_dataset_and_collator(
        tokenizer=tokenizer,
        processor=processor,
        data_config=config.data,
    )

    # Create HuggingFace TrainingArguments
    hf_training_args = config.training.to_hf_training_args()

    # Create trainer
    trainer = QwenTrainer(
        config=config,
        model=model,
        args=hf_training_args,
        processing_class=tokenizer,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )

    if is_rank0:
        trainer.print_model_status()

    # Add Ray callback with required processor and prompt_info
    callback = RayTrainReportCallback(processor=processor, prompt_info=prompt_info)
    trainer.add_callback(callback)

    # Prepare for distributed training
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Check for existing checkpoint
    if is_rank0:
        log_section("Training", "🏃")

    checkpoint = get_checkpoint()
    if checkpoint:
        logger.info("Found Ray checkpoint", extra={"checkpoint": str(checkpoint)})
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = Path(ckpt_dir)
            has_lora = (ckpt_path / "adapter_config.json").exists()
            has_full = (ckpt_path / "config.json").exists()
            has_trainer = (ckpt_path / "trainer_state.json").exists()

            if has_trainer and (has_lora or has_full):
                log_info(f"Resuming from: {ckpt_path}")
                trainer.train(resume_from_checkpoint=str(ckpt_path))
            else:
                log_warning("Invalid checkpoint, starting fresh")
                trainer.train()
    else:
        log_info("Starting training from scratch")
        trainer.train()

    if is_rank0:
        log_success("Training completed on rank 0")


def train_driver(config: TrainConfig, infra: InfraConfig) -> ray.train.Result:
    """Driver function that orchestrates distributed training."""
    log_section("Qwen2.5-VL Training Pipeline", "🚀")

    # Load metadata to get prompt info for tagging
    log_info("Loading metadata from DVC...")
    metadata = load_metadata_from_dvc(
        repo=infra.dvc_repo,
        version=infra.dvc_data_version_tag,
        metadata_path=infra.dvc_metadata_path,
    )
    prompt_info = metadata.get("prompt")
    if prompt_info is None:
        raise RuntimeError(
            f"No prompt info in metadata for {infra.dvc_data_version_tag}"
        )

    # Show training configuration
    log_training_summary(
        model=config.model.name,
        method=config.training_method_summary,
        dataset=f"DVC: {infra.dvc_data_version_tag}",
        steps=config.training.max_steps,
        batch_size=config.training.batch_size,
        lr=config.training.learning_rate,
    )

    # Setup MLflow
    log_section("MLflow Setup", "📊")
    check_mlflow_env()
    mlflow.set_tracking_uri(infra.mlflow_tracking_uri)
    mlflow.set_experiment(infra.mlflow_experiment_name)

    log_key_value("Tracking URI", infra.mlflow_tracking_uri)
    log_key_value("Experiment", infra.mlflow_experiment_name)

    # Workflow tags from infra
    workflow_tags = {
        "argo_workflow_uid": infra.argo_workflow_uid,
        "docker_image_tag": infra.docker_image_tag,
        "dvc_data_version_tag": infra.dvc_data_version_tag,
        "prompt_name": prompt_info["prompt_name"],
        "prompt_version": str(prompt_info["prompt_version"]),
    }

    with mlflow.start_run(tags=workflow_tags) as active_run:
        mlflow_run_id = active_run.info.run_id
        log_success(f"MLflow run started: {mlflow_run_id}")

        # Log configuration
        mlflow.log_params(
            {
                "model_name": config.model.name,
                "training_method": config.training_method_summary,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "max_steps": config.training.max_steps,
                "lora_enabled": config.training.lora.enabled,
                "lora_r": config.training.lora.r
                if config.training.lora.enabled
                else None,
                "quantization_enabled": config.training.quantization.enabled,
                "flash_attention": config.training.optimization.flash_attention,
                "dvc_data_version_tag": infra.dvc_data_version_tag,
            }
        )

        # Prepare train loop config
        train_loop_config = {
            "config_dict": config.model_dump(),
            "mlflow_run_id": mlflow_run_id,
        }

        # Create Ray TorchTrainer
        log_section("Ray Trainer Setup", "⚡")
        log_key_value("Workers", infra.ray_num_workers)
        log_key_value("Storage", infra.ray_storage_path)

        ray_trainer = TorchTrainer(
            train_loop_per_worker=train_worker,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=infra.ray_num_workers,
                use_gpu=True,
            ),
            run_config=RunConfig(
                name=f"run_{mlflow_run_id}",
                storage_path=infra.ray_storage_path,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="loss",
                    checkpoint_score_order="min",
                ),
                failure_config=FailureConfig(max_failures=3),
            ),
        )

        # Run training
        log_section("Distributed Training", "🏋️")
        result = ray_trainer.fit()

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Log model to MLflow from best checkpoint
        log_section("Model Registration", "📦")

        if result.checkpoint:
            log_info(f"Best checkpoint: {result.checkpoint}")

            with result.checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)

                # Validate checkpoint
                has_full_model = (checkpoint_path / "config.json").exists()
                has_lora = (checkpoint_path / "adapter_config.json").exists()
                has_processor = (checkpoint_path / "preprocessor_config.json").exists()
                has_prompt = (checkpoint_path / "prompt_info.json").exists()

                if not has_processor:
                    raise RuntimeError("Missing preprocessor_config.json in checkpoint")
                if not has_prompt:
                    raise RuntimeError("Missing prompt_info.json in checkpoint")
                if not has_full_model and not has_lora:
                    raise RuntimeError("Missing model files in checkpoint")

                checkpoint_type = "LoRA" if has_lora else "Full"
                log_info(f"Found {checkpoint_type} checkpoint, logging to MLflow...")

                # Force CPU loading for MLflow validation
                os.environ["MLFLOW_MODEL_DEVICE"] = "cpu"

                try:
                    # Create example input
                    example_image = Image.new("RGB", (224, 224), color="white")
                    example_base64 = pil_to_base64(example_image)

                    # Infer signature
                    signature = infer_signature(
                        [example_base64],
                        ["example output text"],
                    )

                    # Log model
                    model_info = mlflow.pyfunc.log_model(
                        name="model",
                        python_model=QwenVLPyFuncModel(),
                        artifacts={"model": str(checkpoint_path)},
                        signature=signature,
                        pip_requirements=[
                            f"transformers=={version('transformers')}",
                            f"torch=={version('torch')}",
                            f"peft=={version('peft')}",
                            f"pydantic=={version('pydantic')}",
                            "pillow",
                            "accelerate",
                        ],
                        registered_model_name=infra.mlflow_registered_model_name,
                    )

                    log_success(
                        f"Model registered: {infra.mlflow_registered_model_name}"
                    )
                    log_key_value("Model URI", model_info.model_uri)

                finally:
                    os.environ.pop("MLFLOW_MODEL_DEVICE", None)
        else:
            raise RuntimeError(
                "No checkpoint available - training failed to produce checkpoint"
            )

    return result


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    log_section("Configuration", "⚙️")

    # Infrastructure from env vars (CI controls)
    log_info("Loading infrastructure config from environment")
    infra = InfraConfig()
    log_key_value("DVC Version", infra.dvc_data_version_tag)
    log_key_value("MLflow URI", infra.mlflow_tracking_uri)
    log_key_value("Experiment", infra.mlflow_experiment_name)

    # Training params from YAML (devs control)
    log_info(f"Loading training config from: {args.config}")
    config = TrainConfig.from_yaml(args.config)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(logging_config=get_ray_logging_config())
        log_success("Ray initialized with JSON logging")

    # Run training
    result = train_driver(config, infra)

    # Print results
    log_section("Results", "🏁")

    if result.error:
        log_error(f"Training failed: {result.error}")
        raise RuntimeError(f"Training failed: {result.error}")

    log_results_summary(
        metrics=result.metrics or {},
        checkpoint_path=str(result.checkpoint) if result.checkpoint else None,
    )
    log_success("Training pipeline complete!")


if __name__ == "__main__":
    main()
