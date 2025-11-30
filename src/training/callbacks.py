"""
Training Callbacks

Ray Train callback for checkpoint and metrics reporting.
Saves model + processor + prompt info to Ray checkpoints for MLflow logging.
"""

import json
import logging
from pathlib import Path

from ray.train import Checkpoint, get_context, report
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class RayTrainReportCallback(TrainerCallback):
    """
    Ray Train callback that reports checkpoints and metrics.

    Saves to checkpoint:
    - Image processor (required)
    - Prompt info (required)
    """

    def __init__(self, processor, prompt_info: dict):
        """
        Initialize callback.

        Args:
            processor: Qwen processor (required)
            prompt_info: Prompt metadata from DVC (required)
        """
        if processor is None:
            raise ValueError("processor is required")
        if prompt_info is None:
            raise ValueError("prompt_info is required")

        super().__init__()
        self.processor = processor
        self.prompt_info = prompt_info
        self.metrics = {}

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        """Collect metrics on each log step."""
        if logs:
            self.metrics.update(logs)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Report checkpoint to Ray Train after HF Trainer saves."""
        rank = get_context().get_world_rank()
        if rank != 0:
            return

        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint path not found: {checkpoint_path}")

        # Validate checkpoint
        has_full_model = (checkpoint_path / "config.json").exists()
        has_lora = (checkpoint_path / "adapter_config.json").exists()

        if not (checkpoint_path / "trainer_state.json").exists():
            raise RuntimeError("Checkpoint missing trainer_state.json")

        if not has_full_model and not has_lora:
            raise RuntimeError(
                "Checkpoint missing both config.json and adapter_config.json"
            )

        checkpoint_type = "LoRA" if has_lora else "Full"
        logger.info(f"Found {checkpoint_type} checkpoint at: {checkpoint_path}")

        # Save processor
        logger.info("Saving processor to checkpoint")
        self.processor.save_pretrained(checkpoint_path)

        if not (checkpoint_path / "preprocessor_config.json").exists():
            raise RuntimeError("Failed to save processor")

        # Save prompt info
        prompt_info_path = checkpoint_path / "prompt_info.json"
        with open(prompt_info_path, "w") as f:
            json.dump(self.prompt_info, f, indent=2)
        logger.info(
            f"Saved prompt_info.json: {self.prompt_info['prompt_name']} v{self.prompt_info['prompt_version']}"
        )

        # Report to Ray
        checkpoint = Checkpoint.from_directory(str(checkpoint_path))
        logger.info("Reporting checkpoint to Ray Train")
        report(metrics=self.metrics, checkpoint=checkpoint)

        self.metrics = {}

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Report final metrics at end of training."""
        rank = get_context().get_world_rank()
        if rank != 0:
            return

        if self.metrics:
            logger.info("Reporting final metrics to Ray Train")
            report(metrics=self.metrics)
            self.metrics = {}


def load_metadata_from_dvc(
    repo: str,
    version: str,
    metadata_path: str = "data/radiology-mini/metadata.json",
) -> dict:
    """Load only metadata from DVC (no data download)."""
    import os

    from dvc.api import DVCFileSystem

    remote_config = {
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "endpointurl": os.environ.get(
            "AWS_ENDPOINT_URL", os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        ),
    }

    fs = DVCFileSystem(repo=repo, rev=version, remote_config=remote_config)
    metadata_content = fs.read_text(metadata_path)
    return json.loads(metadata_content)
