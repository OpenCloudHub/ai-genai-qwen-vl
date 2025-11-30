"""
DVC Data Loader for Qwen Training

Downloads data from DVC remote by version tag.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from dvc.api import DVCFileSystem

logger = logging.getLogger(__name__)


def load_data_from_dvc(
    repo: str,
    version: str,
    processed_path: str = "data/radiology-mini/processed",
    metadata_path: str = "data/radiology-mini/metadata.json",
    download_dir: str = "/tmp/dvc_data",
) -> tuple[Path, dict]:
    """Download data from DVC and return local path + metadata.

    Args:
        repo: DVC repository URL
        version: DVC version tag (e.g., 'v1.0.0')
        processed_path: Path to processed data in repo
        metadata_path: Path to metadata.json in repo
        download_dir: Where to download data

    Returns:
        Tuple of (local_data_path, metadata_dict)
    """
    logger.info(f"Loading data from DVC: {repo}@{version}")

    # Configure remote credentials
    remote_config = {
        "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "endpointurl": os.environ.get(
            "AWS_ENDPOINT_URL", os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        ),
    }

    # Open DVC filesystem
    fs = DVCFileSystem(repo=repo, rev=version, remote_config=remote_config)

    # Load metadata
    metadata_content = fs.read_text(metadata_path)
    metadata = json.loads(metadata_content)

    logger.info(f"Dataset: {metadata['dataset']['name']}")
    if metadata.get("prompt"):
        logger.info(
            f"Prompt: {metadata['prompt']['prompt_name']} v{metadata['prompt']['prompt_version']}"
        )

    # Setup local path - fresh download
    local_path = Path(download_dir) / version
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True)

    # Download train and test directories
    logger.info("Downloading train data...")
    fs.get(f"{processed_path}/train", str(local_path / "train"), recursive=True)

    logger.info("Downloading test data...")
    fs.get(f"{processed_path}/test", str(local_path / "test"), recursive=True)

    logger.info(f"✅ Downloaded to {local_path}")

    return local_path, metadata
