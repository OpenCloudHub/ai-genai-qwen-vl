"""
Qwen VL Dataset and Data Collation

Image-only dataset for Qwen2.5-VL fine-tuning.
Handles:
- Loading annotations in Qwen conversation format
- Image preprocessing with AutoProcessor
- Tokenization and label masking
- Batch collation with padding

Based on official Qwen2.5-VL training code, simplified for image-only use.
"""

import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from src.training.config import DataConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
DEFAULT_IMAGE_TOKEN = "<image>"


# =============================================================================
# Preprocessing
# =============================================================================
def preprocess_conversations(
    sources: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
    image_grid_thw: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Preprocess conversations for Qwen2.5-VL format.

    Args:
        sources: List of conversations, each conversation is a list of turns
        tokenizer: HuggingFace tokenizer
        image_grid_thw: List of grid sizes for each image (t*h*w // merge_size^2)

    Returns:
        Dictionary with input_ids and labels tensors
    """
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    # Use copy to avoid modifying original tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template

    image_index = 0
    input_ids, targets = [], []

    for source in sources:
        # Ensure conversation starts with human turn
        if roles.get(source[0]["from"]) != "user":
            source = source[1:]

        input_id, target = [], []

        # Add system message
        system_tokens = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        input_id.extend(system_tokens)
        target.extend([IGNORE_INDEX] * len(system_tokens))

        # Process each conversation turn
        for turn in source:
            role = roles.get(
                turn.get("role", turn.get("from")), turn.get("role", turn.get("from"))
            )
            content = turn.get("content", turn.get("value", ""))

            # Replace <image> tokens with vision tokens
            if "<image>" in content and image_grid_thw is not None:
                parts = content.split("<image>")
                new_parts = []
                for i, part in enumerate(parts[:-1]):
                    new_parts.append(part)
                    # Insert vision tokens for this image
                    num_tokens = image_grid_thw[image_index]
                    replacement = (
                        "<|vision_start|>"
                        + "<|image_pad|>" * num_tokens
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                    image_index += 1
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            # Tokenize turn
            turn_tokens = tokenizer.apply_chat_template(
                [{"role": role, "content": content}]
            )
            input_id.extend(turn_tokens)

            # Mask user/system turns, keep assistant turns as targets
            if role in ["user", "system"]:
                target.extend([IGNORE_INDEX] * len(turn_tokens))
            else:
                # Mask the first 3 tokens (<|im_start|>assistant\n)
                target_tokens = turn_tokens.copy()
                target_tokens[:3] = [IGNORE_INDEX] * 3
                target.extend(target_tokens)

        assert len(input_id) == len(target), (
            f"Length mismatch: {len(input_id)} != {len(target)}"
        )
        input_ids.append(input_id)
        targets.append(target)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
    }


# =============================================================================
# Dataset
# =============================================================================
class QwenVLDataset(Dataset):
    """
    Dataset for Qwen2.5-VL supervised fine-tuning.

    Expects data structure:
        dataset_path/
        ├── train/
        │   ├── annotations.json
        │   └── images/
        └── test/
            ├── annotations.json
            └── images/

    Annotation format:
        [
            {
                "image": "image_001.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nDescribe this image."},
                    {"from": "gpt", "value": "This image shows..."}
                ]
            },
            ...
        ]
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin,
        data_config: DataConfig,
        split: str = "train",
    ):
        """
        Initialize dataset.

        Args:
            tokenizer: HuggingFace tokenizer
            processor: Qwen processor (for image_processor)
            data_config: Data configuration
            split: "train" or "test"
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = processor.image_processor
        self.data_config = data_config
        self.split = split

        # Configure image processor
        self.image_processor.max_pixels = data_config.max_pixels
        self.image_processor.min_pixels = data_config.min_pixels
        # self.image_processor.size["longest_edge"] = data_config.max_pixels
        # self.image_processor.size["shortest_edge"] = data_config.min_pixels

        # Get paths based on split
        if split == "train":
            annotation_path = data_config.train_annotation_path
            self.images_dir = data_config.train_images_path
        else:
            annotation_path = data_config.eval_annotation_path
            self.images_dir = data_config.eval_images_path

        # Load annotations
        logger.info(f"Loading {split} annotations from: {annotation_path}")
        with open(annotation_path) as f:
            self.annotations = json.load(f)

        logger.info(f"Loaded {len(self.annotations)} {split} samples")

        # Apply sampling
        if data_config.sampling_percent < 1.0:
            original_count = len(self.annotations)
            sample_size = int(len(self.annotations) * data_config.sampling_percent)
            self.annotations = random.sample(self.annotations, sample_size)
            logger.info(f"Sampled {len(self.annotations)}/{original_count} samples")

        # Shuffle training data
        if split == "train":
            random.shuffle(self.annotations)

    def __len__(self) -> int:
        return len(self.annotations)

    def _process_image(self, image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single image through the image processor.

        Returns:
            Tuple of (pixel_values, grid_thw)
        """
        image = Image.open(image_path).convert("RGB")

        processed = self.image_processor.preprocess(image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        grid_thw = processed["image_grid_thw"][0]

        if isinstance(pixel_values, list):
            pixel_values = pixel_values[0]

        return pixel_values, grid_thw

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single training sample.

        Returns:
            Dictionary containing:
                - input_ids: Token IDs
                - labels: Target labels (masked for non-assistant tokens)
                - attention_mask: Sequence length for attention
                - position_ids: 3D position IDs for RoPE
                - pixel_values: Processed image tensor
                - image_grid_thw: Image grid dimensions
        """
        sample = self.annotations[idx]

        # Process image
        image_filename = sample["image"]
        image_path = self.images_dir / image_filename

        try:
            pixel_values, grid_thw = self._process_image(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

        # Calculate merged grid size for tokenization
        merge_size = self.image_processor.merge_size
        grid_thw_merged = grid_thw.prod() // (merge_size**2)

        # Preprocess conversations
        conversations = sample["conversations"]
        data_dict = preprocess_conversations(
            sources=[conversations],
            tokenizer=self.tokenizer,
            image_grid_thw=[grid_thw_merged.item()],
        )

        # Generate 3D position IDs for multimodal RoPE
        position_ids = self._get_rope_index(
            input_ids=data_dict["input_ids"],
            image_grid_thw=grid_thw.unsqueeze(0),
        )

        return {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "attention_mask": [data_dict["input_ids"].shape[1]],
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw.unsqueeze(0),
        }

    def _get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate 3D RoPE position indices for multimodal inputs.
        Based on original Qwen2.5-VL get_rope_index_25.
        """
        spatial_merge_size = self.image_processor.merge_size
        image_token_id = 151655
        vision_start_token_id = 151652

        batch_size, seq_len = input_ids.shape
        position_ids = torch.ones(
            3,
            batch_size,
            seq_len,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        for batch_idx in range(batch_size):
            tokens = input_ids[batch_idx]

            # Find vision start positions
            vision_start_indices = torch.argwhere(
                tokens == vision_start_token_id
            ).squeeze(1)

            if len(vision_start_indices) == 0:
                # Text-only: simple sequential positions
                pos = torch.arange(seq_len, device=input_ids.device)
                position_ids[:, batch_idx, :] = pos.unsqueeze(0).expand(3, -1)
                continue

            # Count images
            vision_tokens = tokens[vision_start_indices + 1]
            num_images = (vision_tokens == image_token_id).sum().item()

            input_tokens = tokens.tolist()
            llm_pos_ids_list = []
            st = 0
            image_idx = 0

            for _ in range(num_images):
                # Find next image token
                if image_token_id not in input_tokens[st:]:
                    break
                ed = input_tokens.index(image_token_id, st)

                # Get image grid dimensions
                t, h, w = image_grid_thw[image_idx]
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size

                # Text positions before this image
                text_len = ed - st
                st_idx = (
                    llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                )

                if text_len > 0:
                    text_pos = torch.arange(text_len, device=input_ids.device)
                    llm_pos_ids_list.append(text_pos.view(1, -1).expand(3, -1) + st_idx)
                    st_idx += text_len

                # Image positions (3D grid)
                t_indices = (
                    torch.arange(llm_grid_t, device=input_ids.device)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_indices = (
                    torch.arange(llm_grid_h, device=input_ids.device)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_indices = (
                    torch.arange(llm_grid_w, device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )

                llm_pos_ids_list.append(
                    torch.stack([t_indices, h_indices, w_indices]) + st_idx
                )

                # Move past image tokens
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                image_idx += 1

            # Remaining text after last image
            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max().item() + 1 if llm_pos_ids_list else 0
                )
                text_len = len(input_tokens) - st
                text_pos = torch.arange(text_len, device=input_ids.device)
                llm_pos_ids_list.append(text_pos.view(1, -1).expand(3, -1) + st_idx)

            # Concatenate all positions
            if llm_pos_ids_list:
                all_positions = torch.cat(llm_pos_ids_list, dim=1)
                position_ids[:, batch_idx, : all_positions.shape[1]] = all_positions

        return position_ids


# =============================================================================
# Data Collator
# =============================================================================
@dataclass
class QwenVLDataCollator:
    """
    Collate batch of samples for Qwen2.5-VL training.

    Handles:
    - Padding input_ids and labels
    - Stacking position_ids with padding
    - Concatenating image tensors
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: list[dict]) -> dict[str, torch.Tensor]:
        # Extract components
        input_ids = [inst["input_ids"].squeeze(0) for inst in instances]
        labels = [inst["labels"].squeeze(0) for inst in instances]
        position_ids = [inst["position_ids"] for inst in instances]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # Pad and stack position_ids
        max_len = max(p.shape[2] for p in position_ids)
        padded_positions = []
        for p in position_ids:
            pad_len = max_len - p.shape[2]
            padded = torch.nn.functional.pad(p, (0, pad_len), value=1)
            padded_positions.append(padded)
        position_ids = torch.cat(padded_positions, dim=1)

        # Truncate to model max length if needed
        max_length = getattr(self.tokenizer, "model_max_length", 2048)
        input_ids = input_ids[:, :max_length]
        labels = labels[:, :max_length]
        position_ids = position_ids[:, :, :max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
        }

        # Concatenate images
        pixel_values = [inst["pixel_values"] for inst in instances]
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)

        grid_thw = [inst["image_grid_thw"] for inst in instances]
        batch["image_grid_thw"] = torch.cat(grid_thw, dim=0)

        return batch


# =============================================================================
# Factory Function
# =============================================================================
def create_dataset_and_collator(
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin,
    data_config: DataConfig,
) -> dict:
    """
    Create dataset and collator for training.

    Args:
        tokenizer: HuggingFace tokenizer
        processor: Qwen processor
        data_config: Data configuration

    Returns:
        Dictionary with train_dataset, eval_dataset (optional), and data_collator
    """
    train_dataset = None
    eval_dataset = None

    if data_config.do_train:
        train_dataset = QwenVLDataset(
            tokenizer=tokenizer,
            processor=processor,
            data_config=data_config,
            split="train",
        )
        logger.info(f"Created training dataset with {len(train_dataset)} samples")

    if data_config.do_eval:
        eval_dataset = QwenVLDataset(
            tokenizer=tokenizer,
            processor=processor,
            data_config=data_config,
            split="test",
        )
        logger.info(f"Created evaluation dataset with {len(eval_dataset)} samples")

    data_collator = QwenVLDataCollator(tokenizer=tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
