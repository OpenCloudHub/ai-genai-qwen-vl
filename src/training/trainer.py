"""
Qwen VL Trainer

Custom HuggingFace Trainer with:
- Multi-component learning rates (vision, projector, LLM)
- Flash attention patches for data flattening
- Model component training configuration

Based on official Qwen2.5-VL training code.
"""

import logging
from typing import Optional

import torch
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names

from src.training.config import TrainConfig

logger = logging.getLogger(__name__)


class QwenTrainer(Trainer):
    """
    Custom trainer for Qwen2.5-VL with multi-modal optimizer support.

    Features:
    - Separate learning rates for vision tower, projector, and LLM
    - Configurable component freezing
    - Flash attention patches for packed sequences
    """

    def __init__(
        self,
        config: TrainConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.full_config = config
        self.model_config = config.model
        self.training_config = config.training

        # Configure which components are trainable
        self._setup_trainable_components()

        # Apply flash attention patches if using data flattening
        if config.data.data_flatten:
            self._apply_flash_attention_patches()

    # =========================================================================
    # Component Training Setup
    # =========================================================================
    def _setup_trainable_components(self):
        """Configure which model components are trainable based on config."""
        logger.info("Configuring trainable parameters...")

        # Skip if LoRA is enabled - PEFT handles trainability
        if self.training_config.lora.enabled:
            logger.info("LoRA enabled - PEFT manages trainable parameters")
            self._log_parameter_summary()
            return

        model = self.model
        cfg = self.model_config

        # Vision encoder
        self._set_component_trainable(
            patterns=["visual", "vision_model", "vision_tower"],
            trainable=cfg.tune_vision,
            component_name="Vision encoder",
        )

        # Vision-language connector/merger
        self._set_component_trainable(
            patterns=[
                "visual.merger",
                "vision_model.merger",
                "multi_modal_projector",
                "mm_projector",
            ],
            trainable=cfg.tune_mlp,
            component_name="Vision-language connector",
            nested=True,
        )

        # Language model
        self._set_component_trainable(
            patterns=["model", "language_model", "llm", "text_model"],
            trainable=cfg.tune_llm,
            component_name="Language model",
            check_layers=True,
        )

        # Handle lm_head separately
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = cfg.tune_llm
            logger.info(f"  lm_head: trainable={cfg.tune_llm}")

        self._log_parameter_summary()

    def _set_component_trainable(
        self,
        patterns: list[str],
        trainable: bool,
        component_name: str,
        nested: bool = False,
        check_layers: bool = False,
    ):
        """Set trainability for model components matching patterns."""
        model = self.model
        found = False

        for pattern in patterns:
            try:
                if nested:
                    # Handle nested patterns like "visual.merger"
                    parts = pattern.split(".")
                    module = model
                    for part in parts:
                        module = getattr(module, part)
                else:
                    if not hasattr(model, pattern):
                        continue
                    module = getattr(model, pattern)

                # For LLM, verify it has layers
                if check_layers and not (
                    hasattr(module, "layers") or hasattr(module, "transformer")
                ):
                    continue

                for param in module.parameters():
                    # Skip quantized params (uint8) - only float tensors can have gradients
                    if param.is_floating_point():
                        param.requires_grad = trainable

                found = True
                logger.info(f"  {component_name} ({pattern}): trainable={trainable}")
                break

            except AttributeError:
                continue

        if not found:
            # Fallback: search by parameter name
            logger.warning(
                f"  {component_name}: not found by pattern, searching by name..."
            )
            keywords = patterns[0].lower().split(".")
            for name, param in model.named_parameters():
                if any(kw in name.lower() for kw in keywords):
                    # Skip quantized params (uint8) - only float tensors can have gradients
                    if param.is_floating_point():
                        param.requires_grad = trainable

    def _log_parameter_summary(self):
        """Log summary of trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("Parameter summary:")
        logger.info(f"  Total: {total:,}")
        logger.info(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

        if trainable == 0:
            raise ValueError("No trainable parameters! Check model configuration.")

    # =========================================================================
    # Custom Optimizer with Multi-Component LRs
    # =========================================================================
    def create_optimizer(self):
        """
        Create optimizer with separate learning rates for different components.

        Supports:
        - Base learning rate for LLM
        - mm_projector_lr for vision-language connector
        - vision_lr for vision encoder
        """
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        cfg = self.training_config

        # Get parameters that should have weight decay
        decay_params = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        # Check for custom learning rates
        has_projector_lr = cfg.mm_projector_lr is not None and cfg.mm_projector_lr > 0
        has_vision_lr = cfg.vision_lr is not None and cfg.vision_lr > 0

        if not has_projector_lr:
            # Standard optimizer - single learning rate
            param_groups = self._create_standard_param_groups(decay_params)
        elif not has_vision_lr:
            # Two-way split: LLM+Vision vs Projector
            param_groups = self._create_two_way_param_groups(decay_params)
        else:
            # Three-way split: LLM vs Vision vs Projector
            param_groups = self._create_three_way_param_groups(decay_params)

        # Create optimizer
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        # Log configuration
        logger.info(f"Created optimizer with {len(param_groups)} parameter groups")
        logger.info(f"  Base LR: {cfg.learning_rate}")
        if has_projector_lr:
            logger.info(f"  Projector LR: {cfg.mm_projector_lr}")
        if has_vision_lr:
            logger.info(f"  Vision LR: {cfg.vision_lr}")

        return self.optimizer

    def _create_standard_param_groups(self, decay_params: list[str]) -> list[dict]:
        """Standard parameter groups with uniform learning rate."""
        model = self.model
        return [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_params and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_params and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

    def _create_two_way_param_groups(self, decay_params: list[str]) -> list[dict]:
        """Two-way split: LLM+Vision vs Projector."""
        model = self.model
        cfg = self.training_config

        projector_params = [
            n for n, _ in model.named_parameters() if "merger" in n or "projector" in n
        ]

        return [
            # LLM + Vision with decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_params
                    and n not in projector_params
                    and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            # LLM + Vision without decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_params
                    and n not in projector_params
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
            # Projector with decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_params and n in projector_params and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": cfg.mm_projector_lr,
            },
            # Projector without decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_params
                    and n in projector_params
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": cfg.mm_projector_lr,
            },
        ]

    def _create_three_way_param_groups(self, decay_params: list[str]) -> list[dict]:
        """Three-way split: LLM vs Vision vs Projector."""
        model = self.model
        cfg = self.training_config

        projector_params = [
            n for n, _ in model.named_parameters() if "merger" in n or "projector" in n
        ]
        vision_params = [
            n
            for n, _ in model.named_parameters()
            if "visual" in n and "merger" not in n
        ]

        return [
            # LLM with decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_params
                    and n not in projector_params
                    and n not in vision_params
                    and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            # LLM without decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_params
                    and n not in projector_params
                    and n not in vision_params
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
            # Vision with decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_params
                    and n not in projector_params
                    and n in vision_params
                    and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": cfg.vision_lr,
            },
            # Vision without decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_params
                    and n not in projector_params
                    and n in vision_params
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": cfg.vision_lr,
            },
            # Projector with decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_params and n in projector_params and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "lr": cfg.mm_projector_lr,
            },
            # Projector without decay
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_params
                    and n in projector_params
                    and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": cfg.mm_projector_lr,
            },
        ]

    # =========================================================================
    # Flash Attention Patches
    # =========================================================================
    def _apply_flash_attention_patches(self):
        """
        Apply flash attention patches for packed/flattened sequences.

        Only needed when data_flatten=True for efficient batch processing.
        """
        try:
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_module
            from flash_attn.flash_attn_interface import flash_attn_varlen_func
        except ImportError:
            logger.warning("Flash attention not available, skipping patches")
            return

        def _flash_attention_forward(
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            attention_mask: torch.Tensor,
            query_length: int,
            is_causal: bool,
            dropout: float = 0.0,
            softmax_scale: Optional[float] = None,
            use_top_left_mask: bool = False,
            softcap: Optional[float] = None,
            **kwargs,
        ):
            """Flash attention for packed sequences."""
            assert (
                query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
            )

            query_states = query_states.squeeze(0)
            key_states = key_states.squeeze(0)
            value_states = value_states.squeeze(0)
            cu_seqlens = attention_mask

            with torch.no_grad():
                max_seqlen = max(
                    cu_seqlens[idx + 1] - cu_seqlens[idx]
                    for idx in range(cu_seqlens.size(0) - 1)
                ).item()

            causal = (
                is_causal
                if not use_top_left_mask
                else (is_causal and query_length != 1)
            )

            flash_kwargs = {"softcap": softcap} if softcap is not None else {}

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                **flash_kwargs,
            )

            return attn_output.unsqueeze(0)

        def _update_causal_mask(
            self,
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            output_attentions,
        ):
            """Pass through attention mask for flash attention."""
            return attention_mask

        # Apply patches
        qwen_module._flash_attention_forward = _flash_attention_forward
        qwen_module.Qwen2_5_VLModel._update_causal_mask = _update_causal_mask

        logger.info("Applied flash attention patches for data flattening")

    # =========================================================================
    # Debug/Info Methods
    # =========================================================================
    def print_model_status(self):
        """Print detailed model training status."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING STATUS")
        print("=" * 60)

        # Vision encoder status
        self._print_component_status("visual", "Vision Encoder")

        # Merger/Projector status
        self._print_component_status(
            "visual.merger", "Vision-Language Connector", nested=True
        )

        # LLM status
        self._print_component_status("model", "Language Model")

        print("=" * 60)

    def _print_component_status(self, pattern: str, name: str, nested: bool = False):
        """Print trainability status for a model component."""
        try:
            if nested:
                parts = pattern.split(".")
                module = self.model
                for part in parts:
                    module = getattr(module, part)
            else:
                module = getattr(self.model, pattern)

            trainable = sum(1 for p in module.parameters() if p.requires_grad)
            total = sum(1 for p in module.parameters())

            status = "TRAINING" if trainable > 0 else "FROZEN"
            print(f"{name}: {status} ({trainable}/{total} params trainable)")

        except AttributeError:
            print(f"{name}: NOT FOUND")
