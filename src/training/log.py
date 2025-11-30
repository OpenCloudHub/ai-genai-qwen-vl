"""
Logging for Ray Training

Setup:
- Ray's native JSON logging
- Rich console for visual separators

Usage:
    from log import get_logger, log_section, log_success, get_ray_logging_config

    # In driver, before ray.init()
    ray.init(logging_config=get_ray_logging_config())

    logger = get_logger(__name__)
    log_section("Training Started")
    logger.info("Loading model", extra={"model": "qwen-3b"})
    log_success("Model loaded!")
"""

import logging
import os
from typing import Any

import ray
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# =============================================================================
# Rich Console (for visual output only, not logging)
# =============================================================================
console = Console(force_terminal=True, stderr=True)


# =============================================================================
# Visual Helpers (print-based, not logging)
# =============================================================================
def log_section(title: str, emoji: str = "📌") -> None:
    """Print a visual section separator."""
    console.rule(f"[bold cyan]{emoji} {title}[/]", style="cyan")


def log_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✅ {message}[/]")


def log_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠️  {message}[/]")


def log_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]❌ {message}[/]")


def log_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[cyan]ℹ️  {message}[/]")


def log_step(step: int, total: int, message: str) -> None:
    """Print a step progress message."""
    console.print(f"[dim]({step}/{total})[/] {message}")


def log_key_value(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair."""
    spaces = " " * indent
    console.print(f"{spaces}[dim]{key}:[/] [cyan]{value}[/]")


def log_config_table(config: dict, title: str = "Configuration") -> None:
    """Print a configuration as a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(str(key), str(value))

    console.print(table)


def log_panel(content: str, title: str = "", style: str = "cyan") -> None:
    """Print content in a panel box."""
    console.print(Panel(content, title=title, border_style=style))


def log_training_summary(
    model: str,
    method: str,
    dataset: str,
    steps: int,
    batch_size: int,
    lr: float,
) -> None:
    """Print a training summary panel."""
    content = f"""[bold]Model:[/] {model}
[bold]Method:[/] {method}
[bold]Dataset:[/] {dataset}
[bold]Steps:[/] {steps}
[bold]Batch Size:[/] {batch_size}
[bold]Learning Rate:[/] {lr}"""

    console.print(
        Panel(content, title="🚀 Training Configuration", border_style="green")
    )


def log_results_summary(metrics: dict, checkpoint_path: str = None) -> None:
    """Print a results summary panel."""
    lines = [f"[bold]{k}:[/] {v}" for k, v in metrics.items()]
    if checkpoint_path:
        lines.append(f"\n[bold]Checkpoint:[/] {checkpoint_path}")

    console.print(Panel("\n".join(lines), title="📊 Results", border_style="green"))


# =============================================================================
# Logger Setup
# =============================================================================
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Ray's LoggingConfig handles the formatting (JSON).
    Just get the logger and use it.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = True  # Let Ray handle output

    return logger


def get_ray_logging_config() -> ray.LoggingConfig:
    """
    Get Ray LoggingConfig for structured JSON logs.

    Usage:
        ray.init(logging_config=get_ray_logging_config())

    The JSON logs automatically include:
    - job_id, worker_id, node_id
    - task_id, task_name (for tasks)
    - actor_id, actor_name (for actors)
    - timestamp_ns
    """
    return ray.LoggingConfig(
        encoding="JSON",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


# =============================================================================
# Structured Logging Helpers
# =============================================================================
def log_config(logger: logging.Logger, config: dict, name: str = "config") -> None:
    """Log configuration as structured data."""
    logger.info(f"Loaded {name}", extra={name: config})


def log_metrics(
    logger: logging.Logger, metrics: dict[str, Any], step: int = None
) -> None:
    """Log metrics as structured data."""
    extra = {"metrics": metrics}
    if step is not None:
        extra["step"] = step
    logger.info("Metrics", extra=extra)


def log_event(logger: logging.Logger, event: str, **kwargs) -> None:
    """Log a structured event with arbitrary fields."""
    logger.info(event, extra=kwargs)
