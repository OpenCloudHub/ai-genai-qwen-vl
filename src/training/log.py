"""
Logging Utilities
=================

Loguru-based logging with Rich console output for visual helpers.

- Structured loguru format with timestamps and colored output
- Intercepts stdlib logging (for mlflow, transformers, ray, etc.)
- Rich console helpers for visual sections and summaries

Functions
---------
get_logger : Returns a bound loguru logger with name context
log_section, log_success, log_error, log_info : Rich console helpers
log_training_summary, log_results_summary : Formatted training output

Usage
-----
    from src.training.log import get_logger, log_section

    logger = get_logger(__name__)
    logger.info("Loading model", model="qwen-3b")
    log_section("Training", "🚀")
"""

import logging
import sys
import warnings
from os import getenv
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from ray._private.ray_logging.logging_config import LoggingConfig
# =============================================================================
# Configuration
# =============================================================================
LOG_LEVEL = getenv("LOG_LEVEL", "INFO").upper()

# =============================================================================
# Rich Console (for visual output only, not logging)
# =============================================================================
console = Console(force_terminal=True, stderr=True)

# =============================================================================
# Loguru Setup
# =============================================================================
logger.remove()

# Bound loggers (from get_logger)
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> | <level>{message}</level>",
    colorize=True,
    filter=lambda record: "name" in record["extra"],
)

# Intercepted stdlib loggers
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <dim>{name}</dim> | <level>{message}</level>",
    colorize=True,
    filter=lambda record: "name" not in record["extra"],
)


# =============================================================================
# Stdlib Interception (mlflow, transformers, ray, etc. → loguru)
# =============================================================================
class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Silence noisy libraries unless DEBUG
if LOG_LEVEL != "DEBUG":
    for name in [
        "mlflow",
        "urllib3",
        "botocore",
        "boto3",
        "fsspec",
        "git",
        "ray",
        "httpx",
        "httpcore",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore")


# =============================================================================
# Logger Factory
# =============================================================================
def get_logger(name: str):
    """
    Get a named loguru logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Bound loguru logger with name context

    Usage:
        logger = get_logger(__name__)
        logger.info("Loading model", model="qwen-3b")
    """
    return logger.bind(name=name)   


# =============================================================================
# Visual Helpers (Rich console, not logging)
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
