# syntax=docker/dockerfile:1
#==============================================================================#
# Build arguments
#==============================================================================#
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG NVIDIA_TAG=25.06-py3
ARG RAY_VERSION=2.51.0
ARG RAY_PY_TAG=py${PYTHON_MAJOR}${PYTHON_MINOR}-gpu
ARG UV_VERSION=0.9.13

#==============================================================================#
# Stage: UV binary source
# Workaround: --from in COPY doesn't support ARG variable substitution
#==============================================================================#
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_source

#==============================================================================#
# Stage: Base image with NVIDIA PyTorch + UV + build tools
# - NVIDIA base provides CUDA toolkit for flash-attn compilation
# - UV for fast Python dependency management
# - Build tools for compiling native extensions
#==============================================================================#
FROM nvcr.io/nvidia/pytorch:${NVIDIA_TAG} AS base

# Copy UV binary from source stage
COPY --from=uv_source /uv /uvx /usr/local/bin/

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/project

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src"

#==============================================================================#
# Stage: Training dependencies with flash-attn
# - Expensive flash-attn CUDA kernel compilation (~10-20 min)
# - Cached separately to avoid rebuilds on code changes
#==============================================================================#
FROM base AS training_deps

COPY pyproject.toml uv.lock ./

# Flash-attn build configuration
ENV MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

# Install training dependencies (flash-attn built here)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra training --no-dev --no-install-project

#==============================================================================#
# Stage: DEVELOPMENT
# - Full dependencies (training + serving + dev + test)
# - Source mounted via devcontainer, not copied
#==============================================================================#
FROM base AS dev

COPY pyproject.toml uv.lock ./

# Copy venv with pre-built flash-attn wheel
COPY --from=training_deps /workspace/project/.venv /workspace/project/.venv

# Install all remaining dependencies (fast - flash-attn already built)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --all-groups --no-install-project

ENV ENVIRONMENT=development

#==============================================================================#
# Stage: TRAINING
# - Production training image
# - Minimal: only training dependencies + source code
#==============================================================================#
FROM base AS training

COPY pyproject.toml uv.lock ./

# Copy venv with pre-built flash-attn wheel
COPY --from=training_deps /workspace/project/.venv /workspace/project/.venv

COPY --chown=ray:users src/ ./src/

ENV ENVIRONMENT=training

#==============================================================================#
# Stage: SERVING
# - Production GPU serving image
# - Ray Serve base with serving dependencies
#==============================================================================#
ARG RAY_VERSION
ARG RAY_PY_TAG
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS serving

WORKDIR /workspace/project

# Copy UV binary
COPY --from=uv_source /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./

# Install base + serving dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra serving --no-dev --no-install-project

COPY --chown=ray:users src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project:/workspace/project/src" \
    ENVIRONMENT=production