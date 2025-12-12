<a id="readme-top"></a>

<!-- PROJECT LOGO & TITLE -->

<div align="center">
  <a href="https://github.com/opencloudhub">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg">
    <!-- Fallback -->
    <img alt="OpenCloudHub Logo" src="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg" style="max-width:700px; max-height:175px;">
  </picture>
  </a>

<h1 align="center">Qwen2.5-VL Radiology - MLOps Demo</h1>

<p align="center">
    Vision-Language Model fine-tuning for radiology image captioning, demonstrating multimodal MLOps with Ray, MLflow, and DVC.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#thesis-context">Thesis Context</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#data-pipeline">Data Pipeline</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#serving">Serving</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🏥 About</h2>

This repository demonstrates fine-tuning [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for radiology image captioning using the [ROCO dataset](https://huggingface.co/datasets/unsloth/Radiology_mini). It serves as part of the OpenCloudHub project and an accompanying master's thesis on MLOps, showcasing how to integrate Vision-Language Models into a reproducible ML pipeline.

**Why this exists:**

- MLflow doesn't natively support Vision-Language Models - this shows how to make it work using a custom PyFunc wrapper
- Demonstrates multimodal AI (image + text) in an MLOps context
- Shows how to combine HuggingFace Transformers with Ray for GPU training
- Provides a reference for QLoRA/LoRA fine-tuning with proper experiment tracking
- Illustrates data versioning with DVC and prompt tracking through the full pipeline

This is not a production system - it's a learning resource and integration showcase.

______________________________________________________________________

<h2 id="thesis-context">📚 Thesis Context</h2>

<!-- TODO:  -->

### Key Technical Contributions

| Challenge                 | Solution                            | Location                         |
| ------------------------- | ----------------------------------- | -------------------------------- |
| VLM experiment tracking   | Custom MLflow PyFunc wrapper        | `src/training/mlflow_wrapper.py` |
| Prompt-model consistency  | Prompt baked into checkpoint        | `src/training/callbacks.py`      |
| Memory-efficient training | QLoRA with gradient checkpointing   | `src/training/trainer.py`        |
| Memory-efficient serving  | 4-bit quantized inference           | `SERVE_QUANTIZED=true` env var   |
| Reproducible data         | DVC versioning with metadata        | `src/training/dvc_loader.py`     |
| Config separation         | Env vars (infra) vs YAML (training) | `src/training/config.py`         |

### Reading the Code

For thesis reviewers, the recommended reading order is:

1. **[config.py](src/training/config.py)** - Understand the configuration philosophy (infra vs training separation)
1. **[train.py](src/training/train.py)** - Entry point showing the driver-worker pattern
1. **[mlflow_wrapper.py](src/training/mlflow_wrapper.py)** - The key innovation: PyFunc wrapper for VLMs
1. **[callbacks.py](src/training/callbacks.py)** - How prompts and processors are bundled with checkpoints
1. **[serve.py](src/serving/serve.py)** - How models are loaded and served via Ray Serve

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🖼️ **Multimodal Training**: Fine-tune Qwen2.5-VL for image-to-text tasks
- ⚡ **QLoRA/LoRA Support**: Memory-efficient fine-tuning with 4-bit quantization
- 📊 **MLflow Integration**: Custom PyFunc wrapper for VLM tracking and serving
- 📦 **DVC Data Pipeline**: Versioned datasets with prompt tracking through lineage
- 🔀 **Ray Train**: Distributed GPU training with fault tolerance and checkpointing
- 🚀 **Ray Serve**: Scalable inference with hot model reloading
- ⚙️ **Config Separation**: Infrastructure (env vars) vs training params (YAML)
- 🐳 **Containerized**: Docker-based training for reproducibility

______________________________________________________________________

<h2 id="architecture">🏗️ Architecture</h2>

### System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline (DVC)                         │
├─────────────────────────────────────────────────────────────────────┤
│  HuggingFace  ──►  Download  ──►  Process  ──►  Analyze             │
│  Dataset           (raw)         (+ prompt)     (metadata)          │
│                                      │                              │
│             MLflow Prompt Registry / DVC Data Registry              │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Training (Ray Train)                           │
├─────────────────────────────────────────────────────────────────────┤
│  Load DVC Data  ──►  Qwen2.5-VL  ──►  QLoRA/LoRA  ──►  Checkpoint   │
│  (with prompt)       + Processor      Fine-tuning      + Processor  │
│                                                        + Prompt     │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Model Registry (MLflow)                        │
├─────────────────────────────────────────────────────────────────────┤
│  PyFunc Wrapper  ──►  Artifacts (checkpoint)  ──►  Model Versions   │
│  (custom VLM)         + prompt_info.json           dev.model/1      │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Serving (Ray Serve)                           │
├─────────────────────────────────────────────────────────────────────┤
│  Load from MLflow  ──►  QwenVLDeployment  ──►  /predict (base64)    │
│  (model + prompt)       FastAPI + scaling       JSON response       │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Architecture (Ray Train)

The training uses a **driver-worker pattern** with single-GPU execution:

```text
┌──────────────────────────────────────────────────────────────────┐
│                        HEAD NODE (Driver)                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  train_driver()                                            │  │
│  │  • Downloads data from DVC (once)                          │  │
│  │  • Creates MLflow run with tags                            │  │
│  │  • Configures Ray TorchTrainer                             │  │
│  │  • Registers final model to MLflow                         │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                         GPU WORKER                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  train_worker()                                            │  │
│  │  • Loads model with QLoRA/LoRA                             │  │
│  │  • Creates PyTorch Dataset from local path                 │  │
│  │  • Runs HuggingFace Trainer                                │  │
│  │  • Reports checkpoints to Ray                              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

> **Note:** This demo uses single-GPU training. For distributed DDP training, one would load
> only shards of data in the worker nodes. An example of this using
> Ray Data for distributed training with data sharding,
> see [Ray Data with PyTorch Lightning](https://github.com/OpenCloudHub/ai-dl-lightning).

### Serving Architecture (Ray Serve)

```text
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client    │────▶│    Ray Serve     │────▶│     MLflow      │
│  (FastAPI)  │     │   (Scaling)      │     │    (Models)     │
└─────────────┘     └──────────────────┘     └─────────────────┘
       │                     │
       │              ┌──────┴──────┐
       │              ▼             ▼
       │         ┌────────┐   ┌────────┐
       │         │Replica │   │Replica │  (Autoscaling)
       │         │  0.25  │   │  0.25  │  (Fractional GPU)
       │         │  GPU   │   │  GPU   │
       │         └────────┘   └────────┘
       │
  Endpoints:
  GET  /        → Service info
  GET  /health  → Kubernetes readiness probe
  GET  /info    → Model metadata from MLflow
  POST /predict → Image analysis (file upload)
```

### Key Integration Points

1. **Prompt Lineage**: Prompts are versioned in MLflow, embedded during data processing, tracked through training, and used at inference
1. **Checkpoint Contents**: Model weights + processor + prompt_info.json (everything needed to serve)
1. **MLflow PyFunc**: Custom wrapper handles VLM loading since MLflow doesn't support vision-language natively
1. **Quantized Serving**: `SERVE_QUANTIZED=true` enables 4-bit inference (~4GB vs ~8GB VRAM)

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU training)
- VS Code with DevContainers extension (recommended)
- Access to MLflow tracking server and MinIO (S3-compatible storage)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/opencloudhub/ai-radiology-qwen.git
cd ai-radiology-qwen
```

2. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

3. **Configure environment**

Two environment files are provided:

| File            | Use Case                                           |
| --------------- | -------------------------------------------------- |
| `.env.docker`   | Local Docker Compose (MLflow + MinIO on localhost) |
| `.env.minikube` | Minikube/Kubernetes (internal cluster URLs)        |

```bash
# For local Docker Compose setup
set -a && source .env.docker && set +a

# Or for Minikube
set -a && source .env.minikube && set +a
```

See [Configuration](#configuration) for details on all environment variables.

4. **Start Ray**

```bash
ray start --head --num-gpus 1 --num-cpus 12
```

______________________________________________________________________

<h2 id="configuration">⚙️ Configuration</h2>

Configuration is split into two categories by design - this separation is intentional:

### Infrastructure Config (Environment Variables)

**CI/CD controls these** - developers should not override via YAML. This ensures reproducibility and prevents accidental use of wrong endpoints.

| Variable                       | Description                   | Required         |
| ------------------------------ | ----------------------------- | ---------------- |
| `DVC_DATA_VERSION`             | Data version tag              | **Yes**          |
| `DVC_REPO`                     | DVC repository URL            | No (has default) |
| `MLFLOW_TRACKING_URI`          | MLflow server URL             | No (has default) |
| `MLFLOW_EXPERIMENT_NAME`       | Experiment name               | No (has default) |
| `MLFLOW_REGISTERED_MODEL_NAME` | Model registry name           | No (has default) |
| `RAY_NUM_WORKERS`              | Number of Ray workers         | No (default: 1)  |
| `RAY_GPU_FRACTION`             | GPU fraction per worker (0-1) | No (default: 1)  |
| `ARGO_WORKFLOW_UID`            | Workflow tracking ID          | No               |
| `DOCKER_IMAGE_TAG`             | Image tag for reproducibility | No               |

### Training Config (YAML)

**Developers control these** via config files in `configs/`:

```yaml
# configs/qlora.yaml
data:
  max_pixels: 451584      # Max image resolution
  min_pixels: 12544       # Min image resolution
  sampling_percent: 1.0   # Use 100% of data

model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  tune_vision: false      # Freeze vision encoder
  tune_mlp: true          # Train vision-language connector
  tune_llm: false         # Freeze language model

training:
  max_steps: 100
  batch_size: 1
  learning_rate: 2.0e-4
  mm_projector_lr: 2.0e-5  # Separate LR for projector

  lora:
    enabled: true
    r: 64                  # LoRA rank
    alpha: 128             # LoRA alpha
    target_modules: "all-linear"

  quantization:
    enabled: true          # Enable 4-bit quantization
    type: "nf4"            # NormalFloat4
    double_quant: true     # Double quantization

  optimization:
    gradient_checkpointing: true  # Save memory
    flash_attention: false        # Requires flash-attn package
    bf16: true
```

______________________________________________________________________

<h2 id="data-pipeline">📦 Data Pipeline</h2>

Data preparation and versioning is managed in a separate repository using DVC pipelines.

### External Data Registry

The data pipeline lives at: [**OpenCloudHub/data-registry**](https://github.com/OpenCloudHub/data-registry/tree/main/pipelines/roco-radiology)

This separation allows:

- **Decoupled versioning**: Data versions are independent of model code
- **Reusable pipelines**: Multiple training projects can share the same data
- **Clear lineage**: DVC tracks data provenance with prompt metadata

### Pipeline Stages

```text
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Download  │────▶│  Process   │────▶│  Analyze   │
│            │     │            │     │            │
│ HuggingFace│     │ + Prompt   │     │ Statistics │
│ → Images   │     │ → Qwen fmt │     │ → metadata │
└────────────┘     └────────────┘     └────────────┘
                         ▲
                         │
                   ┌─────┴─────┐
                   │  MLflow   │
                   │  Prompt   │
                   │  Registry │
                   └───────────┘
```

### Prompt Tracking

Prompts are versioned in MLflow and flow through the entire pipeline:

```text
MLflow Registry → DVC Process → metadata.json → Training → Checkpoint → Serving
```

The model always uses the exact prompt it was trained with - this prevents train-serve skew.

______________________________________________________________________

<h2 id="training">🏋️ Training</h2>

### Quick Start

```bash
# Local training
python src/training/train.py --config configs/debug_qlora.yaml

# Via Ray Job API (closer to production)
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
  --working-dir . \
  -- python src/training/train.py --config configs/debug_qlora_flash.yaml
```

### Training Methods

| Method    | Config                                            | VRAM   | Use Case                |
| --------- | ------------------------------------------------- | ------ | ----------------------- |
| **QLoRA** | `lora.enabled=true`, `quantization.enabled=true`  | ~9GB   | Default, single GPU     |
| LoRA      | `lora.enabled=true`, `quantization.enabled=false` | ~16GB  | Better quality          |
| Full      | Both disabled                                     | ~24GB+ | Best quality, multi-GPU |

### GPU Memory Usage (RTX 4070 Ti Super 16GB)

| Workload                         | VRAM        |
| -------------------------------- | ----------- |
| QLoRA Training                   | ~9GB        |
| Serving (unquantized)            | ~8GB        |
| Serving (quantized)              | ~4GB        |
| **Training + Quantized Serving** | **~13GB** ✓ |

This allows demonstrating training and serving simultaneously on a single GPU.

### What Gets Logged to MLflow

**Parameters:**

- `model_name`, `training_method`, `batch_size`, `learning_rate`, `max_steps`
- `lora_r`, `lora_alpha`, `quantization_enabled`, `flash_attention`
- `dvc_data_version`

**Tags:**

- `argo_workflow_uid`, `docker_image_tag`, `prompt_name`, `prompt_version`

**Artifacts:**

- Model weights (LoRA adapters or full model)
- Processor configuration
- `prompt_info.json` (ensures serving uses training prompt)

______________________________________________________________________

<h2 id="serving">🚀 Serving</h2>

### Start Serving

```bash
# Standard serving
serve run src.serving.serve:app_builder \
  model_uri="models:/dev.roco-radiology-vqa/1"

# With quantization (saves ~50% VRAM)
SERVE_QUANTIZED=true serve run src.serving.serve:app_builder \
  model_uri="models:/dev.roco-radiology-vqa/1"
```

### API Endpoints

| Endpoint   | Method | Description                  |
| ---------- | ------ | ---------------------------- |
| `/`        | GET    | Service info and links       |
| `/health`  | GET    | Kubernetes readiness probe   |
| `/info`    | GET    | Model metadata from MLflow   |
| `/predict` | POST   | Image analysis (file upload) |
| `/docs`    | GET    | Swagger UI                   |

### Usage Example

```bash
# Single image
curl -X POST http://localhost:8000/predict \
  -F "files=@chest_xray.jpg"

# Multiple images
curl -X POST http://localhost:8000/predict \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

**Response:**

```json
{
  "predictions": [
    {
      "image_index": 0,
      "filename": "chest_xray.jpg",
      "text": "Chest X-ray showing bilateral pulmonary infiltrates..."
    }
  ],
  "num_images": 1,
  "model_uri": "models:/dev.roco-radiology-vqa/1",
  "processing_time_ms": 1234.5
}
```

### Hot Model Reloading

Update the model without restarting the service:

```python
import requests

requests.post(
    "http://localhost:8000/reconfigure",
    json={"model_uri": "models:/dev.roco-radiology-vqa/2"},
)
```

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```text
ai-radiology-qwen/
├── src/
│   ├── training/                 # Training pipeline
│   │   ├── train.py              # Entry point (driver-worker pattern)
│   │   ├── config.py             # Configuration (env + YAML separation)
│   │   ├── trainer.py            # Custom HF Trainer with multi-LR
│   │   ├── callbacks.py          # Checkpoint callback (bundles prompt)
│   │   ├── data.py               # Dataset and collator (3D RoPE)
│   │   ├── dvc_loader.py         # Data versioning integration
│   │   ├── mlflow_wrapper.py     # PyFunc wrapper (key innovation)
│   │   └── log.py                # JSON logging for observability
│   │
│   └── serving/                  # Inference API
│       ├── serve.py              # Ray Serve + FastAPI
│       └── schemas.py            # Pydantic request/response models
│
├── configs/                      # Training configurations
│   ├── debug_qlora.yaml          # Quick test (10 steps)
│   ├── debug_qlora_flash.yaml    # With flash attention
│   ├── debug_lora.yaml           # LoRA without quantization
│   └── demo.yaml                 # Demo configuration
│
├── data/                         # Sample data for testing
│   └── radiology_mini/
│
├── notebooks/                    # Exploration notebooks
│
├── .github/workflows/            # CI/CD
│   ├── ci-code-quality.yaml      # Linting, formatting
│   ├── ci-docker-build-push.yaml # Build container
│   └── train.yaml                # Trigger training
│
├── Dockerfile                    # Multi-stage build
├── pyproject.toml                # Dependencies (uv)
└── README.md                     # You are here
```

### Module Descriptions

| Module              | Purpose                                                             |
| ------------------- | ------------------------------------------------------------------- |
| `train.py`          | Orchestrates training: loads config, sets up MLflow, runs Ray Train |
| `config.py`         | Separates infrastructure (env) from training (YAML) configuration   |
| `trainer.py`        | Custom HuggingFace Trainer with per-component learning rates        |
| `callbacks.py`      | Bundles processor and prompt into checkpoints for serving           |
| `data.py`           | Handles Qwen's conversation format and 3D position encoding         |
| `mlflow_wrapper.py` | Enables MLflow to serve VLMs (the key innovation)                   |
| `serve.py`          | FastAPI application with health checks and batch inference          |

______________________________________________________________________

<h2 id="contributing">👥 Contributing</h2>

Contributions are welcome! This project follows OpenCloudHub's contribution standards.

Please see our [Contributing Guidelines](https://github.com/opencloudhub/.github/blob/main/.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/opencloudhub/.github/blob/main/.github/CODE_OF_CONDUCT.md) for more details.

______________________________________________________________________

<h2 id="license">📄 License</h2>

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

______________________________________________________________________

<h2 id="contact">📬 Contact</h2>

Organization Link: [https://github.com/OpenCloudHub](https://github.com/OpenCloudHub)

Project Link: [https://github.com/opencloudhub/ai-radiology-qwen](https://github.com/opencloudhub/ai-radiology-qwen)

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [Official QwenVL Repo](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) - Vision-Language Model
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Ray](https://ray.io/) - Distributed training and serving
- [DVC](https://dvc.org/) - Data version control
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

<p align="right">(<a href="#readme-top">back to top</a>)</p>
