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

<!-- TODO: Add flash attention and test config -->

This repository demonstrates fine-tuning [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for radiology image captioning using the [ROCO dataset](https://huggingface.co/datasets/unsloth/Radiology_mini). It serves as part of the OpenCloudHub project and an accompanying master's thesis on MLOps, showcasing how to integrate Vision-Language Models into a reproducible ML pipeline.

**Why this exists:**

- MLflow doesn't natively support Vision-Language Models - this shows how to make it work using a custom PyFunc wrapper
- Demonstrates multimodal AI (image + text) in an MLOps context
- Shows how to combine HuggingFace Transformers with Ray for distributed GPU training
- Provides a reference for QLoRA/LoRA fine-tuning with proper experiment tracking
- Illustrates data versioning with DVC and prompt tracking through the full pipeline

This is not a production system - it's a learning resource and integration showcase.

______________________________________________________________________

<h2 id="features">✨ Features</h2>

- 🖼️ **Multimodal Training**: Fine-tune Qwen2.5-VL for image-to-text tasks
- ⚡ **QLoRA/LoRA Support**: Memory-efficient fine-tuning with 4-bit quantization
- 📊 **MLflow Integration**: Custom PyFunc wrapper for VLM tracking and serving
- 📦 **DVC Data Pipeline**: Versioned datasets with prompt tracking through lineage
- 🔀 **Ray Train**: Distributed GPU training with checkpointing
- 🚀 **Ray Serve**: Scalable inference with base64 image input
- ⚙️ **Config Separation**: Infrastructure (env vars) vs training params (YAML)
- 🐳 **Containerized**: Docker-based training for reproducibility

______________________________________________________________________

<h2 id="architecture">🏗️ Architecture</h2>

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline (DVC)                         │
├─────────────────────────────────────────────────────────────────────┤
│  HuggingFace  ──►  Download  ──►  Process  ──►  Analyze             │
│  Dataset           (raw)         (+ prompt)     (metadata)          │
│                                      │                              │
│                         MLflow Prompt Registry                      │
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
│  (model + prompt)       FastAPI + batching       JSON response      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Integration Points:**

1. **Prompt Lineage**: Prompts are versioned in MLflow, embedded during data processing, tracked through training, and used at inference
1. **Checkpoint Contents**: Model weights + processor + prompt_info.json (everything needed to serve)
1. **MLflow PyFunc**: Custom wrapper handles VLM loading since MLflow doesn't support vision-language natively

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU training)
- VS Code with DevContainers extension (recommended)
- Access to MLflow tracking server and MinIO (S3-compatible storage)

### Setup

1. **Clone the repository**

```bash
   git clone https://github.com/opencloudhub/ai-qwen-demo.git
   cd ai-qwen-demo
```

2. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally**:

```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync --dev
```

3. **Configure environment**

```bash
   cp .env.example .env
   # Edit .env with your MLflow/MinIO credentials
```

Required environment variables:

```bash
    AWS_ACCESS_KEY_ID=admin
    AWS_SECRET_ACCESS_KEY=admin123
    AWS_ENDPOINT_URL=https://minio-api.internal.opencloudhub.org
    DVC_REPO=https://github.com/OpenCloudHub/data-registry

    # MLflow
    MLFLOW_TRACKING_URI=http://localhost:5000
    MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    MLFLOW_TRACKING_INSECURE_TLS=true
    MLFLOW_LOGGING_LEVEL=DEBUG
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
    MLFLOW_EXPERIMENT_NAME=roco-radiology-vqa
    MLFLOW_REGISTERED_MODEL_NAME=dev.roco-radiology-vqa

    # Ray
    RAY_TRAIN_V2_ENABLED=1
    RAY_STORAGE_PATH=/tmp/ray_results

    # Tracking
    ARGO_WORKFLOW_UID=DEV
    DOCKER_IMAGE_TAG=DEV
    DVC_DATA_VERSION_TAG=roco-radiology-v1.0.0
```

4. **Start Ray**

```bash
   ray start --head --num-gpus 1
```

______________________________________________________________________

<h2 id="configuration">⚙️ Configuration</h2>

Configuration is split into two categories by design:

### Infrastructure Config (Environment Variables)

CI/CD controls these - developers cannot override via YAML:

| Variable                       | Description                                      | Required              |
| ------------------------------ | ------------------------------------------------ | --------------------- |
| `DVC_DATA_VERSION`             | Data version tag (e.g., `radiology-mini-v1.0.0`) | Yes                   |
| `DVC_REPO`                     | DVC repository URL                               | No (has default)      |
| `MLFLOW_TRACKING_URI`          | MLflow server URL                                | No (has default)      |
| `MLFLOW_EXPERIMENT_NAME`       | Experiment name                                  | No (has default)      |
| `MLFLOW_REGISTERED_MODEL_NAME` | Model registry name                              | No (has default)      |
| `RAY_NUM_WORKERS`              | Number of Ray workers                            | No (default: 1)       |
| `ARGO_WORKFLOW_UID`            | Workflow tracking ID                             | No (default: "local") |
| `DOCKER_IMAGE_TAG`             | Image tag for reproducibility                    | No (default: "dev")   |

### Training Config (YAML)

Developers control these via config files:

```yaml
# configs/qlora.yaml
data:
  max_pixels: 451584
  min_pixels: 12544

model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  tune_vision: false
  tune_mlp: true      # Train vision-language connector
  tune_llm: false

training:
  max_steps: 100
  batch_size: 1
  learning_rate: 2.0e-4

  lora:
    enabled: true
    r: 64
    alpha: 128

  quantization:
    enabled: true
    type: "nf4"
    load_in_4bit: true

  optimization:
    gradient_checkpointing: true
    bf16: true
```

To make it even more flexible one could look for integrations with e.g. [Hydra](https://hydra.cc/).

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

The external pipeline handles:

1. **download**: Fetch ROCO dataset from HuggingFace, save as images + captions JSON
1. **process**: Load prompt from MLflow, convert to Qwen conversation format, train/test split
1. **analyze**: Compute statistics, embed prompt metadata into `metadata.json`

### Prompt Tracking

Prompts are versioned in MLflow and flow through the entire pipeline:

```text
MLflow Registry → DVC Process → metadata.json → Training → Checkpoint → Serving
```

The model always uses the exact prompt it was trained with.

______________________________________________________________________

<h2 id="training">🏋️ Training</h2>

### Local Training

```bash
# Run training
python src/training/train.py --config configs/debug_qlora.yaml
```

### Via Ray Job API

This spins up a local cluster to submit the job to for testing close to how workflows would run in production.
```bash
ray start --head --num-cpus 8
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
  --working-dir . \
  -- python src/training/train.py --config configs/debug_qlora.yaml
```

### Training Methods

| Method | Config                                            | Memory | Use Case                        |
| ------ | ------------------------------------------------- | ------ | ------------------------------- |
| QLoRA  | `lora.enabled=true`, `quantization.enabled=true`  | ~8GB   | Default, fits on most GPUs      |
| LoRA   | `lora.enabled=true`, `quantization.enabled=false` | ~16GB  | Better quality, needs more VRAM |
| Full   | Both disabled                                     | ~24GB+ | Best quality, needs large GPU   |

### What Gets Logged

- **MLflow Params**: Model name, training method, hyperparameters, DVC version, prompt version
- **MLflow Tags**: `argo_workflow_uid`, `docker_image_tag`, `dvc_data_version`, `prompt_name`, `prompt_version`
- **Checkpoint**: Model weights, processor, `prompt_info.json`

______________________________________________________________________

<h2 id="serving">🚀 Serving</h2>

### Start Serving

```bash
serve run src.serving.serve:app_builder \
  model_uri="models:/dev.roco-radiology-vqa/1" \
  --reload
```

### API Usage

**Endpoint**: `POST /predict`

```bash
# Encode image to base64
BASE64_IMAGE=$(base64 -w 0 test_image.jpg)

# Call API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$BASE64_IMAGE\"}"
```

**Response:**

```json
{
  "description": "Chest X-ray showing bilateral pulmonary infiltrates..."
}
```

### Swagger UI

Access interactive docs at `http://localhost:8000/docs`

### How Serving Works

1. Ray Serve loads model from MLflow using the custom PyFunc wrapper
1. Wrapper reads `prompt_info.json` from checkpoint artifacts
1. Each request: decode base64 → apply training prompt → generate → return text

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```text
ai-qwen-demo/
├── src/
│   ├── training/
│   │   ├── train.py              # Entry point
│   │   ├── config.py             # InfraConfig (env) + TrainConfig (YAML)
│   │   ├── trainer.py            # Custom HF Trainer with multi-LR support
│   │   ├── callbacks.py          # Ray checkpoint callback
│   │   ├── data.py               # Dataset and collator
│   │   ├── dvc_loader.py         # Load data from DVC
│   │   ├── mlflow_wrapper.py     # PyFunc wrapper for VLM
│   │   └── log.py                # Structured logging
│   └── serving/
│       └── serve.py              # Ray Serve deployment
├── configs/
│   ├── debug_qlora.yaml          # Quick test config
│   ├── debug_lora.yaml           # LoRA without quantization
│   └── qlora.yaml                # Full training config
├── tests/
├── .devcontainer/
├── .github/workflows/
│   ├── build.yaml                # Build and push Docker image
│   └── train.yaml                # Trigger training in cluster
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

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

Project Link: [https://github.com/opencloudhub/ai-qwen-demo](https://github.com/opencloudhub/ai-qwen-demo)

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) - Vision-Language Model
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Ray](https://ray.io/) - Distributed training and serving
- [DVC](https://dvc.org/) - Data version control
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

<p align="right">(<a href="#readme-top">back to top</a>)</p>
