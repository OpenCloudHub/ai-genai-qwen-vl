"""
Model Serving Application
=========================

Ray Serve + FastAPI application for Qwen2.5-VL inference.

Loads models from MLflow, supports hot reloading, fractional GPU sharing,
and Kubernetes-compatible health checks.

Classes
-------
QwenVLClassifier : Ray Serve deployment with FastAPI endpoints
AppBuilderArgs : Pydantic model for serve CLI arguments

Endpoints
---------
GET /        : Service info
GET /health  : Health check (Kubernetes readiness probe)
GET /info    : Model metadata from MLflow
POST /predict: Image analysis (file upload)

Usage
-----
    $ serve run src.serving.serve:app_builder model_uri="models:/qwen-vl/1"

Environment Variables
---------------------
SERVE_QUANTIZED=true : Enable 4-bit quantization (~50% VRAM reduction)

See Also
--------
README.md : Full API documentation and deployment guide
"""

import base64

# Use standard logging for Ray Serve compatibility (not Ray Train's JSON logger)
import logging
from datetime import datetime, timezone
from typing import List

import mlflow
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application

from src.serving.schemas import (
    APIStatus,
    ErrorResponse,
    HealthResponse,
    ImagePrediction,
    InferenceParams,
    ModelInfo,
    PredictionResponse,
    RootResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

app = FastAPI(
    title="🤖 Qwen2.5-VL Vision-Language API",
    description="Medical image analysis using Qwen2.5-VL via Ray Serve + MLflow",
    version="1.0.0",
)


@serve.deployment(
    # ray_actor_options={
    #     "num_gpus": 0.25
    # },  # FIXME: This is usually enabled by the serving cluster config, you can uncomment if needed for local testing
)
@serve.ingress(app)
class QwenRadiologyVQA:
    def __init__(
        self,
        model_uri: str | None = None,
        inference_params: InferenceParams = InferenceParams(),
    ) -> None:
        """Initialize the classifier, optionally with a model URI."""
        logger.info("🤖 Initializing Qwen2.5-VL Vision-Language Service")
        self.status = APIStatus.NOT_READY
        self.model = None
        self.model_info: ModelInfo | None = None
        self.start_time = datetime.now(timezone.utc)
        self.inference_params = inference_params

        # Load model if URI provided at init
        if model_uri:
            try:
                self._load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load model during initialization: {e}")
                self.status = APIStatus.UNHEALTHY

    def _load_model(self, model_uri: str) -> None:
        """Internal method to load model and fetch metadata."""
        logger.info(f"📦 Loading model from: {model_uri}")
        self.status = APIStatus.LOADING

        try:
            # Get model info first to validate URI
            info = mlflow.models.get_model_info(model_uri)

            # Get training run metadata
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(info.run_id)

            # Extract metadata from run
            base_model = run.data.params.get("base_model", "unknown")
            training_method = run.data.params.get("training_method", "unknown")

            # Extract training timestamp
            training_timestamp = datetime.fromtimestamp(
                run.info.start_time / 1000.0, tz=timezone.utc
            )

            # Load the model
            logger.info("Loading PyFunc model from MLflow...")
            self.model = mlflow.pyfunc.load_model(model_uri)

            # Build ModelInfo
            self.model_info = ModelInfo(
                model_uri=model_uri,
                model_uuid=info.model_uuid,
                run_id=info.run_id,
                model_signature=info.signature.to_dict() if info.signature else None,
                training_timestamp=training_timestamp,
                base_model=base_model,
                training_method=training_method,
            )

            self.status = APIStatus.HEALTHY
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Model UUID: {self.model_info.model_uuid}")
            logger.info(f"   Run ID: {self.model_info.run_id}")
            logger.info(f"   Base Model: {base_model}")
            logger.info(f"   Training Method: {training_method}")

        except mlflow.exceptions.MlflowException as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ MLflow error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model from MLflow: {str(e)}",
            )
        except Exception as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ Unexpected error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error loading model: {str(e)}",
            )

    def reconfigure(self, config: dict) -> None:
        """Handle model updates without restarting the deployment.

        Check: https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html

        Update via: serve.run(..., user_config={"model_uri": "new_uri"})
        """
        new_model_uri = config.get("model_uri")

        if not new_model_uri:
            logger.warning("⚠️ No model_uri provided in config")
            return

        # If no model loaded yet, load it
        if self.model_info is None:
            logger.info("🆕 Initial model load via reconfigure")
            self._load_model(new_model_uri)
            return

        # Check if URI changed
        if self.model_info.model_uri != new_model_uri:
            logger.info(
                f"🔄 Updating model from {self.model_info.model_uri} to {new_model_uri}"
            )
            self._load_model(new_model_uri)
        else:
            logger.info("ℹ️ Model URI unchanged, skipping reload")

    @app.get(
        "/",
        response_model=RootResponse,
        summary="Root endpoint",
        responses={
            200: {"description": "Service information"},
            503: {"description": "Service not healthy"},
        },
    )
    async def root(self):
        """Root endpoint with basic info."""
        return RootResponse(
            service="Qwen2.5-VL Vision-Language API",
            version="1.0.0",
            status=self.status.value,
            docs="/docs",
            health="/health",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        responses={
            200: {"description": "Service is healthy"},
            503: {"description": "Service is not ready or unhealthy"},
        },
    )
    async def health(self):
        """Health check endpoint."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        response = HealthResponse(
            status=self.status,
            model_loaded=self.model is not None,
            model_uri=self.model_info.model_uri if self.model_info else None,
            uptime_seconds=int(uptime),
        )

        # Return 503 if not healthy
        if self.status != APIStatus.HEALTHY:
            detail = response.model_dump()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            )

        return response

    @app.get(
        "/info",
        response_model=ModelInfo,
        summary="Model Information",
        responses={
            200: {"description": "Model information"},
            503: {"description": "Model not loaded", "model": ErrorResponse},
        },
    )
    async def info(self):
        """Get detailed model information."""
        if self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please configure the deployment with a model_uri.",
            )
        return self.model_info

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Analyze Medical Images",
        responses={
            200: {"description": "Successful prediction"},
            400: {"description": "Invalid input", "model": ErrorResponse},
            503: {"description": "Model not loaded", "model": ErrorResponse},
            500: {"description": "Internal server error", "model": ErrorResponse},
        },
    )
    async def predict(
        self,
        files: List[UploadFile] = File(
            ..., description="One or more image files to analyze"
        ),
    ):
        """
        Analyze one or more medical images.

        Returns one prediction per image.

        **Single image:**
        ```bash
        curl -X POST "http://localhost:8000/predict" \
        -F "files=@image.jpg"
        ```

        **Multiple images:**
        ```bash
        curl -X POST "http://localhost:8000/predict" \
        -F "files=@image1.jpg" \
        -F "files=@image2.jpg"
        ```
        """
        if self.model is None or self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        start_time = datetime.now(timezone.utc)

        try:
            # Convert uploaded files to base64 and track filenames
            base64_images = []
            filenames = []

            for file in files:
                # Validate it's an image
                if not file.content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File {file.filename} must be an image",
                    )

                # Read file and convert to base64
                contents = await file.read()
                img_base64 = base64.b64encode(contents).decode("utf-8")

                base64_images.append(img_base64)
                filenames.append(file.filename)

            logger.info(f"🔮 Running inference on {len(base64_images)} image(s)")

            # Call model predict with base64 images
            predictions = self.model.predict(
                base64_images, params=self.inference_params.model_dump()
            )

            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            logger.info(f"✅ Inference complete in {processing_time:.0f}ms")

            # Build response with one prediction per image
            image_predictions = [
                ImagePrediction(
                    image_index=idx,
                    filename=filename,
                    text=prediction,
                )
                for idx, (filename, prediction) in enumerate(
                    zip(filenames, predictions)
                )
            ]

            return PredictionResponse(
                predictions=image_predictions,
                num_images=len(base64_images),
                model_uri=self.model_info.model_uri,
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_time,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )


class AppBuilderArgs(BaseModel):
    """Arguments for building the Ray Serve application."""

    model_uri: str | None = Field(
        None,
        description="MLflow model URI to load (e.g., models:/qwen-vl-model/1 or runs:/run_id/model)",
    )


def app_builder(args: AppBuilderArgs) -> Application:
    """Helper function to build the deployment with optional model URI.

    Examples:
        Basic usage:
        >>> serve run src.serving.serve_qwen:app_builder model_uri="models:/qwen-vl-model/1"

        With hot reload for development:
        >>> serve run src.serving.serve_qwen:app_builder model_uri="runs:/abc123/model" --reload

    Args:
        args: Configuration arguments including model URI

    Returns:
        Ray Serve Application ready to deploy
    """
    return QwenRadiologyVQA.bind(model_uri=args.model_uri)
