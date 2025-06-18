"""
TRL Remote FSDP Provider Server
==============================

This module implements the HTTP server for the remote FSDP provider.
The server runs independently of Llama Stack and provides distributed
DPO training capabilities via HTTP API endpoints.

The server architecture:
1. FastAPI HTTP server with async endpoints
2. Provider instance that handles distributed training
3. torchrun subprocess management for distributed coordination
4. Real-time job monitoring and status updates

Usage:
    python -m llama_stack_provider_trl_remote.server --config config.yaml
    
The server will start and listen for HTTP requests from Llama Stack.
"""

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)

from .config import TrlRemoteFSDPConfig
from .post_training import TrlRemoteFSDPImpl

logger = logging.getLogger(__name__)


# Request/Response models for the HTTP API
class PreferenceOptimizeRequest(BaseModel):
    """Request model for starting DPO training job."""
    job_uuid: str
    model: str
    finetuned_model: str
    algorithm_config: DPOAlignmentConfig
    training_config: TrainingConfig
    hyperparam_search_config: dict[str, Any] = {}
    logger_config: dict[str, Any] = {}
    checkpoint_dir: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    provider_type: str
    distributed_backend: str
    world_size: int


# Global provider instance
provider_instance: TrlRemoteFSDPImpl | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for provider initialization and cleanup."""
    global provider_instance
    
    logger.info("Starting remote FSDP provider server")
    
    # Initialize provider (this would normally come from config file)
    config = TrlRemoteFSDPConfig()
    
    # For now, create mock API dependencies
    # In production, these would be actual connections to Llama Stack APIs
    class MockDatasetIO:
        async def iterrows(self, dataset_id: str, limit: int = -1):
            # This is a mock - in production, this would communicate with Llama Stack
            return type('obj', (object,), {'data': []})()
    
    class MockDatasets:
        pass
    
    provider_instance = TrlRemoteFSDPImpl(
        config=config,
        datasetio_api=MockDatasetIO(),
        datasets_api=MockDatasets(),
    )
    
    logger.info("Remote FSDP provider initialized")
    
    yield
    
    # Cleanup
    if provider_instance:
        logger.info("Shutting down remote FSDP provider")
        await provider_instance.shutdown()
    
    logger.info("Remote FSDP provider server stopped")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="TRL Remote FSDP Provider",
    description="Remote provider for distributed FSDP DPO training",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not provider_instance:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    
    return HealthResponse(
        status="healthy",
        provider_type="remote::trl_fsdp",
        distributed_backend="nccl",
        world_size=provider_instance.config.torch_run_config.nproc_per_node,
    )


@app.post("/v1/post-training/preference-optimize", response_model=PostTrainingJob)
async def preference_optimize(request: PreferenceOptimizeRequest):
    """Start distributed FSDP DPO training job."""
    if not provider_instance:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    
    try:
        job = await provider_instance.preference_optimize(
            job_uuid=request.job_uuid,
            model=request.model,
            finetuned_model=request.finetuned_model,
            algorithm_config=request.algorithm_config,
            training_config=request.training_config,
            hyperparam_search_config=request.hyperparam_search_config,
            logger_config=request.logger_config,
            checkpoint_dir=request.checkpoint_dir,
        )
        return job
        
    except Exception as e:
        logger.error(f"Error starting DPO training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/post-training/jobs", response_model=ListPostTrainingJobsResponse)
async def get_training_jobs():
    """Get list of all training jobs."""
    if not provider_instance:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    
    return await provider_instance.get_training_jobs()


@app.get("/v1/post-training/job/status", response_model=PostTrainingJobStatusResponse)
async def get_training_job_status(job_uuid: str):
    """Get status of specific training job."""
    if not provider_instance:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    
    try:
        status = await provider_instance.get_training_job_status(job_uuid)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
        return status
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/post-training/job/cancel")
async def cancel_training_job(job_uuid: str):
    """Cancel training job."""
    if not provider_instance:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    
    try:
        await provider_instance.cancel_training_job(job_uuid)
        return {"message": f"Job {job_uuid} cancellation requested"}
        
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/post-training/job/artifacts", response_model=PostTrainingJobArtifactsResponse)
async def get_training_job_artifacts(job_uuid: str):
    """Get artifacts from training job."""
    if not provider_instance:
        raise HTTPException(status_code=503, detail="Provider not initialized")
    
    try:
        artifacts = await provider_instance.get_training_job_artifacts(job_uuid)
        if artifacts is None:
            raise HTTPException(status_code=404, detail=f"Job {job_uuid} not found")
        return artifacts
        
    except Exception as e:
        logger.error(f"Error getting job artifacts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/tmp/trl_remote_fsdp_provider.log')
        ]
    )


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main function to start the remote FSDP provider server."""
    parser = argparse.ArgumentParser(description="TRL Remote FSDP Provider Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8322, help="Port to bind to")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("Starting TRL Remote FSDP Provider Server")
    logger.info(f"Server will listen on {args.host}:{args.port}")
    
    # Run the server
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 