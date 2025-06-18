"""
TRL Remote FSDP Post-Training Provider Implementation
===================================================

This file contains the main implementation of the remote TRL provider for FSDP
(Fully Sharded Data Parallel) distributed training. Unlike the inline provider,
this runs as a separate service and communicates with Llama Stack via HTTP.

Key Differences from Inline Provider:
- Runs as external service (separate process/container)
- Uses torch.distributed.run (torchrun) to launch distributed training
- Supports multi-GPU FSDP training for large models
- Handles distributed coordination and resource management
- Remote service communication with status updates

Architecture:
1. Remote service runs independently with its own job scheduler
2. Receives training requests via HTTP API from Llama Stack
3. Launches distributed training using torchrun subprocess
4. Monitors distributed training progress and resources
5. Provides real-time status updates and artifact management
"""

import asyncio
import logging
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    Checkpoint,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)

from llama_stack.providers.utils.scheduler import JobArtifact, Scheduler
from llama_stack.providers.utils.scheduler import JobStatus as SchedulerJobStatus
from llama_stack.schema_utils import webmethod

from .config import TrlRemoteFSDPConfig
from .recipes.dpo_training_fsdp import DPOTrainingFSDP

logger = logging.getLogger(__name__)


class DistributedTrainingArtifactType(Enum):
    """Types of artifacts produced by distributed FSDP training."""
    CHECKPOINT = "checkpoint"
    DISTRIBUTED_RESOURCES = "distributed_resources"
    TORCH_RUN_LOG = "torch_run_log"


# Job type identifier for FSDP DPO training
_JOB_TYPE_FSDP_DPO_TRAINING = "fsdp-dpo-training"


class TrlRemoteFSDPImpl:
    """
    Remote FSDP provider implementation for distributed DPO training.
    
    This class implements Llama Stack's PostTraining protocol as a remote service
    that can handle FSDP distributed training across multiple GPUs using torchrun.
    
    Key Responsibilities:
    1. Handle training job requests from Llama Stack via HTTP
    2. Launch distributed training using torch.distributed.run (torchrun)
    3. Monitor distributed training progress across multiple processes
    4. Coordinate resource allocation and cleanup
    5. Provide real-time status updates and artifact management
    6. Handle FSDP-specific checkpointing and model saving
    
    Remote Service Architecture:
    - Runs as independent service separate from Llama Stack
    - Uses async job scheduling for non-blocking training
    - Launches torchrun subprocess for distributed coordination
    - Provides HTTP endpoints for job management and monitoring
    """
    
    def __init__(
        self,
        config: TrlRemoteFSDPConfig,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        """
        Initialize the remote FSDP provider.
        
        Args:
            config: TrlRemoteFSDPConfig containing FSDP distributed training settings
                   including torch.distributed configuration, FSDP parameters,
                   and remote service settings
            datasetio_api: DatasetIO API for loading training datasets
            datasets_api: Datasets API for dataset operations
        """
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api
        
        # Create scheduler for managing distributed training jobs
        self._scheduler = Scheduler()
        
        # Track active torchrun processes
        self._active_processes = {}

    async def shutdown(self) -> None:
        """
        Clean shutdown of the remote provider.
        
        This method ensures all distributed training processes are properly
        terminated and resources are cleaned up.
        """
        # Terminate any active torchrun processes
        for job_uuid, process in self._active_processes.items():
            if process.returncode is None:  # Process still running
                logger.info(f"Terminating distributed training process for job {job_uuid}")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Force killing distributed training process for job {job_uuid}")
                    process.kill()
        
        # Shutdown scheduler
        await self._scheduler.shutdown()

    def _create_torchrun_command(
        self,
        job_uuid: str,
        script_path: str,
        model: str,
        output_dir: str | None,
        dpo_config: DPOAlignmentConfig,
        config: TrainingConfig,
    ) -> list[str]:
        """
        Create torchrun command for launching distributed FSDP training.
        
        This method builds the torchrun command that will launch the distributed
        training across multiple GPUs/nodes.
        
        Args:
            job_uuid: Unique identifier for the training job
            script_path: Path to the training script to execute
            model: Base model identifier
            output_dir: Directory for saving checkpoints
            dpo_config: DPO algorithm configuration
            config: General training configuration
            
        Returns:
            List of command arguments for torchrun
        """
        torch_run_config = self.config.torch_run_config
        
        # Build base torchrun command
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(torch_run_config.nproc_per_node),
            "--nnodes", str(torch_run_config.nnodes),
            "--node_rank", str(torch_run_config.node_rank),
            "--master_addr", torch_run_config.master_addr,
            "--master_port", str(torch_run_config.master_port),
            "--max_restarts", str(torch_run_config.max_restarts),
            "--rdzv_timeout", str(torch_run_config.rdzv_timeout),
        ]
        
        # Add the training script
        cmd.append(script_path)
        
        # Add training arguments
        cmd.extend([
            "--job_uuid", job_uuid,
            "--model", model,
            "--n_epochs", str(config.n_epochs),
            "--dpo_beta", str(self.config.dpo_beta),
        ])
        
        if output_dir:
            cmd.extend(["--output_dir", output_dir])
        
        if config.data_config:
            cmd.extend([
                "--dataset_id", config.data_config.dataset_id,
                "--batch_size", str(config.data_config.batch_size),
            ])
        
        # Add FSDP-specific arguments
        fsdp_config = self.config.fsdp_config
        cmd.extend([
            "--fsdp_sharding_strategy", fsdp_config.sharding_strategy,
            "--fsdp_mixed_precision", fsdp_config.mixed_precision_policy or "none",
            "--fsdp_cpu_offload", str(fsdp_config.cpu_offload).lower(),
            "--max_seq_length", str(self.config.max_seq_length),
        ])
        
        return cmd

    @staticmethod
    def _checkpoint_to_artifact(checkpoint: Checkpoint) -> JobArtifact:
        """Convert a Checkpoint to JobArtifact for distributed training."""
        return JobArtifact(
            type=DistributedTrainingArtifactType.CHECKPOINT.value,
            name=checkpoint.identifier,
            uri=checkpoint.path,
            metadata=dict(checkpoint),
        )

    @staticmethod
    def _distributed_resources_to_artifact(resources: dict[str, Any]) -> JobArtifact:
        """Convert distributed resource statistics to JobArtifact."""
        return JobArtifact(
            type=DistributedTrainingArtifactType.DISTRIBUTED_RESOURCES.value,
            name=DistributedTrainingArtifactType.DISTRIBUTED_RESOURCES.value,
            metadata=resources,
        )

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        """
        Supervised Fine-Tuning - NOT IMPLEMENTED in remote FSDP provider.
        
        The remote FSDP provider specializes in DPO training only.
        For supervised fine-tuning with FSDP, this method would need to be
        implemented with appropriate distributed training logic.
        
        This method exists to satisfy Llama Stack's PostTraining protocol requirements.
        """
        raise NotImplementedError(
            "Supervised fine-tuning is not implemented in the remote FSDP provider. "
            "This provider specializes in distributed DPO training only. "
            "Use preference_optimize() for FSDP DPO training."
        )

    async def preference_optimize(
        self,
        job_uuid: str,
        model: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        checkpoint_dir: str | None = None,
    ) -> PostTrainingJob:
        """
        Start distributed FSDP DPO training job using torchrun.
        
        This method launches distributed DPO training across multiple GPUs
        using torch.distributed.run for coordination. The training runs in
        separate processes managed by torchrun.
        
        Args:
            job_uuid: Unique identifier for the training job
            model: Base model to train (e.g., "meta-llama/Llama-2-7b-hf")
            finetuned_model: Name for the output model
            algorithm_config: DPO-specific configuration
            training_config: General training configuration with dataset info
            hyperparam_search_config: Hyperparameter search settings (not used yet)
            logger_config: Logging configuration
            checkpoint_dir: Directory for saving distributed checkpoints
            
        Returns:
            PostTrainingJob: Job object for tracking distributed training progress
        """
        
        async def handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb):
            """
            Async job handler for distributed FSDP DPO training.
            
            This handler launches torchrun subprocess and monitors the distributed
            training progress, collecting artifacts and status updates.
            """
            on_log_message_cb("Starting distributed FSDP DPO training with torchrun")
            
            try:
                # Create training script (we'll use the recipe directly)
                # In production, you might have a separate script file
                training_script = Path(__file__).parent / "recipes" / "dpo_training_fsdp.py"
                
                # Create torchrun command
                cmd = self._create_torchrun_command(
                    job_uuid,
                    str(training_script),
                    model,
                    checkpoint_dir,
                    algorithm_config,
                    training_config,
                )
                
                on_log_message_cb(f"Launching torchrun with command: {' '.join(cmd)}")
                
                # Launch distributed training process
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
                )
                
                # Store process for cleanup
                self._active_processes[job_uuid] = process
                
                on_status_change_cb(SchedulerJobStatus.running)
                on_log_message_cb("Distributed training process started")
                
                # Monitor training process
                output_lines = []
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    
                    line_str = line.decode().strip()
                    output_lines.append(line_str)
                    
                    # Log important messages
                    if any(keyword in line_str.lower() for keyword in 
                           ["epoch", "loss", "error", "completed", "saved"]):
                        on_log_message_cb(f"Training: {line_str}")
                
                # Wait for process completion
                return_code = await process.wait()
                
                # Remove from active processes
                if job_uuid in self._active_processes:
                    del self._active_processes[job_uuid]
                
                if return_code == 0:
                    on_log_message_cb("Distributed FSDP training completed successfully")
                    
                    # Collect artifacts if training completed successfully
                    if checkpoint_dir:
                        checkpoint_path = Path(checkpoint_dir) / "fsdp_dpo_model"
                        if checkpoint_path.exists():
                            checkpoint = Checkpoint(
                                identifier=f"{model}-fsdp-dpo-{training_config.n_epochs}",
                                created_at=datetime.now(timezone.utc),
                                epoch=training_config.n_epochs,
                                post_training_job_id=job_uuid,
                                path=str(checkpoint_path),
                            )
                            
                            artifact = self._checkpoint_to_artifact(checkpoint)
                            on_artifact_collected_cb(artifact)
                    
                    # Report distributed resource usage
                    resources = {
                        "world_size": self.config.torch_run_config.nproc_per_node,
                        "distributed_backend": "nccl",
                        "fsdp_strategy": self.config.fsdp_config.sharding_strategy,
                        "training_completed": True,
                    }
                    
                    resource_artifact = self._distributed_resources_to_artifact(resources)
                    on_artifact_collected_cb(resource_artifact)
                    
                    on_status_change_cb(SchedulerJobStatus.completed)
                else:
                    on_log_message_cb(f"Distributed training failed with return code: {return_code}")
                    
                    # Save torch run logs as artifact for debugging
                    log_artifact = JobArtifact(
                        type=DistributedTrainingArtifactType.TORCH_RUN_LOG.value,
                        name="torchrun_output",
                        metadata={"output_lines": output_lines, "return_code": return_code},
                    )
                    on_artifact_collected_cb(log_artifact)
                    
                    on_status_change_cb(SchedulerJobStatus.failed)
                    
            except Exception as e:
                on_log_message_cb(f"Error in distributed training handler: {str(e)}")
                on_status_change_cb(SchedulerJobStatus.failed)
                
                # Clean up process if still running
                if job_uuid in self._active_processes:
                    process = self._active_processes[job_uuid]
                    if process.returncode is None:
                        process.terminate()
                    del self._active_processes[job_uuid]

        # Schedule the distributed training job
        job_uuid = self._scheduler.schedule(_JOB_TYPE_FSDP_DPO_TRAINING, job_uuid, handler)
        
        return PostTrainingJob(job_uuid=job_uuid)

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """Get list of all distributed training jobs."""
        return ListPostTrainingJobsResponse(
            data=[PostTrainingJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )

    @staticmethod
    def _get_artifacts_metadata_by_type(job, artifact_type):
        """Extract metadata for artifacts of specific type."""
        return [artifact.metadata for artifact in job.artifacts if artifact.type == artifact_type]

    @classmethod
    def _get_checkpoints(cls, job):
        """Get checkpoint artifacts from distributed training job."""
        return cls._get_artifacts_metadata_by_type(job, DistributedTrainingArtifactType.CHECKPOINT.value)

    @classmethod
    def _get_distributed_resources(cls, job):
        """Get distributed resource statistics from training job."""
        data = cls._get_artifacts_metadata_by_type(job, DistributedTrainingArtifactType.DISTRIBUTED_RESOURCES.value)
        return data[0] if data else None

    @webmethod(route="/post-training/job/status")
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        """
        Get status of distributed training job.
        
        Provides real-time information about FSDP distributed training progress,
        including distributed resource usage and checkpoint information.
        """
        job = self._scheduler.get_job(job_uuid)

        # Convert scheduler status to API status
        match job.status:
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                raise NotImplementedError(f"Unknown job status: {job.status}")

        return PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=status,
            scheduled_at=job.scheduled_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            checkpoints=self._get_checkpoints(job),
            resources_allocated=self._get_distributed_resources(job),
        )

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None:
        """
        Cancel distributed training job.
        
        This method cancels the torchrun process and all distributed
        training processes associated with the job.
        """
        # Cancel the job in scheduler
        self._scheduler.cancel(job_uuid)
        
        # Terminate torchrun process if running
        if job_uuid in self._active_processes:
            process = self._active_processes[job_uuid]
            if process.returncode is None:
                logger.info(f"Terminating distributed training process for job {job_uuid}")
                process.terminate()
                
                # Give process time to shut down gracefully
                try:
                    await asyncio.wait_for(process.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Force killing distributed training process for job {job_uuid}")
                    process.kill()
            
            del self._active_processes[job_uuid]

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        """
        Get artifacts from distributed training job.
        
        Returns checkpoints and other artifacts produced by FSDP distributed training.
        """
        job = self._scheduler.get_job(job_uuid)
        
        return PostTrainingJobArtifactsResponse(
            job_uuid=job_uuid,
            checkpoints=self._get_checkpoints(job)
        ) 