"""
TRL Remote FSDP Provider Configuration
=====================================

Configuration for the remote TRL provider that supports FSDP (Fully Sharded Data Parallel)
distributed training. This extends the base TRL configuration with FSDP-specific settings,
torch.distributed configuration, and multi-GPU coordination parameters.

Key Configuration Areas:
- FSDP sharding strategies and memory optimization
- torch.distributed and torch run configuration  
- Multi-GPU training coordination
- Remote service communication settings
- Resource allocation across distributed nodes
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class FSDPConfig(BaseModel):
    """
    FSDP (Fully Sharded Data Parallel) specific configuration.
    
    FSDP allows training very large models by sharding model parameters,
    gradients, and optimizer states across multiple GPUs.
    """
    
    # FSDP sharding strategy
    # - FULL_SHARD: Shard parameters, gradients, and optimizer states (most memory efficient)
    # - SHARD_GRAD_OP: Shard gradients and optimizer states only
    # - NO_SHARD: Keep everything local (equivalent to DDP)
    # - HYBRID_SHARD: Hybrid approach for multi-node setups
    sharding_strategy: Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"] = "FULL_SHARD"
    
    # CPU offloading for parameters and gradients to save GPU memory
    # Warning: CPU offload reduces training speed but enables larger models
    cpu_offload: bool = False
    
    # Mixed precision policy for FSDP
    # - "bf16": Use bfloat16 (recommended for modern GPUs)
    # - "fp16": Use float16 (more compatible but less stable)
    # - None: Use full precision (fp32)
    mixed_precision_policy: Literal["bf16", "fp16"] | None = "bf16"
    
    # Limit of non-FSDP parameters (parameters not wrapped by FSDP)
    # Smaller values = more aggressive sharding = more memory savings
    limit_all_gathers: bool = True
    
    # Forward prefetch for improved performance
    # Prefetches next layer's parameters during forward pass
    forward_prefetch: bool = True
    
    # Backward prefetch for improved performance
    # Prefetches parameters needed for backward pass
    backward_prefetch: Literal["BACKWARD_PRE", "BACKWARD_POST"] = "BACKWARD_PRE"


class TorchRunConfig(BaseModel):
    """
    Configuration for torch.distributed.run (torchrun) launcher.
    
    torchrun is PyTorch's recommended way to launch distributed training jobs.
    It handles process spawning, environment setup, and fault tolerance.
    """
    
    # Number of nodes (machines) in the distributed training job
    # For single-machine multi-GPU training, this should be 1
    nnodes: int = 1
    
    # Number of processes per node (typically equal to number of GPUs per node)
    nproc_per_node: int = 2
    
    # Node rank (which node this is in multi-node setup)
    # For single-node training, this should be 0
    node_rank: int = 0
    
    # Master address for distributed coordination
    # For single-node training, use "localhost"
    # For multi-node training, use the IP of the master node
    master_addr: str = "localhost"
    
    # Master port for distributed coordination
    # Make sure this port is available and not blocked by firewall
    master_port: int = 29500
    
    # Maximum number of worker process restarts
    # Helps with fault tolerance in distributed training
    max_restarts: int = 0
    
    # Timeout for process initialization (in seconds)
    rdzv_timeout: int = 1800  # 30 minutes


class RemoteServiceConfig(BaseModel):
    """
    Configuration for the remote service communication.
    
    The remote provider runs as a separate service that Llama Stack
    communicates with via HTTP.
    """
    
    # Host address where the remote provider service runs
    host: str = "localhost"
    
    # Port for the remote provider service
    port: int = 8322  # Different from main Llama Stack port (8321)
    
    # Request timeout for communication with remote service (seconds)
    request_timeout: int = 3600  # 1 hour for long training jobs
    
    # Health check interval (seconds)
    health_check_interval: int = 30
    
    # Maximum number of concurrent training jobs
    max_concurrent_jobs: int = 1


class TrlRemoteFSDPConfig(BaseModel):
    """
    Complete configuration for TRL Remote FSDP Provider.
    
    This configuration combines FSDP distributed training settings with
    remote service configuration and DPO training parameters.
    """
    
    # === DISTRIBUTED TRAINING CONFIGURATION ===
    
    # FSDP-specific settings
    fsdp_config: FSDPConfig = Field(default_factory=FSDPConfig)
    
    # torch.distributed.run configuration
    torch_run_config: TorchRunConfig = Field(default_factory=TorchRunConfig)
    
    # Remote service configuration
    service_config: RemoteServiceConfig = Field(default_factory=RemoteServiceConfig)
    
    # === HARDWARE AND DEVICE SETTINGS ===
    
    # Device type for distributed training
    # FSDP requires CUDA-capable GPUs
    device: Literal["cuda"] = "cuda"
    
    # CUDA device IDs to use for training
    # Empty list means use all available GPUs
    # Example: [0, 1, 2, 3] to use first 4 GPUs
    cuda_devices: list[int] = Field(default_factory=list)
    
    # === MODEL AND CHECKPOINT SETTINGS ===
    
    # Checkpoint format optimized for FSDP
    # - "fsdp": FSDP-optimized checkpoint format (recommended)
    # - "huggingface": Standard HuggingFace format (compatibility)
    checkpoint_format: Literal["fsdp", "huggingface"] = "fsdp"
    
    # Whether to use FSDP state dict for checkpointing
    # FSDP state dict is more memory efficient for large models
    use_fsdp_state_dict: bool = True
    
    # === TRAINING CONFIGURATION ===
    
    # Maximum sequence length for FSDP training
    # Can be longer than single-GPU training due to memory sharding
    max_seq_length: int = 4096  # Doubled from single-GPU default
    
    # Gradient checkpointing for memory efficiency
    # Highly recommended for large model FSDP training
    gradient_checkpointing: bool = True
    
    # === DPO-SPECIFIC CONFIGURATION ===
    
    # DPO beta parameter (same as inline provider)
    dpo_beta: float = 0.1
    
    # Whether to use reference model for DPO
    # With FSDP, reference model also benefits from sharding
    use_reference_model: bool = True
    
    # DPO loss type
    dpo_loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    
    # === PERFORMANCE OPTIMIZATION ===
    
    # Activation checkpointing strategy for FSDP
    # - "selective": Only checkpoint expensive operations
    # - "full": Checkpoint all activations (max memory savings)
    activation_checkpointing: Literal["selective", "full", "none"] = "selective"
    
    # Compile model for better performance (PyTorch 2.0+)
    # Can significantly speed up training with large models
    compile_model: bool = False  # Set to True if using PyTorch 2.0+
    
    # Data loading optimization for distributed training
    dataloader_num_workers: int = 4  # More workers for distributed
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    
    # === LOGGING AND MONITORING ===
    
    # Logging configuration for distributed training
    # Only rank 0 should log to avoid duplicate logs
    log_on_each_node: bool = False
    logging_steps: int = 10
    
    # Save steps for distributed training
    # Should be coordinated across all processes
    save_steps: int = 100
    save_total_limit: int = 3
    
    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        """
        Provide a sample configuration for FSDP distributed training.
        
        This returns a configuration suitable for 2-GPU FSDP training
        on a single node, which is a common setup for development and
        small-scale distributed training.
        
        Args:
            __distro_dir__: Directory for the distribution
            **kwargs: Additional configuration overrides
            
        Returns:
            Dictionary containing sample FSDP configuration
        """
        return {
            "device": "cuda",
            "fsdp_config": {
                "sharding_strategy": "FULL_SHARD",
                "mixed_precision_policy": "bf16",
                "cpu_offload": False,
            },
            "torch_run_config": {
                "nnodes": 1,
                "nproc_per_node": 2,  # 2 GPUs
                "master_addr": "localhost", 
                "master_port": 29500,
            },
            "service_config": {
                "host": "localhost",
                "port": 8322,
                "max_concurrent_jobs": 1,
            },
            "checkpoint_format": "fsdp",
            "use_fsdp_state_dict": True,
            "max_seq_length": 4096,
            "gradient_checkpointing": True,
            "dpo_beta": 0.1,
            "use_reference_model": True,
        } 