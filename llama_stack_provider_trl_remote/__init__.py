"""
TRL Remote FSDP Provider Entry Point
===================================

This file serves as the entry point for the remote TRL provider that supports
FSDP (Fully Sharded Data Parallel) training. Unlike the inline provider, this
runs as a separate service and communicates with Llama Stack via HTTP.

Key Differences from Inline Provider:
- Runs as external service (separate process/container)
- Supports multi-GPU FSDP training using torch.distributed
- Uses torch run for launching distributed training
- Handles larger models that don't fit on single GPU

Architecture:
1. Remote server runs independently of Llama Stack
2. Llama Stack makes HTTP calls to the remote provider
3. Remote provider launches distributed training via torch run
4. Training results are communicated back via API
"""

from typing import Any
from llama_stack.distribution.datatypes import Api
from .config import TrlRemoteFSDPConfig


async def get_provider_impl(
    config: TrlRemoteFSDPConfig,
    deps: dict[Api, Any],
):
    """
    Create and configure a remote TRL FSDP provider instance.
    
    This creates a remote provider that can handle FSDP distributed training
    across multiple GPUs using torch.distributed.
    
    Args:
        config: TrlRemoteFSDPConfig containing FSDP training settings:
                - world_size: Number of GPUs to use
                - fsdp_config: FSDP sharding strategy and settings
                - torch_run_config: Configuration for torch run launcher
                - Training hyperparameters optimized for distributed training
        
        deps: Dictionary of API dependencies:
              - Api.datasetio: For loading training datasets
              - Api.datasets: For dataset operations
              
    Returns:
        TrlRemoteFSDPImpl: Remote provider instance that coordinates
                          FSDP training across multiple devices
    """
    from .post_training import TrlRemoteFSDPImpl

    impl = TrlRemoteFSDPImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    
    return impl 