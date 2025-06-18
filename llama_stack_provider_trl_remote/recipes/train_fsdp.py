#!/usr/bin/env python3
"""
Standalone FSDP DPO Training Script
==================================

This script can be launched directly with torchrun for distributed FSDP DPO training.
It's designed to be called by the remote provider but can also be used independently.

Usage:
    torchrun --nproc_per_node=2 train_fsdp.py --model meta-llama/Llama-2-7b-hf --dataset_id my-dataset

The script handles:
- Distributed process initialization
- FSDP model setup and wrapping
- DPO training with preference data
- Distributed checkpointing
- Resource monitoring and cleanup
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_stack_provider_trl_remote.recipes.dpo_training_fsdp import DPOTrainingFSDP, setup_distributed
from llama_stack_provider_trl_remote.config import TrlRemoteFSDPConfig, FSDPConfig, TorchRunConfig
from llama_stack.apis.post_training import DPOAlignmentConfig, TrainingConfig, DataConfig


def setup_logging():
    """Setup logging for distributed training."""
    # Only log on rank 0 to avoid duplicate logs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/tmp/fsdp_training.log')
            ]
        )
        logging.getLogger().rank = rank
    else:
        logging.basicConfig(level=logging.WARNING)


def parse_args():
    """Parse command line arguments for FSDP training."""
    parser = argparse.ArgumentParser(description="FSDP DPO Training Script")
    
    # Required arguments
    parser.add_argument("--job_uuid", required=True, help="Unique job identifier")
    parser.add_argument("--model", required=True, help="Base model to train")
    parser.add_argument("--dataset_id", required=True, help="Dataset ID for training")
    
    # Training configuration
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    
    # DPO configuration
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--use_reference_model", action="store_true", default=True, help="Use reference model")
    
    # FSDP configuration
    parser.add_argument("--fsdp_sharding_strategy", default="FULL_SHARD", 
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
                        help="FSDP sharding strategy")
    parser.add_argument("--fsdp_mixed_precision", default="bf16", 
                        choices=["bf16", "fp16", "none"],
                        help="FSDP mixed precision policy")
    parser.add_argument("--fsdp_cpu_offload", action="store_true", default=False,
                        help="Enable FSDP CPU offloading")
    
    # Output configuration
    parser.add_argument("--output_dir", help="Output directory for checkpoints")
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create provider configuration from command line arguments."""
    # FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy=args.fsdp_sharding_strategy,
        mixed_precision_policy=args.fsdp_mixed_precision if args.fsdp_mixed_precision != "none" else None,
        cpu_offload=args.fsdp_cpu_offload,
    )
    
    # torch.distributed configuration (read from environment)
    torch_run_config = TorchRunConfig(
        nnodes=int(os.environ.get("WORLD_SIZE", 1)) // int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        nproc_per_node=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        node_rank=int(os.environ.get("GROUP_RANK", 0)),
        master_addr=os.environ.get("MASTER_ADDR", "localhost"),
        master_port=int(os.environ.get("MASTER_PORT", 29500)),
    )
    
    # Provider configuration
    provider_config = TrlRemoteFSDPConfig(
        fsdp_config=fsdp_config,
        torch_run_config=torch_run_config,
        max_seq_length=args.max_seq_length,
        dpo_beta=args.dpo_beta,
        use_reference_model=args.use_reference_model,
        gradient_checkpointing=True,  # Always use for FSDP
    )
    
    return provider_config


def create_training_configs(args):
    """Create training and DPO configurations from arguments."""
    # Data configuration
    data_config = DataConfig(
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        shuffle=True,
        data_format="instruct",
        train_split_percentage=0.9,
    )
    
    # Training configuration
    training_config = TrainingConfig(
        n_epochs=args.n_epochs,
        max_steps_per_epoch=0,  # No limit
        learning_rate=args.learning_rate,
        warmup_steps=0,
        lr_scheduler_type="constant",
        gradient_accumulation_steps=1,
        data_config=data_config,
    )
    
    # DPO algorithm configuration
    dpo_config = DPOAlignmentConfig(
        type="dpo",
        reward_scale=1.0,
        reward_clip=5.0,
        epsilon=0.1,
        gamma=0.99,
    )
    
    return training_config, dpo_config


class MockDatasetIO:
    """Mock DatasetIO for standalone script (in production, this would connect to Llama Stack)."""
    
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
    
    async def iterrows(self, dataset_id: str, limit: int = -1):
        """Return mock preference data for testing."""
        # In production, this would load real data from Llama Stack
        mock_data = [
            {
                "prompt": "What is FSDP?",
                "chosen": "FSDP (Fully Sharded Data Parallel) is a distributed training technique that shards model parameters, gradients, and optimizer states across multiple devices to enable training of large models.",
                "rejected": "FSDP is just another name for distributed training."
            },
            {
                "prompt": "How does distributed training work?",
                "chosen": "Distributed training splits the computation across multiple devices or nodes. In FSDP, the model parameters are sharded across devices, and during forward/backward passes, parameters are gathered as needed and then resharded.",
                "rejected": "Distributed training just uses multiple computers to train faster."
            },
            {
                "prompt": "What are the benefits of FSDP over DDP?",
                "chosen": "FSDP provides better memory efficiency than DDP by sharding model parameters instead of replicating them. This allows training of larger models that wouldn't fit in memory with DDP.",
                "rejected": "FSDP is just a newer version of DDP."
            }
        ]
        
        return type('obj', (object,), {'data': mock_data})()


class MockDatasets:
    """Mock Datasets API for standalone script."""
    pass


async def main():
    """Main training function."""
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Initialize distributed training
    setup_distributed()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if rank == 0:
        logging.info(f"Starting FSDP DPO training: {args.job_uuid}")
        logging.info(f"Model: {args.model}")
        logging.info(f"Dataset: {args.dataset_id}")
        logging.info(f"World size: {world_size}, Local rank: {local_rank}")
        logging.info(f"FSDP strategy: {args.fsdp_sharding_strategy}")
    
    try:
        # Create configurations
        provider_config = create_config_from_args(args)
        training_config, dpo_config = create_training_configs(args)
        
        # Create training recipe
        recipe = DPOTrainingFSDP(
            job_uuid=args.job_uuid,
            datasetio_api=MockDatasetIO(args.dataset_id),
            datasets_api=MockDatasets(),
        )
        
        # Run distributed training
        if rank == 0:
            logging.info("Starting FSDP distributed DPO training...")
        
        memory_stats, checkpoints = await recipe.train(
            model=args.model,
            output_dir=args.output_dir,
            job_uuid=args.job_uuid,
            dpo_config=dpo_config,
            config=training_config,
            provider_config=provider_config,
        )
        
        if rank == 0:
            logging.info("FSDP training completed successfully!")
            if checkpoints:
                logging.info(f"Saved checkpoints: {[cp.path for cp in checkpoints]}")
            logging.info(f"Final memory stats: {memory_stats}")
    
    except Exception as e:
        if rank == 0:
            logging.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if rank == 0:
            logging.info("Cleaning up distributed training...")


if __name__ == "__main__":
    # Run the training
    asyncio.run(main()) 