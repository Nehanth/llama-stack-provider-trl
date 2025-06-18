"""
DPO Training Recipe for FSDP (Fully Sharded Data Parallel)
=========================================================

This file implements DPO training using FSDP for distributed training across multiple GPUs.
FSDP enables training very large models by sharding model parameters, gradients, and optimizer
states across multiple devices, allowing models that don't fit on a single GPU.

Key FSDP Concepts:
- Parameter Sharding: Model parameters are distributed across devices
- Gradient Sharding: Gradients are computed and stored distributedly  
- All-Gather: Temporarily reconstruct full parameters for forward/backward
- Reduce-Scatter: Distribute gradient updates across devices
- CPU Offloading: Move parameters to CPU to save GPU memory

This implementation uses torch.distributed for coordination and TRL's DPOTrainer
with FSDP integration for the actual DPO training.
"""

import gc
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    LocalStateDictConfig,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from trl import DPOConfig, DPOTrainer

# Llama Stack API imports
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    Checkpoint,
    DataConfig,
    DPOAlignmentConfig,
    TrainingConfig,
)

from ..config import TrlRemoteFSDPConfig, FSDPConfig

logger = logging.getLogger(__name__)


def setup_distributed():
    """
    Initialize distributed training environment.
    
    This function sets up torch.distributed for FSDP training.
    It should be called at the beginning of each distributed process.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training is not available")
    
    if not dist.is_initialized():
        # Initialize distributed training
        # torch.distributed.init_process_group is typically called by torchrun
        dist.init_process_group(backend="nccl")
    
    # Set device for this process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Initialized distributed training - Rank: {dist.get_rank()}, "
                f"World Size: {dist.get_world_size()}, Local Rank: {local_rank}")


def get_fsdp_sharding_strategy(strategy_name: str) -> ShardingStrategy:
    """Convert string strategy name to FSDP ShardingStrategy enum."""
    strategy_mapping = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP, 
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    if strategy_name not in strategy_mapping:
        raise ValueError(f"Unknown FSDP sharding strategy: {strategy_name}")
    
    return strategy_mapping[strategy_name]


def get_fsdp_mixed_precision_policy(policy_name: str | None) -> MixedPrecision | None:
    """Create FSDP MixedPrecision policy from configuration string."""
    if policy_name is None:
        return None
    
    if policy_name == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif policy_name == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unknown mixed precision policy: {policy_name}")


def get_fsdp_cpu_offload(enabled: bool) -> CPUOffload | None:
    """Create FSDP CPU offload configuration."""
    if enabled:
        return CPUOffload(offload_params=True)
    return None


def get_auto_wrap_policy(model_type: str):
    """
    Get the appropriate auto-wrap policy for FSDP based on model type.
    
    The auto-wrap policy determines which layers/modules get wrapped
    with FSDP. This is crucial for memory efficiency and performance.
    """
    # Common transformer layer types for different models
    transformer_layer_cls = {
        "llama": LlamaDecoderLayer,
        "gpt2": GPT2Block,
        # Add more as needed for other model types
    }
    
    # Try to detect model type from model name if not specified
    if model_type.lower() in transformer_layer_cls:
        layer_cls = transformer_layer_cls[model_type.lower()]
    else:
        # Default to LlamaDecoderLayer for unknown models
        # This works for many modern transformer models
        layer_cls = LlamaDecoderLayer
        logger.warning(f"Unknown model type {model_type}, using LlamaDecoderLayer for auto-wrap")
    
    return transformer_auto_wrap_policy({layer_cls})


def get_distributed_memory_stats() -> dict[str, Any]:
    """Get memory statistics for distributed training monitoring."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    stats = {
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "local_rank": local_rank,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
    }
    
    # GPU memory stats
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        stats["gpu_memory"] = {
            "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        }
    
    # System memory stats (only on rank 0 to avoid duplicates)
    if dist.get_rank() == 0:
        vm = psutil.virtual_memory()
        stats["system_memory"] = {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent": vm.percent,
        }
    
    return stats


class DPOTrainingFSDP:
    """
    FSDP-based DPO training implementation for distributed multi-GPU training.
    
    This class implements DPO training using PyTorch's FSDP (Fully Sharded Data Parallel)
    for efficient distributed training of large language models.
    
    Key Features:
    - Multi-GPU distributed training using FSDP
    - Memory-efficient training through parameter sharding
    - Support for very large models that don't fit on single GPU
    - CPU offloading for even larger models
    - Distributed checkpointing and artifact management
    - Integration with torch.distributed for coordination
    
    FSDP Benefits over standard DDP:
    - Lower memory usage through parameter sharding
    - Ability to train larger models
    - Better scaling to many GPUs
    - CPU offloading capabilities
    """
    
    def __init__(
        self,
        job_uuid: str,
        datasetio_api: DatasetIO,
        datasets_api: Datasets,
    ) -> None:
        """
        Initialize the FSDP DPO training recipe.
        
        Args:
            job_uuid: Unique identifier for the training job
            datasetio_api: DatasetIO API for loading datasets
            datasets_api: Datasets API for dataset operations
        """
        self.job_uuid = job_uuid
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets_api

    def validate_preference_dataset(self, rows: list[dict]) -> bool:
        """
        Validate that the dataset has the required fields for DPO training.
        Same validation as single-device version.
        """
        required_fields = ["prompt", "chosen", "rejected"]
        
        if not rows:
            logger.warning("Dataset is empty")
            return False
        
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                logger.warning(f"Row {i} is not a dictionary")
                return False
                
            for field in required_fields:
                if field not in row:
                    logger.warning(f"Row {i} missing required DPO field: {field}")
                    return False
                    
                if not isinstance(row[field], str):
                    logger.warning(f"Row {i} field '{field}' is not a string")
                    return False
                    
                if not row[field].strip():
                    logger.warning(f"Row {i} field '{field}' is empty")
                    return False
        
        # Only log on rank 0 to avoid duplicate logs
        if dist.get_rank() == 0:
            logger.info(f"DPO dataset validation passed: {len(rows)} preference examples")
        return True

    def create_dpo_dataset(
        self, rows: list[dict], config: TrainingConfig, provider_config: TrlRemoteFSDPConfig
    ) -> Dataset:
        """Create HuggingFace Dataset from preference data for FSDP DPO training."""
        dpo_examples = []
        for row in rows:
            if all(field in row for field in ["prompt", "chosen", "rejected"]):
                dpo_examples.append({
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })

        if not dpo_examples:
            raise ValueError("No valid preference examples found in dataset")

        if dist.get_rank() == 0:
            logger.info(f"Created DPO dataset with {len(dpo_examples)} preference pairs")
        
        return Dataset.from_list(dpo_examples)

    async def load_preference_data(self, dataset_id: str) -> list[dict[str, Any]]:
        """Load preference dataset (only on rank 0, then broadcast)."""
        if dist.get_rank() == 0:
            try:
                all_rows = await self.datasetio_api.iterrows(dataset_id=dataset_id, limit=-1)
                if not isinstance(all_rows.data, list):
                    raise RuntimeError("Expected dataset data to be a list")
                return all_rows.data
            except Exception as e:
                raise RuntimeError(f"Failed to load preference dataset: {str(e)}") from e
        else:
            # Non-rank-0 processes return empty list
            # In practice, you'd want to broadcast the data from rank 0
            return []

    async def load_dataset(
        self,
        model: str,
        config: TrainingConfig,
        provider_config: TrlRemoteFSDPConfig,
    ) -> tuple[Dataset, Dataset, AutoTokenizer]:
        """
        Load and prepare preference dataset for FSDP DPO training.
        
        This method handles distributed dataset loading where only rank 0
        loads the data and then coordinates with other processes.
        """
        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        # Load preference dataset (only on rank 0)
        if dist.get_rank() == 0:
            logger.info(f"Loading preference dataset: {config.data_config.dataset_id}")
        
        rows = await self.load_preference_data(config.data_config.dataset_id)
        
        # Validate dataset
        if dist.get_rank() == 0:
            if not self.validate_preference_dataset(rows):
                raise ValueError("Dataset missing required DPO fields: prompt, chosen, rejected")
            logger.info(f"Loaded {len(rows)} preference examples")

        # Initialize tokenizer (all processes)
        if dist.get_rank() == 0:
            logger.info(f"Initializing tokenizer for model: {model}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            tokenizer.padding_side = "right"
            tokenizer.truncation_side = "right"
            tokenizer.model_max_length = provider_config.max_seq_length

            if dist.get_rank() == 0:
                logger.info("Tokenizer configured for FSDP DPO training")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}") from e

        # Create DPO dataset (all processes need the same data)
        # In a production system, you'd want to broadcast the data from rank 0
        # For now, we assume all processes can load the same data
        if dist.get_rank() == 0:
            logger.info("Creating DPO preference dataset for FSDP training")
        
        try:
            ds = self.create_dpo_dataset(rows, config, provider_config)
            
            # Split for training and evaluation
            train_val_split = ds.train_test_split(test_size=0.1, seed=42)
            train_dataset = train_val_split["train"]
            eval_dataset = train_val_split["test"]
            
            if dist.get_rank() == 0:
                logger.info(f"FSDP dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        except Exception as e:
            raise ValueError(f"Failed to create DPO dataset: {str(e)}") from e

        return train_dataset, eval_dataset, tokenizer

    def load_model_with_fsdp(
        self,
        model: str,
        provider_config: TrlRemoteFSDPConfig,
    ) -> AutoModelForCausalLM:
        """
        Load model and wrap with FSDP for distributed training.
        
        This is the core FSDP setup that enables distributed training
        of large models across multiple GPUs.
        """
        if dist.get_rank() == 0:
            logger.info("Loading model for FSDP DPO training")
        
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            
            # Load model on current device
            device = torch.cuda.current_device()
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16 if provider_config.fsdp_config.mixed_precision_policy == "bf16" else "auto",
                config=model_config,
                trust_remote_code=True,
            )
            
            # Get FSDP configuration
            fsdp_config = provider_config.fsdp_config
            
            # Set up FSDP wrapping policy
            auto_wrap_policy = get_auto_wrap_policy(model)
            
            # Configure FSDP parameters
            fsdp_kwargs = {
                "sharding_strategy": get_fsdp_sharding_strategy(fsdp_config.sharding_strategy),
                "auto_wrap_policy": auto_wrap_policy,
                "mixed_precision": get_fsdp_mixed_precision_policy(fsdp_config.mixed_precision_policy),
                "cpu_offload": get_fsdp_cpu_offload(fsdp_config.cpu_offload),
                "limit_all_gathers": fsdp_config.limit_all_gathers,
                "forward_prefetch": fsdp_config.forward_prefetch,
                "backward_prefetch": fsdp_config.backward_prefetch,
                "device_id": device,
            }
            
            # Wrap model with FSDP
            model_obj = FSDP(model_obj, **fsdp_kwargs)
            
            if dist.get_rank() == 0:
                logger.info(f"Model wrapped with FSDP - Strategy: {fsdp_config.sharding_strategy}")
                logger.info(f"FSDP model device: {device}")
            
            return model_obj
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model with FSDP: {str(e)}") from e

    def setup_fsdp_dpo_config(
        self,
        config: TrainingConfig,
        provider_config: TrlRemoteFSDPConfig,
        dpo_config: DPOAlignmentConfig,
        output_dir_path: Path | None,
        steps_per_epoch: int,
    ) -> DPOConfig:
        """Setup DPO training configuration optimized for FSDP distributed training."""
        if dist.get_rank() == 0:
            logger.info("Configuring DPO training arguments for FSDP")
        
        # DPO learning rate (typically lower for distributed training)
        lr = 5e-5  # Slightly lower than single-GPU default
        if config.optimizer_config:
            lr = config.optimizer_config.lr

        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")
        data_config = config.data_config

        # Calculate training steps (adjusted for world size)
        world_size = dist.get_world_size()
        effective_batch_size = data_config.batch_size * world_size
        steps_per_epoch_distributed = steps_per_epoch // world_size
        
        total_steps = steps_per_epoch_distributed * config.n_epochs
        max_steps = min(config.max_steps_per_epoch, total_steps) if config.max_steps_per_epoch > 0 else total_steps
        
        eval_steps = max(1, steps_per_epoch_distributed // 10)
        save_steps = max(1, steps_per_epoch_distributed // 5)
        logging_steps = max(1, steps_per_epoch_distributed // 50)

        if dist.get_rank() == 0:
            logger.info("FSDP DPO training configuration:")
            logger.info(f"- World size: {world_size}")
            logger.info(f"- Effective batch size: {effective_batch_size}")
            logger.info(f"- Steps per epoch (per process): {steps_per_epoch_distributed}")
            logger.info(f"- Total steps: {total_steps}")
            logger.info(f"- DPO beta: {provider_config.dpo_beta}")
            logger.info(f"- Learning rate: {lr}")

        save_strategy = "steps" if output_dir_path else "no"

        return DPOConfig(
            # Training steps and duration
            max_steps=max_steps,
            output_dir=str(output_dir_path) if output_dir_path is not None else None,
            num_train_epochs=config.n_epochs,
            
            # Distributed batch settings
            per_device_train_batch_size=data_config.batch_size,
            per_device_eval_batch_size=min(data_config.batch_size, 2),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # FSDP-specific settings
            fsdp=f"full_shard auto_wrap",  # Enable FSDP integration in DPOTrainer
            fsdp_config={
                "fsdp_sharding_strategy": provider_config.fsdp_config.sharding_strategy.lower(),
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            },
            
            # Mixed precision for FSDP
            fp16=False,  # FSDP handles precision through MixedPrecision policy
            bf16=provider_config.fsdp_config.mixed_precision_policy == "bf16",
            
            # Evaluation and monitoring
            eval_strategy="steps" if eval_steps > 0 else "no",
            eval_steps=eval_steps if eval_steps > 0 else None,
            
            # Distributed checkpointing
            save_strategy=save_strategy,
            save_steps=save_steps if save_strategy == "steps" else None,
            load_best_model_at_end=True if output_dir_path else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=provider_config.save_total_limit,
            
            # Distributed logging (only rank 0)
            report_to=[],
            logging_steps=logging_steps,
            logging_first_step=True,
            log_level="info" if dist.get_rank() == 0 else "warning",
            disable_tqdm=dist.get_rank() != 0,  # Only show progress on rank 0
            
            # DPO-specific settings
            max_length=provider_config.max_seq_length,
            max_prompt_length=provider_config.max_seq_length // 2,
            gradient_checkpointing=provider_config.gradient_checkpointing,
            remove_unused_columns=False,
            
            # Optimizer settings for distributed training
            learning_rate=lr,
            warmup_ratio=0.1,
            weight_decay=0.01,
            
            # Distributed data loading
            dataloader_pin_memory=provider_config.dataloader_pin_memory,
            dataloader_num_workers=provider_config.dataloader_num_workers,
            dataloader_persistent_workers=provider_config.dataloader_persistent_workers,
            dataloader_drop_last=True,  # Important for distributed training
            
            # Distributed training settings
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            
            # DPO algorithm parameters
            beta=provider_config.dpo_beta,
            loss_type=provider_config.dpo_loss_type,
            label_smoothing=0.0,
        )

    async def train(
        self,
        model: str,
        output_dir: str | None,
        job_uuid: str,
        dpo_config: DPOAlignmentConfig,
        config: TrainingConfig,
        provider_config: TrlRemoteFSDPConfig,
    ) -> tuple[dict[str, Any], list[Checkpoint]]:
        """
        Execute FSDP distributed DPO training.
        
        This method coordinates distributed training across multiple GPUs
        using FSDP for memory-efficient training of large models.
        """
        # Initialize distributed training
        setup_distributed()
        
        if dist.get_rank() == 0:
            logger.info("Starting FSDP distributed DPO training")
            logger.info(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")

        output_dir_path = None
        if output_dir:
            output_dir_path = Path(output_dir)

        # Track memory usage (only on rank 0 to avoid overhead)
        memory_stats = {
            "initial": get_distributed_memory_stats() if dist.get_rank() == 0 else None,
            "after_training": None,
            "final": None,
        }

        if not config.data_config:
            raise ValueError("DataConfig is required for DPO training")

        try:
            # Load preference dataset and tokenizer
            train_dataset, eval_dataset, tokenizer = await self.load_dataset(model, config, provider_config)

            # Calculate steps
            steps_per_epoch = len(train_dataset) // config.data_config.batch_size

            # Setup FSDP DPO configuration
            training_args = self.setup_fsdp_dpo_config(
                config,
                provider_config,
                dpo_config,
                output_dir_path,
                steps_per_epoch,
            )

            # Load model with FSDP wrapping
            model_obj = self.load_model_with_fsdp(model, provider_config)
            
            # Load reference model with FSDP (if using reference model)
            ref_model = None
            if provider_config.use_reference_model:
                if dist.get_rank() == 0:
                    logger.info("Loading reference model with FSDP")
                ref_model = self.load_model_with_fsdp(model, provider_config)

            # Initialize DPO trainer for FSDP training
            if dist.get_rank() == 0:
                logger.info("Initializing DPOTrainer for FSDP distributed training")
            
            trainer = DPOTrainer(
                model=model_obj,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )

            # Execute distributed DPO training
            if dist.get_rank() == 0:
                logger.info("Starting FSDP distributed DPO training")
            
            trainer.train()
            
            if dist.get_rank() == 0:
                logger.info("FSDP distributed DPO training completed successfully")

            # Collect memory stats after training
            if dist.get_rank() == 0:
                memory_stats["after_training"] = get_distributed_memory_stats()

            # Save final model (coordinated across all processes)
            checkpoints = None
            if output_dir_path and dist.get_rank() == 0:
                logger.info("Saving final FSDP DPO model")
                save_path = output_dir_path / "fsdp_dpo_model"
                
                # Save FSDP model using appropriate state dict
                if provider_config.use_fsdp_state_dict:
                    # Use FSDP-optimized state dict
                    FSDP.set_state_dict_type(
                        trainer.model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    )
                
                trainer.model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                
                if dist.get_rank() == 0:
                    logger.info(f"FSDP DPO model saved to {save_path}")

                checkpoint = Checkpoint(
                    identifier=f"{model}-fsdp-dpo-{config.n_epochs}",
                    created_at=datetime.now(timezone.utc),
                    epoch=config.n_epochs,
                    post_training_job_id=job_uuid,
                    path=str(save_path),
                )
                checkpoints = [checkpoint]

            # Synchronize all processes before returning
            dist.barrier()

            return memory_stats, checkpoints

        except Exception as e:
            if dist.get_rank() == 0:
                logger.error(f"FSDP DPO training failed: {str(e)}")
            raise
        finally:
            # Cleanup distributed resources
            if dist.get_rank() == 0:
                logger.info("Cleaning up FSDP training resources")
            
            # Clean up models
            if 'trainer' in locals():
                del trainer
            if 'model_obj' in locals():
                del model_obj
            if 'ref_model' in locals() and ref_model:
                del ref_model
            
            # Collect garbage and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Final memory stats
            if dist.get_rank() == 0:
                memory_stats["final"] = get_distributed_memory_stats()


def main():
    """
    Main function for running FSDP DPO training when launched with torch.distributed.run.
    
    This function is called when the training script is launched with torchrun.
    It handles the distributed training coordination and execution.
    """
    # This would be called when running with torchrun
    # Example: torchrun --nproc_per_node=2 dpo_training_fsdp.py
    
    # Initialize distributed training
    setup_distributed()
    
    # Training logic would go here
    # In practice, this would be called by the remote provider server
    pass


if __name__ == "__main__":
    main() 