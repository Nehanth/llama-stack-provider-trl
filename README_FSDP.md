# TRL Provider for Llama Stack - FSDP Distributed Training

This repository now includes **two complementary TRL providers** for different training scenarios:

1. **Inline Provider** (`llama_stack_provider_trl`) - Single-device DPO training
2. **Remote FSDP Provider** (`llama_stack_provider_trl_remote`) - Multi-GPU distributed DPO training

## Provider Comparison

| Feature | Inline Provider | Remote FSDP Provider |
|---------|----------------|----------------------|
| **Training Type** | Single device only | Multi-GPU distributed |
| **Architecture** | Runs in Llama Stack process | Separate service + torchrun |
| **Memory Usage** | Limited to single GPU | Sharded across multiple GPUs |
| **Model Size** | Small to medium models | Large models (7B+ parameters) |
| **Setup Complexity** | Simple | Moderate (distributed setup) |
| **Use Case** | Development, small models | Production, large models |

## Quick Start - FSDP Distributed Training

### 1. Setup Environment

```bash
# Install dependencies for both providers
./scripts/prepare-env.sh

# Build distribution with FSDP support
llama stack build --config build_fsdp.yaml
```

### 2. Start Remote FSDP Provider

In one terminal, start the remote FSDP provider server:

```bash
# Start the remote provider server (runs on port 8322)
./scripts/run-remote-fsdp-server.sh
```

The server will start and show:
```
Starting TRL Remote FSDP Provider Server...
CUDA detected - distributed training will use GPUs
Using CUDA devices: 0,1
Server will listen on localhost:8322
```

### 3. Start Llama Stack with FSDP Configuration

In another terminal, start Llama Stack configured to use the remote provider:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Llama Stack with FSDP configuration
llama stack run --image-type venv --image-name trl-post-training-fsdp run_fsdp.yaml
```

### 4. Run Distributed DPO Training

Use the same API as the inline provider, but now training will be distributed:

```python
import requests

base_url = "http://127.0.0.1:8321"

# Upload preference dataset (same as inline provider)
dataset_payload = {
    "dataset_id": "large-dpo-dataset",
    "purpose": "post-training/messages",
    "dataset_type": "preference",
    "source": {
        "type": "rows",
        "rows": [
            {
                "prompt": "Explain distributed training",
                "chosen": "Distributed training splits model and data across multiple GPUs to enable training of larger models that don't fit on a single GPU. FSDP (Fully Sharded Data Parallel) shards model parameters, gradients, and optimizer states across devices.",
                "rejected": "Distributed training is just using multiple computers."
            },
            # Add more preference pairs for better training
        ]
    }
}

# Start distributed FSDP DPO training
train_payload = {
    "job_uuid": "fsdp-dpo-large-model",
    "model": "meta-llama/Llama-2-7b-hf",  # Large model that benefits from FSDP
    "finetuned_model": "llama-2-7b-dpo-fsdp",
    "checkpoint_dir": "./fsdp_checkpoints",
    "algorithm_config": {
        "type": "dpo",
        "reward_scale": 1.0,
        "reward_clip": 5.0,
    },
    "training_config": {
        "n_epochs": 3,
        "max_steps_per_epoch": 100,
        "learning_rate": 5e-5,  # Lower LR for distributed training
        "data_config": {
            "dataset_id": "large-dpo-dataset",
            "batch_size": 4,  # Per-device batch size
            "shuffle": True,
        }
    }
}

response = requests.post(f"{base_url}/v1/post-training/preference-optimize", 
                        json=train_payload)
print("FSDP Training Started:", response.json())
```

## FSDP Configuration Guide

### Key FSDP Parameters

The remote provider supports extensive FSDP configuration in `run_fsdp.yaml`:

```yaml
fsdp_config:
  # Sharding strategies (memory vs communication tradeoff)
  sharding_strategy: "FULL_SHARD"     # Most memory efficient
  # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
  
  # Mixed precision for performance
  mixed_precision_policy: "bf16"      # Use bfloat16
  # Options: "bf16", "fp16", null
  
  # CPU offloading for very large models
  cpu_offload: false                  # Set true for 13B+ models
  
  # Performance optimizations
  limit_all_gathers: true
  forward_prefetch: true
  backward_prefetch: "BACKWARD_PRE"

torch_run_config:
  nnodes: 1                          # Number of machines
  nproc_per_node: 2                  # GPUs per machine
  master_port: 29500                 # Coordination port
```

### Memory Optimization Tips

1. **For 7B models on 2x24GB GPUs:**
   ```yaml
   fsdp_config:
     sharding_strategy: "FULL_SHARD"
     mixed_precision_policy: "bf16"
     cpu_offload: false
   max_seq_length: 4096
   batch_size: 2
   ```

2. **For 13B+ models:**
   ```yaml
   fsdp_config:
     sharding_strategy: "FULL_SHARD"
     mixed_precision_policy: "bf16"
     cpu_offload: true              # Offload to CPU
   max_seq_length: 2048             # Shorter sequences
   batch_size: 1                    # Smaller batch size
   gradient_checkpointing: true     # Essential for memory
   ```

3. **For 70B+ models (8+ GPUs):**
   ```yaml
   torch_run_config:
     nproc_per_node: 8              # Use all 8 GPUs
   fsdp_config:
     sharding_strategy: "FULL_SHARD"
     cpu_offload: true
   activation_checkpointing: "full"  # Maximum memory savings
   ```

## Distributed Training Architecture

### Process Flow

1. **Llama Stack** receives training request on port 8321
2. **Remote Provider** receives HTTP request on port 8322
3. **torchrun** launches multiple training processes:
   ```
   Rank 0 (GPU 0): Model shard + coordination
   Rank 1 (GPU 1): Model shard + training
   ```
4. **FSDP** handles parameter synchronization between ranks
5. **Checkpoints** saved in FSDP format for efficient loading

### Monitoring Distributed Training

Check training progress:

```python
# Monitor job status
response = requests.get(f"{base_url}/v1/post-training/job/status?job_uuid=fsdp-dpo-large-model")
status = response.json()

print(f"Status: {status['status']}")
print(f"Resources: {status['resources_allocated']}")
if status['checkpoints']:
    print(f"Latest checkpoint: {status['checkpoints'][-1]['path']}")
```

### Log Files

- **Remote Provider**: `/tmp/trl_remote_fsdp_provider.log`
- **Distributed Training**: Check console output from torchrun
- **Individual Ranks**: Logs from each GPU process

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   - Reduce `batch_size` or `max_seq_length`
   - Enable `cpu_offload: true`
   - Use `activation_checkpointing: "full"`

2. **"Process group initialization failed"**
   - Check `master_port` is available
   - Ensure NCCL is properly installed
   - Verify GPU visibility with `nvidia-smi`

3. **"Remote provider not responding"**
   - Confirm remote server is running on port 8322
   - Check firewall settings
   - Verify network connectivity

4. **Slow distributed training**
   - Use faster interconnect (InfiniBand > Ethernet)
   - Increase `batch_size` to improve GPU utilization
   - Enable `forward_prefetch` and `backward_prefetch`

### Performance Tuning

1. **Optimize Batch Size**: Use largest batch size that fits in memory
2. **Tune Learning Rate**: Distributed training often needs lower LR
3. **Monitor GPU Utilization**: Use `nvidia-smi` during training
4. **Profile Memory Usage**: Check memory stats in job status

## Advanced Usage

### Multi-Node Training

For training across multiple machines:

```yaml
torch_run_config:
  nnodes: 2                    # 2 machines
  nproc_per_node: 4           # 4 GPUs per machine
  node_rank: 0                # Set to 0 on master, 1 on worker
  master_addr: "10.0.0.100"   # IP of master node
  master_port: 29500
```

Start on master node (node_rank: 0):
```bash
./scripts/run-remote-fsdp-server.sh
```

Start on worker nodes (node_rank: 1):
```bash
export NODE_RANK=1
export MASTER_ADDR=10.0.0.100
./scripts/run-remote-fsdp-server.sh
```

### Custom Model Support

To add support for new model architectures, update the auto-wrap policy in `dpo_training_fsdp.py`:

```python
def get_auto_wrap_policy(model_type: str):
    transformer_layer_cls = {
        "llama": LlamaDecoderLayer,
        "gpt2": GPT2Block,
        "mistral": MistralDecoderLayer,  # Add your model here
    }
    # ...
```

## Comparison with Inline Provider

You can run both providers simultaneously for different use cases:

- **Inline Provider** (port 8321): Development and small model experiments
- **Remote FSDP Provider** (port 8322): Production training of large models

Choose based on your model size and computational requirements:

| Model Size | Recommended Provider | Configuration |
|------------|---------------------|---------------|
| < 1B parameters | Inline | Single GPU, standard settings |
| 1B - 7B | Either | Single GPU (inline) or 2-GPU FSDP |
| 7B - 30B | Remote FSDP | 2-8 GPUs with FSDP |
| 30B+ | Remote FSDP | 8+ GPUs, CPU offload, aggressive sharding |

This dual-provider approach gives you flexibility to use the right tool for each training scenario. 