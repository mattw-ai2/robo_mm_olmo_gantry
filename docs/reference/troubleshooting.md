# Troubleshooting Guide

Solutions to common issues when using Molmo.

## Installation Issues

### Import Errors

**Problem:** `ModuleNotFoundError` when importing molmo

**Solution:**
```bash
# Reinstall in development mode
pip install -e .[all] --force-reinstall

# Verify Python path
echo $PYTHONPATH

# Activate virtual environment
source venv/bin/activate
```

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns `False`

**Solution:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Missing Dependencies

**Problem:** Missing packages when running code

**Solution:**
```bash
# Install all dependencies
pip install -e .[all]

# For MoE models
pip install git+https://github.com/Muennighoff/megablocks.git@olmoe

# For serving
pip install vllm
```

## Training Issues

### Out of Memory (OOM)

**Problem:** CUDA out of memory during training

**Solutions:**

1. **Reduce batch size:**
```bash
--device_batch_size=2  # Instead of 4
```

2. **Enable gradient checkpointing:**
```bash
--activation_checkpointing=full
```

3. **Use FSDP with full sharding:**
```bash
--fsdp=True --fsdp.sharding_strategy=FULL_SHARD
```

4. **Lower image resolution:**
```bash
--vision_backbone.image_default_input_size=224,224
```

5. **Use gradient accumulation:**
```bash
--device_batch_size=1
--global_train_batch_size=256  # Accumulate gradients
```

### NaN Loss

**Problem:** Loss becomes NaN during training

**Solutions:**

1. **Reduce learning rate:**
```bash
--optimizer.learning_rate=1e-6  # Lower LR
```

2. **Increase warmup:**
```bash
--scheduler.t_warmup=2000  # More warmup steps
```

3. **Use BF16 instead of FP16:**
```bash
--fsdp.precision=bf16
```

4. **Enable gradient clipping:**
```bash
--max_grad_norm=1.0
```

5. **Check data quality:**
```python
# Verify no NaN/Inf in data
for batch in dataloader:
    assert not torch.isnan(batch["images"]).any()
    assert not torch.isinf(batch["images"]).any()
```

### Slow Training

**Problem:** Training is much slower than expected

**Solutions:**

1. **Increase data loading workers:**
```bash
--data.num_workers=8
```

2. **Use local/fast storage:**
```bash
# Copy data to local SSD
cp -r /network/data /local/ssd/data
export MOLMO_DATA_DIR=/local/ssd/data
```

3. **Enable compilation (PyTorch 2.x):**
```bash
--compiler.mode=default
```

4. **Reduce activation checkpointing:**
```bash
--activation_checkpointing=one_in_two  # Instead of full
```

5. **Check data loading time:**
```python
import time
start = time.time()
batch = next(iter(dataloader))
print(f"Data loading time: {time.time() - start:.2f}s")
```

### Training Stops/Hangs

**Problem:** Training freezes or stops progressing

**Solutions:**

1. **Check for deadlocks:**
- Ensure all processes reach barriers
- Check for unbalanced data across ranks

2. **Increase timeout:**
```bash
export NCCL_TIMEOUT_MINUTES=30
```

3. **Check disk space:**
```bash
df -h $MOLMO_DATA_DIR
df -h $SAVE_FOLDER
```

4. **Monitor GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

### Checkpoint Loading Fails

**Problem:** Cannot load checkpoint

**Solutions:**

1. **Check path:**
```python
import os
assert os.path.exists(checkpoint_path)
```

2. **Use correct loading method:**
```python
# For sharded checkpoints
model.load_checkpoint(path, sharded=True)

# For unsharded
model.load_checkpoint(path, sharded=False)
```

3. **Handle version mismatch:**
```python
# Load with strict=False for missing keys
model.load_checkpoint(path, strict=False)
```

## Inference Issues

### Slow Generation

**Problem:** Text generation is very slow

**Solutions:**

1. **Use vLLM for production:**
```bash
vllm serve allenai/Molmo-7B-D-0924 --trust-remote-code
```

2. **Enable KV caching:**
```python
# Automatically enabled in generate()
output = model.generate(**inputs)
```

3. **Use smaller model:**
```python
# Use MolmoE-1B instead of Molmo-7B
model = "allenai/MolmoE-1B-0924"
```

4. **Reduce max_new_tokens:**
```python
output = model.generate(**inputs, max_new_tokens=50)
```

5. **Use FP16/BF16:**
```python
model = model.to(torch.float16)
```

### Poor Quality Outputs

**Problem:** Model generates low-quality responses

**Solutions:**

1. **Use higher resolution:**
```python
# Resize images to higher resolution
image = image.resize((768, 768))
```

2. **Adjust generation parameters:**
```python
output = model.generate(
    **inputs,
    temperature=0.7,  # Lower for more focused
    top_p=0.9,
    do_sample=True
)
```

3. **Use larger model:**
```python
# Use Molmo-7B instead of Molmo-1B
model = "allenai/Molmo-7B-D-0924"
```

4. **Check input formatting:**
```python
# Ensure correct prompt format
text = "Describe this image in detail."  # Clear instruction
```

### Model Returns Empty/Truncated Output

**Problem:** Model generates no text or cuts off early

**Solutions:**

1. **Check EOS token:**
```python
# Ensure EOS token is set correctly
output = model.generate(
    **inputs,
    eos_token_id=processor.tokenizer.eos_token_id
)
```

2. **Increase max_new_tokens:**
```python
output = model.generate(**inputs, max_new_tokens=500)
```

3. **Set min_new_tokens:**
```python
output = model.generate(**inputs, min_new_tokens=10)
```

## Data Issues

### Dataset Not Found

**Problem:** "Dataset not found" or "File not found" errors

**Solutions:**

1. **Set MOLMO_DATA_DIR:**
```bash
export MOLMO_DATA_DIR=/path/to/data
```

2. **Download dataset:**
```bash
python scripts/download_data.py dataset_name --n_proc 12
```

3. **Check path:**
```bash
ls $MOLMO_DATA_DIR/torch_datasets/
```

### Slow Data Loading

**Problem:** Data loading is bottleneck

**Solutions:**

1. **Increase workers:**
```python
dataloader = DataLoader(..., num_workers=8)
```

2. **Use faster storage:**
```bash
# Move to SSD
mv /slow/disk/data /fast/ssd/data
```

3. **Enable offline mode:**
```bash
export HF_DATASETS_OFFLINE=1
```

4. **Prefetch more:**
```python
dataloader = DataLoader(..., prefetch_factor=4)
```

### Corrupted Data

**Problem:** "Cannot identify image file" or parsing errors

**Solutions:**

1. **Redownload dataset:**
```bash
rm -rf $MOLMO_DATA_DIR/dataset_name
python scripts/download_data.py dataset_name
```

2. **Check disk health:**
```bash
# Check for disk errors
dmesg | grep error
```

3. **Verify checksums (if available):**
```bash
# Verify file integrity
md5sum data_file
```

## Evaluation Issues

### Metrics Don't Match Paper

**Problem:** Evaluation results differ from paper

**Possible causes:**

1. **Different settings:**
   - Check resolution (high-res vs low-res)
   - Verify same split (validation vs test)
   - Confirm same model checkpoint

2. **Different preprocessing:**
   - Verify image preprocessing
   - Check prompt format

3. **Solution:**
```bash
# Use exact settings from paper
torchrun --nproc-per-node 8 launch_scripts/eval_downstream.py \
    Molmo-7B-D-0924 text_vqa --high_res --fsdp
```

### Evaluation Crashes

**Problem:** Evaluation fails partway through

**Solutions:**

1. **Reduce batch size:**
```bash
--device_batch_size=1
```

2. **Enable FSDP:**
```bash
--fsdp --device_batch_size=2
```

3. **Use checkpointing:**
```bash
# Evaluation automatically resumes from cached predictions
--skip_if_metrics_cached
```

## Distributed Training Issues

### NCCL Errors

**Problem:** NCCL initialization or communication errors

**Solutions:**

1. **Check network:**
```bash
# Test connectivity between nodes
ping <other_node_ip>
```

2. **Set NCCL debug:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

3. **Use correct interface:**
```bash
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your network interface
```

4. **Increase timeout:**
```bash
export NCCL_TIMEOUT_MINUTES=30
```

### Rank Mismatch

**Problem:** Process rank doesn't match expected

**Solutions:**

1. **Check torchrun arguments:**
```bash
# Ensure correct number of nodes and ranks
torchrun \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank=0  # 0 for first node, 1 for second
```

2. **Verify MASTER_ADDR and MASTER_PORT:**
```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500
```

## Performance Issues

### Low GPU Utilization

**Problem:** GPUs not fully utilized

**Solutions:**

1. **Increase batch size:**
```bash
--device_batch_size=8  # As large as memory allows
```

2. **Remove bottlenecks:**
- Increase num_workers
- Use faster storage
- Reduce logging frequency

3. **Profile code:**
```python
with torch.profiler.profile() as prof:
    for batch in dataloader:
        output = model(**batch)
print(prof.key_averages().table())
```

### High Memory Usage

**Problem:** System running out of RAM

**Solutions:**

1. **Reduce cache sizes:**
```python
# Reduce dataset caching
dataset._shard_cache_size = 2
```

2. **Use lazy loading:**
```python
# Don't load all data at once
dataset = LazyDataset(...)
```

3. **Clear cache periodically:**
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```

## Remote Storage Issues

### Cannot Access Remote Files

**Problem:** Errors accessing GCS/S3/Weka

**Solutions:**

1. **Check credentials:**
```bash
# For GCS
echo $GOOGLE_APPLICATION_CREDENTIALS

# For S3
echo $AWS_ACCESS_KEY_ID

# For Weka
echo $WEKA_ENDPOINT_URL
```

2. **Test connectivity:**
```bash
# Test GCS
gsutil ls gs://bucket/

# Test S3
aws s3 ls s3://bucket/
```

3. **Use local cache:**
```bash
# Download to local first
gsutil -m cp -r gs://bucket/data /local/data
```

## Getting More Help

If these solutions don't resolve your issue:

1. **Check GitHub Issues:** [github.com/allenai/molmo/issues](https://github.com/allenai/molmo/issues)
2. **Ask in Discussions:** [github.com/allenai/molmo/discussions](https://github.com/allenai/molmo/discussions)
3. **Review Documentation:** [Full documentation](../index.md)
4. **Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Reporting Bugs

When reporting issues, include:
- Python version
- PyTorch version
- CUDA version
- Full error traceback
- Minimal reproduction script
- System information (GPU, OS)

