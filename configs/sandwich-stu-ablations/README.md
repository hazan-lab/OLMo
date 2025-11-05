# MLP-Sandwich STU Ablation Experiments

This directory contains configurations for ablation experiments on the MLP-Sandwich STU architecture.

## Architecture Overview

The MLP-Sandwich block implements: **MLP_in → STU → MLP_out**

- `MLP_in`: Projects from `d_model` to `d_hidden` using SwiGLU activation
- `STU`: Spectral Transform Unit operating in hidden space
- `MLP_out`: Projects back from `d_hidden` to `d_model`

**Note**: These experiments use the unified `OLMoSTUBlock` from `olmo/stu.py` with 
`stu_enable_mlp_sandwich: true`. The sandwich features (residual modes, dropout, 
norm placement) are configured via `stu_sandwich_*` config parameters.

## Ablation Dimensions

### Fixed Parameters (Across All Experiments)

- `d_model`: 768
- `n_layers`: 5
- `mlp_ratio`: 8 (→ `d_hidden = 6144`)
- `stu_num_eigh`: 16 (K)
- `stu_use_hankel_L`: false (standard Hankel)
- `stu_use_approx`: true (approximation variant)
- Training steps: 50,000
- Warmup: 5,000 steps
- Global batch size: 1024
- Learning rate: 6e-4

### Ablation Variables

#### 1. Norm Placement (`stu_sandwich_prenorm`)

- **Pre-Norm** (`true`): x + MLP₂(STU(MLP₁(Norm(x))))
- **Post-Norm** (`false`): Norm(x + MLP₂(STU(MLP₁(x))))

#### 2. Residual Mode (`stu_sandwich_residual_mode`)

- **outer**: Standard outer residual wrapping the entire sandwich
- **dual**: Outer residual + internal skip connection inside STU (y = x + STU(x))
- **inner_mlp**: Residual around the projection stack (adds a learnable adapter from `d_hidden` → `d_model`)
- **gated_outer**: Learned gate to modulate residual contribution (y = x + σ(g) · sandwich(x))

#### 3. Dropout (`stu_sandwich_dropout`)

- **0.0**: No dropout
- **0.1**: Dropout after MLP_out and inside STU

## Experiment Configurations

| Config File | Norm | Residual | Dropout |
|-------------|------|----------|---------|
| `OLMo-Sandwich-PreNorm-Outer-150M.yaml` | Pre | Outer | 0.0 |
| `OLMo-Sandwich-PostNorm-Outer-150M.yaml` | Post | Outer | 0.0 |
| `OLMo-Sandwich-PreNorm-Dual-150M.yaml` | Pre | Dual | 0.0 |
| `OLMo-Sandwich-PreNorm-InnerMLP-150M.yaml` | Pre | Inner-MLP | 0.0 |
| `OLMo-Sandwich-PreNorm-Gated-150M.yaml` | Pre | Gated | 0.0 |
| `OLMo-Sandwich-PreNorm-Outer-Drop01-150M.yaml` | Pre | Outer | 0.1 |

## Running Experiments

### Submit All Experiments

```bash
python offline_scripts/run_slurm_job_config.py --config configs/sandwich-stu-ablations/
```

### Submit Individual Experiment

```bash
python offline_scripts/run_slurm_job_config.py --config configs/sandwich-stu-ablations/OLMo-Sandwich-PreNorm-Outer-150M.yaml
```

### Dry Run (Preview SLURM Script)

```bash
python offline_scripts/run_slurm_job_config.py --config configs/sandwich-stu-ablations/OLMo-Sandwich-PreNorm-Outer-150M.yaml --dry-run
```

## Monitoring

All experiments log to W&B:
- **Project**: `olmo-sandwich-stu-ablations`
- **Group**: `sandwich-ablations`

## Expected Results

### Key Metrics to Compare

1. **Validation Loss / Perplexity**: Primary metric for model quality
2. **Throughput (tokens/sec)**: Training efficiency
3. **Peak Memory**: GPU memory usage
4. **Gradient Norm**: Training stability
5. **Downstream Tasks**: PIQA, HellaSwag, WinoGrande, etc.

### Hypotheses

- **Pre-Norm vs Post-Norm**: Pre-Norm typically more stable for deeper models
- **Dual Residual**: May help early optimization and reduce loss spikes
- **Gated Residual**: Can adapt to heterogeneous datasets
- **Inner-MLP Residual**: Helps when `mlp_ratio` is large (≥8)

## Output Locations

Checkpoints are saved to:
```
/scratch/gpfs/EHAZAN/kg4280/OLMo-data/checkpoints/OLMo-Sandwich/{run_name}
```

SLURM logs are saved to:
```
logs/{job_name}_{job_id}.out
logs/{job_name}_{job_id}.err
```

