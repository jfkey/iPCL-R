# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

iPCL-R is a pre-training foundation model for chip layout routing that treats routing patterns as sequences learnable by transformer-based language models. The project implements a complete ML pipeline from EDA data extraction through model training to routing generation and evaluation.

**Key Innovation**: Domain-specific tokenization optimized for spatial reasoning in chip routing, enabling T5-Gemma encoder-decoder models to generate valid routing patterns.

## Architecture

iPCL-R is organized into 4 core modules:

```
iPCL-R/
├── flow/              # Main 3-stage ML pipeline (tokenization → training → evaluation)
├── data_synthesis/    # EDA data processing and ML-ready dataset generation
├── experiments/       # Validation studies and optimization experiments
└── third_party/aieda/ # Git submodule for EDA tool integration
```

### Flow Module (Main Pipeline)

The flow module orchestrates a complete 3-stage machine learning workflow:

**Stage 1: Tokenization**
- **UnifiedTokenizer**: Supports 5 algorithms (DecimalWordLevel, Seg-BPE, Concat-BPE, Seg-BBPE, Concat-BBPE)
- **Spatial Encoding**: Direction tokens (R/L/U/D/T/B) for 3D coordinate conversion
- **Tree Structure**: PUSH/POP and BRANCH/END tokens for routing tree hierarchies
- **Special Tokens**: BOS/EOS, PAD, DRIVER/LOAD for routing semantics
- Entry point: `python -m flow.launch_tokenization --flow-config config.json`

**Stage 2: Training**
- **Model Architecture**: T5-Gemma encoder-decoder transformer with configurable depths/widths
- **Distributed Training**: HuggingFace Accelerate integration for multi-GPU training
- **Optimization**: Support for AdamW, Adafactor, and Lion optimizers with configurable schedulers
- **Monitoring**: TensorBoard logging and comprehensive checkpointing
- Entry point: `accelerate launch -m flow.launch_training --flow-config config.json`

**Stage 3: Evaluation**
- **Multi-metric Assessment**: ROUGE, BLEU (NLP) + RED routing-specific metrics
- **Inference**: Beam search generation with configurable decoding strategies
- **Validation**: Coordinate parsing, tree structure analysis, EDA DEF format output
- Entry point: `accelerate launch -m flow.launch_evaluation --flow-config config.json`

### Configuration System

The entire pipeline is controlled by a unified JSON configuration file (FlowConfig):

```json
{
  "dataset": { "source": "hub|local", "hub_id": "AiEDA/iPCL-R", ... },
  "tokenization": { "paths": {...}, "workflow": {...}, "advanced": {...} },
  "training": { "paths": {...}, "model": {...}, "hyperparameters": {...} },
  "evaluation": { "paths": {...}, "generation": {...}, "metrics": {...} }
}
```

Generate a default config: `python -m flow.pipeline_init --create-flow-config config.json`

See `flow/README.md` for complete configuration reference with all fields and defaults.

### Data Synthesis Module

Converts EDA design data into HuggingFace Dataset format. Generates:
- `net_seqs`: Network sequences with driver/load information
- `pin2pin_pattern_seqs`: Pin-to-pin routing patterns
- `pin2pin_loc_seqs`: Spatial location sequences
- `design_graph`: Design-level connectivity graphs

Entry point: `python -m data_synthesis.main_aggregation`

Uses infrastructure from `third_party/aieda` git submodule for EDA file parsing.

### Experiments Module

Validation and optimization studies:
- **Tokenizer Comparison**: Statistical analysis across 5 algorithms and vocabulary sizes
- **Model Architecture**: Parameter scaling studies (small/medium/large variants)
- **LLM Fine-tuning**: Supervised fine-tuning with LoRA (Qwen support)
- **Symbol Analysis**: Domain-specific vs human language tokenization
- **Feature Ablation**: Input feature importance analysis (planned)
- **Demo**: Serialization and tree visualization

## Common Commands

### Setup and Configuration
```bash
# Install dependencies
pip install -r requirements.txt

# Generate pipeline configuration template
python -m flow.pipeline_init --create-flow-config config.json

# Customize paths and hyperparameters in generated config
vim config.json
```

### Pipeline Execution (Standard Flow)
```bash
# Stage 1: Tokenization (single-GPU)
python -m flow.launch_tokenization --flow-config config.json

# Stage 2: Training (distributed, requires accelerate config)
# First time: run 'accelerate config' to set up distributed training
accelerate launch -m flow.launch_training --flow-config config.json

# Stage 3: Evaluation (distributed)
accelerate launch -m flow.launch_evaluation --flow-config config.json
```

### Accelerate Configuration
```bash
# Initialize accelerate for distributed training (multi-GPU, mixed precision, etc.)
accelerate config

# View current accelerate config
accelerate config --config_file ~/.cache/huggingface/accelerate/default_config.yaml
```

### Experiments
```bash
# Tokenizer comparison analysis
python -m experiments.tokenizer_comparison.init_env --work-dir /path/to/work_dir
source /path/to/work_dir/run_tokenizer_comparison.sh

# Model size comparison
python -m experiments.model_size_comparison.init_env --work-dir /path/to/work_dir
source /path/to/work_dir/run_model_size_comparison.sh

# Symbol analysis (domain-specific vs human language tokenization)
python -m experiments.symbol_analysis.main

# LLM fine-tuning
accelerate launch -m experiments.sft_llm.training
python -m experiments.sft_llm.evaluation

# Demo visualization
python -m experiments.demo.serialization          # Routing serialization
python -m experiments.demo.treeization            # Tree visualization (mp4)
```

### Data Synthesis (Custom Dataset Construction)
```bash
# Process specific designs
python -m data_synthesis.main_aggregation --design_list nvdla shanghai_MS --output_dir /path/to/output

# Batch processing with specific data types
python -m data_synthesis.main_aggregation --design_list design1 design2 --data_types net_seqs pin2pin_pattern_seqs

# With detailed logging and rebuild
python -m data_synthesis.main_aggregation --enable_dataset_logs --rebuild
```

## Key Dependencies

Core ML libraries:
- **transformers**: HuggingFace models, tokenizers, training
- **torch**: Deep learning framework (not in requirements.txt, pre-installed)
- **accelerate**: Distributed training orchestration
- **datasets**: HuggingFace dataset loading and processing
- **lion-pytorch**: Lion optimizer implementation
- **deepspeed**: Memory-efficient distributed training

Data/EDA processing:
- **networkx**: Graph analysis for routing networks
- **pandas**: Data manipulation
- **tokenizers**: Low-level tokenization library
- **networkx, rtree, scipy**: Spatial and graph computations

Monitoring/Visualization:
- **tensorboard**: Training visualization
- **wandb**: Experiment tracking
- **matplotlib, seaborn, scienceplots**: Scientific plotting
- **plotly**: Interactive visualization

Note: torch and torchvision are commented out in requirements.txt as they should be pre-installed based on your CUDA environment.

## Code Organization Notes

### Special Token System
Located in `flow/utils/special_tokens.py`:
- Core tokens: BOS, EOS, PAD, SRC_END, UNK_LEN (always included)
- Source tokens: DRIVER, LOAD (always included)
- Indexed load tokens: RLOAD, ALOAD (1-20 indexed variants + overflow)
- Conditional tokens: Overlap and connectivity info (optional, controlled by config)

The SpecialTokenManager handles all token generation and validation.

### Coordinate System
Located in `flow/utils/constants.py`:
- Directions map to 3D movements: R/L (±X), U/D (±Y), T/B (±Z metal layer)
- Coordinate format: `(x, y, metal_layer)` with regex validation
- Direction tokens: `[RLUDTB]\d+` pattern (e.g., `R123`, `T5`)

### Configuration Hierarchy
- FlowConfig (root) contains DatasetConfig, TokenizationStageConfig, TrainingStageConfig, EvaluationStageConfig
- Each stage config has nested dataclasses for organization (paths, workflow, hyperparameters, performance)
- From-dict parsing with defaults allows flexible JSON loading
- See `flow/config.py` for complete structure

### Model Architecture
Uses T5-Gemma (encoder-decoder) with configurable:
- `hidden_size`, `intermediate_size`: Model dimensions
- `num_hidden_layers`: Depth (encoder and decoder separately)
- `num_attention_heads`, `num_key_value_heads`: Attention configuration
- `max_position_embeddings`: Sequence length support
- `sliding_window`: For efficient attention computation

### Dataset Format (HuggingFace Hub)
Pre-made datasets available:
- **AiEDA/iPCL-R**: Main dataset on HuggingFace Hub with splits: train, validation
- Local dataset support: Point to local directories with `source: "local"` in config

## Important Implementation Details

### Tokenization Pipeline (Stage 1)
1. **Preprocessing**: Corpus preprocessing with coordinate parsing and direction encoding
2. **Tokenizer Training**: UnifiedTokenizer trains on preprocessed corpus
3. **Dataset Conversion**: Raw sequences → token IDs using trained tokenizer
4. **Metadata Export**: Vocabulary size, sequence statistics, special token info

Key classes:
- `UnifiedTokenizer`: Main tokenizer supporting 5 algorithms and coordinate/direction handling
- `TokenizationPipeline`: Orchestrates preprocessing, training, and dataset creation
- `UnifiedTokenPreprocessor`: Handles coordinate parsing, direction encoding, special tokens

### Training Pipeline (Stage 2)
1. **Dataset Loading**: Load and split token dataset from Stage 1
2. **Model Initialization**: T5GemmaForConditionalGeneration with config from FlowConfig
3. **Training Loop**: Seq2SeqTrainer with distributed training via Accelerate
4. **Checkpoint Management**: Periodic saves with early stopping capability
5. **Logging**: TensorBoard and console logging

Optimizers: AdamW (default), Adafactor, Lion with cosine/linear schedulers

### Evaluation Pipeline (Stage 3)
1. **Dataset Preprocessing**: Load validation split and apply tokenization preprocessing
2. **Model Inference**: Generate routing patterns using beam search decoding
3. **Metric Calculation**: 
   - NLP: ROUGE, BLEU, exact match
   - Domain: RED (Routing Edit Distance), coordinate accuracy
   - Structure: Tree validation, connectivity analysis
4. **EDA Output**: Save predictions in DEF format for industry tool verification
5. **Visualization**: Generate metric plots and result summaries

## Data Format (Coordinate and Token Representation)

Raw routing sequences use 3D coordinates and direction encoding:
```
Network: DRIVER <coord> LOAD <coord> LOAD <coord> ROUTE_SEQUENCE
Direction tokens: R/L/U/D (xy-plane), T/B (layer changes)
Direction token format: [DIRECTION][DISTANCE] e.g., R123 (move right 123 units)
Tree structure: PUSH/POP for hierarchy, BRANCH/END for alternatives
```

The `data-format` skill provides detailed token/coordinate specifications.

## Git and Versioning

- **Remote**: https://github.com/jfkey/iPCL-R.git
- **Submodule**: `third_party/aieda` (AiEDA git submodule for EDA integration)
- **License**: Apache 2.0 (see LICENSE file)
- Note: Initialize submodule with `git submodule update --init --recursive` if cloning fresh

## Testing and Validation

The experiments module provides comprehensive validation:
- Tokenization correctness via symbol analysis and comparison studies
- Model training via loss tracking and metric monitoring (TensorBoard)
- Evaluation via multi-metric assessment (ROUGE, BLEU, RED)
- Tree structure integrity checking in coordinate parsing

Use `flow/launch_evaluation.py` to validate trained models or use experiment scripts for comparative analysis.

## Performance Tuning

Key hyperparameters in config:
- **Tokenization**: `num_workers` (preprocessing parallelism), `batch_size` (batched corpus processing)
- **Training**: `batch_size_per_device`, `gradient_accumulation_steps`, `dataloader_num_workers`, `dataloader_pin_memory`
- **Evaluation**: `num_beams` (beam search width affects generation quality/speed)
- **Model**: `hidden_size`, `num_hidden_layers` (trade-off between capacity and speed)

For distributed training, use Accelerate with mixed precision and gradient checkpointing (configurable via `accelerate config`).

## Common Patterns and Conventions

1. **Config-Driven**: All pipelines load configuration from FlowConfig JSON at runtime
2. **Modular Pipelines**: Each stage (tokenization, training, evaluation) is independently executable
3. **Special Token Management**: Centralized in SpecialTokenManager for consistency
4. **Spatial Encoding**: Direction tokens and coordinate parsing are core to all modules
5. **Distributed Ready**: All stages support Accelerate for multi-GPU training/inference
6. **Logging**: Comprehensive logging via flow.utils.setup_logging and TensorBoard

