# OmniQuality-R: Advancing Reward Models through All-Encompassing Quality Assessment

<div align="center">
    <img alt="OmniQuality-R logo" src="./docs/omniquality-logo.png" style="height: 140px;" />
</div>

<div align="center">
<p align="center">
      <a href="https://github.com/OpenRLHF/OpenRLHF-M">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/OpenRLHF/OpenRLHF-M" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF-M/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/OpenRLHF/OpenRLHF-M?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF-M/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/OpenRLHF/OpenRLHF-M?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF-M/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/OpenRLHF/OpenRLHF-M?color=0088ff" />
      <a href="https://github.com/OpenRLHF/OpenRLHF-M/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF-M?color=ccf" />
      </a>
      <br>
      <em>Unified / Multi-task / Interpretable / Reward Modeling</em>
    </p>
</p>
</div>

<hr>

## Abstract

Current visual evaluation approaches are typically constrained to a single task — focusing either on technical quality for low-level distortions, aesthetic quality for subjective visual appeal, or text-image alignment for semantic consistency. With the growing role of reward models in guiding generative systems, there is a need to extend into an all-encompassing quality assessment form that integrates multiple tasks. To address this, we propose **OmniQuality-R**, a unified reward modeling framework that transforms multi-task quality reasoning into continuous and interpretable reward signals for policy optimization.

Inspired by subjective experiments, where participants are given task-specific instructions outlining distinct assessment principles prior to evaluation, we propose OmniQuality-R, a structured reward modeling framework that transforms multi-dimensional reasoning into continuous and interpretable reward signals.

To enable this, we construct a reasoning-enhanced reward modeling dataset by sampling informative plan-reason trajectories via rejection sampling, forming a reliable chain-of-thought (CoT) dataset for supervised fine-tuning (SFT). Building on this, we apply Group Relative Policy Optimization (GRPO) for post-training, using a Gaussian-based reward to support continuous score prediction. To further stabilize the training and improve downstream generalization, we incorporate standard deviation (STD) filtering and entropy gating mechanisms during reinforcement learning. These techniques suppress unstable updates and reduce variance in policy optimization. We evaluate OmniQuality-R on three key IQA tasks: aesthetic quality assessment, technical quality evaluation, and text-image alignment. Experiments show OmniQuality-R improves robustness, explainability, and generalization, and can guide text-to-image generation models at test time without retraining by serving as an interpretable reward function.

## Key Features

- **Multi-task Quality Assessment**: Unified framework for aesthetic quality, technical quality, and text-image alignment
- **Chain-of-Thought Reasoning**: Enhanced reward modeling with informative plan-reason trajectories
- **Group Relative Policy Optimization (GRPO)**: Advanced RL training with Gaussian-based rewards
- **Stability Mechanisms**: STD filtering and entropy gating for robust training
- **Interpretable Rewards**: Continuous and explainable reward signals for policy optimization
- **No Retraining Required**: Can guide text-to-image generation models at test time

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ or 12.1+
- 4+ GPUs (recommended for training)

### Setup Environment

```bash
# Create conda environment
conda create --name omniquality python=3.10
conda activate omniquality

# Clone repository
git clone https://github.com/OpenRLHF/OpenRLHF-M.git
cd OpenRLHF-M

# Install dependencies
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
pip install git+https://github.com/huggingface/Math-Verify.git

# Install additional requirements
pip install -r requirements.txt
```

### Docker Installation (Optional)

We provide Docker support for easy deployment:

```bash
# Build Docker image
docker build -f dockerfile/Dockerfile -t omniquality-r .

# Run with NVIDIA Docker
bash examples/scripts/docker_run.sh
```

## Quick Start

### Three-Stage Training Pipeline

OmniQuality-R follows a three-stage training approach:

#### Stage 1: Chain-of-Thought (CoT) Dataset Construction
Build reasoning-enhanced reward modeling dataset through rejection sampling to create informative plan-reason trajectories.

#### Stage 2: Supervised Fine-Tuning (SFT)
Train the model on the CoT dataset to establish foundational reasoning capabilities.

```bash
# Example SFT training
python -m openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset your_cot_dataset \
    --input_key message \
    --output_key response \
    --train_batch_size 256 \
    --micro_train_batch_size 2 \
    --max_samples 500000 \
    --pretrain Qwen/Qwen2.5-VL-Instruct-8B \
    --save_path ./checkpoint/omniquality-sft \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --load_checkpoint \
    --gradient_checkpointing
```

#### Stage 3: Reinforcement Learning Training

**Stage 3a: Initial RL Training**
```bash
bash ./examples/scripts/omniquality-R/train_grpo_ava_evalmuse_koniq_7B_RLstage1.sh
```

**Stage 3b: Advanced RL Training with Stability Mechanisms**
```bash
bash ./examples/scripts/omniquality-R/train_grpo_ava_evalmuse_koniq_7B_RLstage2.sh
```

### Configuration

Before running the training scripts, modify the configuration variables in the script files:

```bash
# Key configuration variables to modify:
export WORKSPACE_DIR="$(pwd)"                      # Path to project root
export DATASET_PATH="./data_process_v1/train_ava_mini_evalmuse_koniq_llavastyle_openrlhf_merged.jsonl"
export PRETRAIN_MODEL_PATH="./ckpt/output_sft_cot/"  # Path to SFT model
export SAVE_PATH="./checkpoints_rl_cot/"           # Path to save checkpoints
export MODEL_NAME="iqa-r1-ava-evalmuse-koniq-grpo-score-7B-rl-stage1"
```

## Dataset Format

The training data should be in OpenAI-compatible message format:

```json
[
  {
    "message": "[
      {
        \"role\": \"user\",
        \"content\": [
            {
                \"type\": \"image\",
                \"image\": \"file:///path/to/your/image.jpg\",
            },
            {\"type\": \"text\", \"text\": \"Assess the aesthetic quality of this image.\"}
        ],
      }
    ]",
    "answer": 5,
  }
]
```

## Training Details

### Stage 1: CoT Dataset Construction
- **Method**: Rejection sampling to select informative plan-reason trajectories
- **Output**: High-quality chain-of-thought reasoning dataset
- **Purpose**: Establish reliable reasoning patterns for reward modeling

### Stage 2: Supervised Fine-Tuning
- **Model**: Base language model (e.g., Llama-3-8B)
- **Dataset**: CoT reasoning dataset
- **Objective**: Learn foundational reasoning capabilities
- **Key Features**: FlashAttention, gradient checkpointing, mixed precision

### Stage 3: Reinforcement Learning

#### Stage 3a: Initial GRPO Training
- **Algorithm**: Group Relative Policy Optimization
- **Reward Type**: Gaussian RBF with initial sigma 0.8
- **Key Parameters**:
  - Learning rate: 1e-6
  - Batch size: 64 (4 GPUs × 16)
  - Temperature: 1.0
  - Samples per prompt: 16

#### Stage 3b: Advanced RL with Stability Mechanisms
- **Enhanced Features**: STD filtering and entropy gating
- **Key Parameters**:
  - Learning rate: 1e-7 (reduced for stability)
  - Entropy rho: 0.2
  - Min Gaussian std: 0.001
  - Episodes: 8 (increased from 2)

## Model Architecture

OmniQuality-R builds upon the OpenRLHF framework with the following key components:

- **Base Model**: Large Language Model (e.g., Llama-3-8B)
- **Reward Model**: Gaussian-based continuous reward prediction
- **Training Framework**: Ray-based distributed training
- **Inference Engine**: vLLM for efficient generation

## Evaluation

The model is evaluated on three key IQA tasks:

1. **Aesthetic Quality Assessment**: Subjective visual appeal evaluation
2. **Technical Quality Evaluation**: Low-level distortion assessment
3. **Text-Image Alignment**: Semantic consistency verification

## Monitoring and Logging

### TensorBoard
```bash
tensorboard --logdir ./checkpoints_rl_cot/your_model_name/logs
```

### Wandb (Optional)
Set your Wandb API key in the training scripts:
```bash
export WANDB_API_KEY="your_api_key"
export WANDB_MODE=online
```

### Log Files
Training logs are saved to:
```
./checkpoints_rl_cot/your_model_name/logs/timestamp/
├── train.log
├── remote_rm_qa.log
└── process_pids.txt
```

## Performance

OmniQuality-R demonstrates significant improvements in:
- **Robustness**: Enhanced stability through STD filtering
- **Explainability**: Interpretable reward signals via CoT reasoning
- **Generalization**: Better performance across diverse quality assessment tasks
- **Efficiency**: 4.7x speedup compared to baseline methods

## Usage as Reward Function

Once trained, OmniQuality-R can be used as an interpretable reward function for text-to-image generation models without retraining:

```python
from openrlhf.models.remote_rm import OmniQualityRewardModel

# Load trained model
reward_model = OmniQualityRewardModel.load_from_checkpoint("path/to/checkpoint")

# Evaluate image quality
reward_score = reward_model.get_reward(image_path, prompt)
print(f"Quality Score: {reward_score}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Ray Connection Issues**: Ensure ports are available and Ray is properly installed
3. **Model Loading Errors**: Check model paths and ensure checkpoints are compatible

### Performance Optimization

- Use `--flash_attn` for memory efficiency
- Enable `--gradient_checkpointing` for large models
- Adjust `--vllm_gpu_memory_utilization` based on available memory
- Use `--colocate_all_models` for single-node training

## Citation

If you find OmniQuality-R useful for your research, please cite:

```bibtex
@article{omniquality2024,
  title={OmniQuality-R: Advancing Reward Models through All-Encompassing Quality Assessment},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the OpenRLHF team for providing the excellent RLHF infrastructure and the research community for their valuable contributions to multimodal reasoning and reward modeling.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest improvements.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

<div align="center">
  <p><em>Empowering multimodal systems with unified quality assessment</em></p>
</div>