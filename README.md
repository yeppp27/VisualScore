# OmniQuality-R: Advancing Reward Models through All-Encompassing Quality Assessment




<hr>

[![🤗 HF Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/yeeeeeyy/OmniQuality-R) [![📄 Paper](https://img.shields.io/badge/📄-Paper-green)](https://arxiv.org/pdf/2510.10609) [![🌐 GitHub](https://img.shields.io/badge/🌐-GitHub-purple)](https://github.com/yeppp27/OmniQuality-R)

## Abstract

Current visual evaluation approaches are typically constrained to a single task — focusing either on technical quality for low-level distortions, aesthetic quality for subjective visual appeal, or text-image alignment for semantic consistency. With the growing role of reward models in guiding generative systems, there is a need to extend into an all-encompassing quality assessment form that integrates multiple tasks. To address this, we propose **OmniQuality-R**, a unified reward modeling framework that transforms multi-task quality reasoning into continuous and interpretable reward signals for policy optimization.

Inspired by subjective experiments, where participants are given task-specific instructions outlining distinct assessment principles prior to evaluation, we propose OmniQuality-R, a structured reward modeling framework that transforms multi-dimensional reasoning into continuous and interpretable reward signals.

To enable this, we construct a reasoning-enhanced reward modeling dataset by sampling informative plan-reason trajectories via rejection sampling, forming a reliable chain-of-thought (CoT) dataset for supervised fine-tuning (SFT). Building on this, we apply Group Relative Policy Optimization (GRPO) for post-training, using a Gaussian-based reward to support continuous score prediction. To further stabilize the training and improve downstream generalization, we incorporate standard deviation (STD) filtering and entropy gating mechanisms during reinforcement learning. These techniques suppress unstable updates and reduce variance in policy optimization. We evaluate OmniQuality-R on three key IQA tasks: aesthetic quality assessment, technical quality evaluation, and text-image alignment. Experiments show OmniQuality-R improves robustness, explainability, and generalization, and can guide text-to-image generation models at test time without retraining by serving as an interpretable reward function.

## Key Features

- **Multi-task Quality Assessment**: Unified framework for aesthetic quality, technical quality, and text-image alignment
- **Chain-of-Thought Reasoning**: Enhanced reward modeling with informative plan-reason trajectories
- **Test-Time Guidance**: Can guide text-to-image generation models at test time

## Model Checkpoints

We provide pre-trained OmniQuality-R models on Hugging Face:

### 🤗 Hugging Face Models

| Model | Size | Description | Download |
|-------|------|-------------|----------|
| **OmniQuality-R** | 8.29B | Pre-trained OmniQuality-R model with Qwen2.5-VL backbone | [![🤗 HF](https://img.shields.io/badge/🤗-Download-blue)](https://huggingface.co/yeeeeeyy/OmniQuality-R) |


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
git clone https://github.com/yeppp27/OmniQuality-R.git
cd OmniQuality-R

# Install dependencies
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
pip install git+https://github.com/huggingface/Math-Verify.git

# Install additional requirements
pip install -r requirements.txt
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

The training data should be in the following format:

```json
[
  {
    "message": [
      {
        "role": "system",
        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
      },
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "file:///path/to/your/image.jpg"
          },
          {
            "type": "text",
            "text": "Check how aligned the image is with this prompt: \"A nurse in purple scrubs checks a patient's vitals, her straight blonde hair neatly tied back in a ponytail.\"\n\nAnd evaluate the image’s alignment rating.\n\nGive a final rating for these dimensions from 0 to 5 (float, 2 decimals). A rating of 0 represents very poor level, while 5 represents excellent level."
          }
        ]
      }
    ],
    "answer": "2.971"
  }
]

"message": "[{\"role\": \"system\", \"content\": \"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\"}, {\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file:///path/to/your/image.jpg\"}, {\"type\": \"text\", \"text\": \"Check how aligned the image is with this prompt: \\\"A nurse in purple scrubs checks a patient's vitals, her straight blonde hair neatly tied back in a ponytail.\\\"\\n\\nAnd evaluate the image’s alignment rating.\\n\\nGive a final rating for these dimensions from 0 to 5 (float, 2 decimals). A rating of 0 represents very poor level, while 5 represents excellent level\"}]}]", "answer": "2.971"}

```



## Evaluation

The model is evaluated on three key IQA tasks:

1. **Aesthetic Quality Assessment**: Subjective visual appeal evaluation
2. **Technical Quality Evaluation**: Low-level distortion assessment
3. **Text-Image Alignment**: Semantic consistency verification


## Usage as Reward Function

Once trained, OmniQuality-R can be used as an interpretable reward function for text-to-image generation models without retraining:


### Using Local Trained Model

```python
import os
import glob
import re
import json
import argparse
from collections import defaultdict
import pandas as pd
import torch
from PIL import Image
import numpy as np
from contextlib import contextmanager
import signal
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import time

class QualityRater:
    def __init__(self, model_path, device):
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            model_path,
            min_pixels=4 * 28 * 28,
            max_pixels=4000 * 28 * 28
        )


    def load_image(self, image_path):
        """Load and process image with error handling"""
        if isinstance(image_path, str):
            return Image.open(image_path).convert('RGB')
        else:
            return image_path.convert('RGB')

    def infer_cot_score(self, image, question ):
        
        SYSTEM_PROMPT = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        pixel_values = self.load_image(image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[pixel_values],
            padding=True,
            return_tensors="pt",
            min_pixels=4 * 28 * 28,
            max_pixels=4000 * 28 * 28
        )

        inputs = inputs.to(self.device)

        with torch.no_grad():
            generation_model = self._get_model_for_generation()

            with timeout_handler(120):
                output = generation_model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                generated_ids = output.sequences[:, inputs['input_ids'].shape[1]:]
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]

                def extract_answer_with_tags(text):
                    import re
                    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    return None

                score = extract_answer_with_tags(generated_text)

                try:
                    if score is None:
                        score = generated_text.split("<answer>")[-1]
                    score = float(score)
                except (ValueError, TypeError):
                    score = -1

        return {
            "value_logit": score,
            "reason": generated_text,
        }

    def score_image_dimension(self, image_path, scoring_prompt):
        """Score a single image for a specific dimension"""
        try:
            result, think = self.infer_cot_score(image_path, scoring_prompt)
            score = result["value_logit"]
            return score
        except Exception as e:
            print(f"Error scoring image {image_path}: {e}")
            return 0.0

    def score_image_multi_dimension(self, prompt, image_path):
        """Score a single image on three dimensions"""
        alignment_prompt_template = 'Judge the image alignment with the prompt: "{}"\nPlease evaluate how well the image matches each element of provided prompt.\n\nAnd answer with the final alignment rating.\nRate it from 0 to 5 (float, 2 decimals). A rating of 0 represents very poor alignment level, while 5 represents excellent alignment level.'
        technical_prompt = 'Give a technical quality score for this picture between 0 and 5 (float, two decimal places). A rating of 0 represents very poor quality, while 5 represents excellent quality.'
        aesthetic_prompt = 'Provide a float rating between 0 and 5 for the overall aesthetics of this image, rounded to two places. A rating of 0 represents very poor aesthetic quality, while 5 represents excellent aesthetic quality.'

        alignment_prompt = alignment_prompt_template.format(prompt)

        technical_score = self.score_image_dimension(image_path, technical_prompt)
        aesthetic_score = self.score_image_dimension(image_path, aesthetic_prompt)
        alignment_score = self.score_image_dimension(image_path, aesthetic_prompt)

        return {
            'technical': technical_score,
            'aesthetic': aesthetic_score,
            'alignment': alignment_score,
        }
```

## Citation

If you find OmniQuality-R useful for your research, please cite:

```bibtex
@article{omniquality2024,
  title={OmniQuality-R: Advancing Reward Models through All-Encompassing Quality Assessment},
  author={[Authors]},
  year={2025}
}
```


We thank the OpenRLHF team and lmm-R1 for providing the excellent RLHF infrastructure and the research community for their valuable contributions to multimodal reasoning and reward modeling.


