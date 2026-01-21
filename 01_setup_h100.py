# S3-KLQ-v2 vs RLHF Experiment - H100 Optimized
# Notebook 1: Setup and Configuration

"""
# H100 GPU Configuration
- GPU: NVIDIA H100 80GB
- Precision: BF16 with FP8 optional
- Flash Attention 2 enabled
- Large batch sizes for efficiency
"""

# %% [markdown]
# # S3-KLQ-v2 vs PPO-RLHF Comparison
# ## H100 Optimized Implementation on Qwen3-1.5B

# %% Cell 1: Install Dependencies
# !pip install -q torch>=2.2.0 transformers>=4.40.0 accelerate>=0.28.0
# !pip install -q peft>=0.10.0 trl>=0.8.0 datasets>=2.18.0 wandb
# !pip install -q bitsandbytes flash-attn --no-build-isolation

# %% Cell 2: Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb

# %% Cell 3: H100 Configuration
@dataclass
class H100Config:
    """H100-optimized configuration."""
    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B"
    use_flash_attention: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    
    # LoRA
    lora_r: int = 32  # Higher rank for H100
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Training - H100 can handle larger batches
    batch_size: int = 16
    gradient_accumulation: int = 2
    effective_batch_size: int = 32
    max_steps: int = 500
    
    # Learning rates
    policy_lr: float = 2e-5
    critic_lr: float = 1e-4
    
    # Generation
    max_new_tokens: int = 256
    max_length: int = 1024
    
    # S3-KLQ-v2 specific
    alpha_softmin: float = 0.5
    tau_polyak: float = 0.02
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 1.0
    epochs_per_batch: int = 4
    
    # Logging
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100

config = H100Config()
print(f"Using config: {config}")

# %% Cell 4: Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% Cell 5: Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# %% Cell 6: LoRA Configuration
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")

# %% Cell 7: Load Model
def load_model_h100(model_name, lora_config):
    """Load model optimized for H100."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# Load policy model
policy_model = load_model_h100(config.model_name, lora_config)
print("Policy model loaded")

# Load reference model (frozen)
ref_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if config.use_flash_attention else "eager",
    device_map="auto",
    trust_remote_code=True,
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
print("Reference model loaded (frozen)")

# %% Cell 8: Load Dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:500]")

def extract_prompt(example):
    """Extract prompt from HH-RLHF format."""
    text = example["chosen"]
    # Split at first "Assistant:" to get prompt
    if "Assistant:" in text:
        prompt = text.split("Assistant:")[0] + "Assistant:"
    else:
        prompt = text[:200]
    return {"prompt": prompt}

dataset = dataset.map(extract_prompt)
eval_dataset = eval_dataset.map(extract_prompt)
print(f"Dataset: {len(dataset)} train, {len(eval_dataset)} eval")

# %% Cell 9: Reward Model (Simple)
class SimpleRewardModel:
    """Simple reward based on response quality heuristics."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, text: str) -> float:
        """Compute reward for a response."""
        # Simple heuristics (replace with real RM in production)
        reward = 0.0
        
        # Length reward (prefer medium length)
        tokens = self.tokenizer.encode(text)
        length = len(tokens)
        if 50 < length < 300:
            reward += 0.3
        elif length > 300:
            reward += 0.1
        
        # Coherence (no repetition)
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio * 0.3
        
        # Helpfulness keywords
        helpful_words = ["help", "here", "sure", "can", "would", "please"]
        for word in helpful_words:
            if word in text.lower():
                reward += 0.05
        
        # Penalize empty or very short
        if len(text.strip()) < 10:
            reward = -1.0
        
        return min(max(reward, -1.0), 1.0)

reward_model = SimpleRewardModel(tokenizer)
print("Reward model initialized")

# Test reward
test_response = "I'd be happy to help you with that question. Here's what I think..."
print(f"Test reward: {reward_model(test_response):.3f}")

# %% Cell 10: Save Config
config_dict = {
    "model_name": config.model_name,
    "batch_size": config.batch_size,
    "max_steps": config.max_steps,
    "alpha_softmin": config.alpha_softmin,
    "kl_coef": config.kl_coef,
}
with open("experiment_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
print("Config saved")
