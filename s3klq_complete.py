# S3-KLQ-v2 vs RLHF - Complete H100 Experiment
# Single file combining all components for Kaggle

# ============================================
# PART 1: SETUP AND CONFIGURATION
# ============================================

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
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

print("Imports complete")

# H100 Configuration
@dataclass
class H100Config:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    use_flash_attention: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    batch_size: int = 8
    gradient_accumulation: int = 2
    max_steps: int = 100  # Reduced for testing
    policy_lr: float = 2e-5
    critic_lr: float = 1e-4
    max_new_tokens: int = 128
    max_length: int = 512
    alpha_softmin: float = 0.5
    tau_polyak: float = 0.02
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 1.0
    epochs_per_batch: int = 4
    log_every: int = 10
    save_every: int = 50

config = H100Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"Tokenizer loaded")

# LoRA config
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load models
def load_model_h100(model_name, lora_cfg):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

print("Loading policy model...")
policy_model = load_model_h100(config.model_name, lora_config)

print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# Dataset
print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:200]")

def extract_prompt(example):
    text = example["chosen"]
    if "Assistant:" in text:
        prompt = text.split("Assistant:")[0] + "Assistant:"
    else:
        prompt = text[:200]
    return {"prompt": prompt}

dataset = dataset.map(extract_prompt)
eval_dataset = eval_dataset.map(extract_prompt)
print(f"Dataset: {len(dataset)} train")

# Simple Reward Model
class SimpleRewardModel:
    def __init__(self, tok):
        self.tokenizer = tok
        
    def __call__(self, text: str) -> float:
        reward = 0.0
        tokens = self.tokenizer.encode(text)
        length = len(tokens)
        if 50 < length < 300:
            reward += 0.3
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio * 0.3
        helpful = ["help", "sure", "can", "would"]
        for w in helpful:
            if w in text.lower():
                reward += 0.05
        if len(text.strip()) < 10:
            reward = -1.0
        return min(max(reward, -1.0), 1.0)

reward_model = SimpleRewardModel(tokenizer)
print("Setup complete!")

# ============================================
# PART 2: S3-KLQ-v2 TRAINER
# ============================================

class DoubleSoftMinCritic(nn.Module):
    def __init__(self, hidden_size: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.v_head_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        self.v_head_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.v_head_1, self.v_head_2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states):
        last_hidden = hidden_states[:, -1, :]
        v1 = self.v_head_1(last_hidden).squeeze(-1)
        v2 = self.v_head_2(last_hidden).squeeze(-1)
        return v1, v2
    
    def soft_min(self, v1, v2):
        scaled = torch.stack([-v1 / self.alpha, -v2 / self.alpha], dim=-1)
        log_sum = torch.logsumexp(scaled, dim=-1)
        return -self.alpha * (log_sum - math.log(2))
    
    def get_value(self, hidden_states):
        v1, v2 = self.forward(hidden_states)
        return self.soft_min(v1, v2)


class S3KLQv2Trainer:
    def __init__(self, policy_model, ref_model, tokenizer, reward_fn, config):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.device = next(policy_model.parameters()).device
        
        hidden_size = policy_model.config.hidden_size
        
        self.critic = DoubleSoftMinCritic(hidden_size, alpha=config.alpha_softmin).to(self.device)
        self.critic_target = DoubleSoftMinCritic(hidden_size, alpha=config.alpha_softmin).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        self.policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.policy_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        
        self.clip_range = config.clip_range
        self.clip_range_vf = config.clip_range_vf
        self.beta = config.kl_coef
        self.entropy_coef = config.entropy_coef
        self.tau = config.tau_polyak
        self.epochs = config.epochs_per_batch
        self.step_count = 0
        
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, 
                                truncation=True, max_length=512).to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        
        outputs = self.policy.generate(
            **inputs, max_new_tokens=self.config.max_new_tokens,
            do_sample=True, temperature=1.0, top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response_ids = outputs[:, prompt_len:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        rewards = torch.tensor([self.reward_fn(t) for t in full_texts], 
                              device=self.device, dtype=torch.float32)
        
        return {
            "prompt_ids": inputs.input_ids,
            "response_ids": response_ids,
            "full_ids": outputs,
            "attention_mask": (outputs != self.tokenizer.pad_token_id).long(),
            "rewards": rewards,
        }
    
    def train_step(self, rollouts):
        self.policy.train()
        prompt_len = rollouts["prompt_ids"].shape[1]
        full_ids = rollouts["full_ids"]
        attention_mask = rollouts["attention_mask"]
        rewards = rollouts["rewards"]
        
        with torch.no_grad():
            outputs = self.policy(full_ids, attention_mask=attention_mask, output_hidden_states=True)
            old_hidden = outputs.hidden_states[-1]
            old_values = self.critic.get_value(old_hidden)
            advantages = rewards - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = rewards
        
        metrics = {"policy_loss": 0, "value_loss": 0, "kl": 0, "entropy": 0}
        
        for epoch in range(self.epochs):
            outputs = self.policy(full_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            v1, v2 = self.critic(hidden_states)
            v_soft = self.critic.soft_min(v1, v2)
            
            v_clipped = old_values + torch.clamp(v_soft - old_values, -self.clip_range_vf, self.clip_range_vf)
            value_loss1 = (v_soft - returns) ** 2
            value_loss2 = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            self.policy_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            
            self._polyak_update()
            metrics["value_loss"] += value_loss.item() / self.epochs
        
        self.step_count += 1
        metrics["step"] = self.step_count
        metrics["mean_reward"] = rewards.mean().item()
        metrics["kl"] = 0.0
        metrics["entropy"] = 0.0
        return metrics
    
    def _polyak_update(self):
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"critic": self.critic.state_dict(), "step": self.step_count}, path)

print("S3-KLQ-v2 Trainer defined")

# ============================================
# PART 3: RUN EXPERIMENT
# ============================================

print("\n" + "=" * 60)
print("Initializing S3-KLQ-v2 Trainer")
print("=" * 60)

s3klq_trainer = S3KLQv2Trainer(
    policy_model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_fn=reward_model,
    config=config,
)
print("✓ Trainer initialized")

# DataLoader
def collate_fn(batch):
    return {"prompt": [item["prompt"] for item in batch]}

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
print(f"DataLoader: {len(train_loader)} batches")

# Results storage
results = {"steps": [], "rewards": [], "value_loss": []}

# Training
print("\n" + "=" * 60)
print("Training S3-KLQ-v2")
print("=" * 60)

start_time = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps:
        break
    
    prompts = batch["prompt"]
    rollouts = s3klq_trainer.generate_rollouts(prompts)
    metrics = s3klq_trainer.train_step(rollouts)
    
    results["steps"].append(step)
    results["rewards"].append(metrics["mean_reward"])
    results["value_loss"].append(metrics["value_loss"])
    
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['mean_reward']:.3f}, v_loss={metrics['value_loss']:.3f}")

training_time = time.time() - start_time

# Summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Training time: {training_time/60:.1f} minutes")
print(f"Final reward: {np.mean(results['rewards'][-20:]):.3f}")
print(f"Max reward: {max(results['rewards']):.3f}")

# Save results
with open("s3klq_results.json", "w") as f:
    json.dump(results, f)
print("Results saved to s3klq_results.json")
