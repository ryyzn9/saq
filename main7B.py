# main7B.py - S3-KLQ-v2 vs PPO-RLHF at 7B Scale
# 500+ steps, Real Reward Model, Proper KL Tracking, Plotting

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

print("Installing dependencies...")
os.system("pip install -q transformers accelerate peft datasets matplotlib")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

# =============== CONFIG ===============
@dataclass  
class Config:
    # Model - 7B scale
    model_name: str = "Qwen/Qwen2.5-7B"
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    
    # Training
    batch_size: int = 2
    gradient_accumulation: int = 4
    max_steps: int = 500
    policy_lr: float = 1e-5
    critic_lr: float = 5e-5
    max_new_tokens: int = 128
    
    # S3-KLQ-v2 specific
    alpha_softmin: float = 0.5
    tau_polyak: float = 0.02
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    epochs_per_batch: int = 2
    target_kl: float = 0.1
    
    # Logging
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 50

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============== TOKENIZER ===============
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =============== REWARD MODEL (Real) ===============
print("Loading reward model...")
reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    config.reward_model_name, torch_dtype=torch.float16, device_map="auto"
)
reward_model.eval()

def compute_reward(text: str) -> float:
    """Compute reward using actual reward model."""
    inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = reward_model(**inputs)
    return outputs.logits[0, 0].item()

print("Reward model loaded")

# =============== LORA ===============
lora_config = LoraConfig(
    r=config.lora_r, lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# =============== MODELS ===============
print("Loading 7B policy model...")
policy_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
policy_model = get_peft_model(policy_model, lora_config)
policy_model.print_trainable_parameters()

print("Loading 7B ref model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# =============== DATASET ===============
print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:500]")

def extract_prompt(ex):
    t = ex["chosen"]
    return {"prompt": t.split("Assistant:")[0] + "Assistant:" if "Assistant:" in t else t[:300]}

dataset = dataset.map(extract_prompt)
eval_dataset = eval_dataset.map(extract_prompt)
print(f"Dataset: {len(dataset)} train, {len(eval_dataset)} eval")

# =============== DOUBLE SOFT-MIN CRITIC ===============
class DoubleCritic(nn.Module):
    def __init__(self, hidden_size, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.v1 = nn.Sequential(nn.Linear(hidden_size, hidden_size//4), nn.GELU(), nn.Linear(hidden_size//4, 1))
        self.v2 = nn.Sequential(nn.Linear(hidden_size, hidden_size//4), nn.GELU(), nn.Linear(hidden_size//4, 1))
    
    def forward(self, h):
        return self.v1(h[:,-1,:]).squeeze(-1), self.v2(h[:,-1,:]).squeeze(-1)
    
    def soft_min(self, v1, v2):
        s = torch.stack([-v1/self.alpha, -v2/self.alpha], -1)
        return -self.alpha * (torch.logsumexp(s, -1) - math.log(2))

# =============== S3-KLQ-v2 TRAINER ===============
class S3KLQTrainer:
    def __init__(self, policy, ref, tok, reward_fn, cfg):
        self.policy = policy
        self.ref = ref
        self.tok = tok
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.device = next(policy.parameters()).device
        hs = policy.config.hidden_size
        self.critic = DoubleCritic(hs, cfg.alpha_softmin).to(self.device).to(torch.bfloat16)
        self.critic_target = DoubleCritic(hs, cfg.alpha_softmin).to(self.device).to(torch.bfloat16)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters(): p.requires_grad = False
        self.opt_p = torch.optim.AdamW(policy.parameters(), lr=cfg.policy_lr)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.step = 0
        self.beta = cfg.kl_coef  # Adaptive KL coefficient
    
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        inp = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        plen = inp.input_ids.shape[1]
        out = self.policy.generate(**inp, max_new_tokens=self.cfg.max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=self.tok.pad_token_id)
        resp = self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        texts = [p+r for p,r in zip(prompts, resp)]
        rew = torch.tensor([self.reward_fn(t) for t in texts], device=self.device, dtype=torch.float32)
        comp_len = (out[:, plen:] != self.tok.pad_token_id).sum(dim=1).float().mean().item()
        
        # Compute log probs for KL
        policy_out = self.policy(out, attention_mask=(out != self.tok.pad_token_id).long())
        ref_out = self.ref(out, attention_mask=(out != self.tok.pad_token_id).long())
        
        policy_logits = policy_out.logits[:, plen-1:-1, :]
        ref_logits = ref_out.logits[:, plen-1:-1, :]
        
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        tokens = out[:, plen:]
        policy_token_logprobs = torch.gather(policy_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_token_logprobs = torch.gather(ref_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        
        mask = (tokens != self.tok.pad_token_id).float()
        kl_per_token = (policy_token_logprobs - ref_token_logprobs) * mask
        kl = kl_per_token.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return {
            "ids": out, 
            "mask": (out != self.tok.pad_token_id).long(), 
            "rewards": rew,
            "kl": kl.mean().item(),
            "completion_length": comp_len,
            "policy_logprobs": policy_token_logprobs,
            "ref_logprobs": ref_token_logprobs,
        }
    
    def train_step(self, roll):
        self.policy.train()
        ids, mask, rew = roll["ids"], roll["mask"], roll["rewards"]
        kl_measured = roll["kl"]
        
        with torch.no_grad():
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            old_v = self.critic.soft_min(v1, v2)
            adv = (rew - old_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.cfg.epochs_per_batch):
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            v = self.critic.soft_min(v1, v2)
            vc = old_v + torch.clamp(v - old_v, -self.cfg.clip_range_vf, self.cfg.clip_range_vf)
            loss = 0.5 * torch.max((v-rew)**2, (vc-rew)**2).mean()
            total_loss += loss.item()
            
            self.opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.opt_c.step()
            
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1-self.cfg.tau_polyak).add_(self.cfg.tau_polyak * p.data)
        
        # Adaptive KL
        if kl_measured > 1.5 * self.cfg.target_kl:
            self.beta = min(self.beta * 1.5, 10.0)
        elif kl_measured < 0.5 * self.cfg.target_kl:
            self.beta = max(self.beta * 0.5, 0.01)
        
        self.step += 1
        return {
            "step": self.step, 
            "reward": rew.mean().item(),
            "reward_std": rew.std().item(),
            "value_loss": total_loss / self.cfg.epochs_per_batch,
            "kl": kl_measured,
            "completion_length": roll["completion_length"],
            "beta": self.beta,
        }

# =============== PPO TRAINER ===============
class PPOTrainer:
    def __init__(self, policy, ref, tok, reward_fn, cfg):
        self.policy = policy
        self.ref = ref
        self.tok = tok
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.device = next(policy.parameters()).device
        hs = policy.config.hidden_size
        self.v_head = nn.Sequential(nn.Linear(hs, hs//4), nn.GELU(), nn.Linear(hs//4, 1)).to(self.device).to(torch.bfloat16)
        self.opt = torch.optim.AdamW(list(policy.parameters()) + list(self.v_head.parameters()), lr=cfg.policy_lr)
        self.step = 0
    
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        inp = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        plen = inp.input_ids.shape[1]
        out = self.policy.generate(**inp, max_new_tokens=self.cfg.max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=self.tok.pad_token_id)
        resp = self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        texts = [p+r for p,r in zip(prompts, resp)]
        rew = torch.tensor([self.reward_fn(t) for t in texts], device=self.device, dtype=torch.float32)
        comp_len = (out[:, plen:] != self.tok.pad_token_id).sum(dim=1).float().mean().item()
        
        # KL computation
        policy_out = self.policy(out, attention_mask=(out != self.tok.pad_token_id).long())
        ref_out = self.ref(out, attention_mask=(out != self.tok.pad_token_id).long())
        policy_logits = policy_out.logits[:, plen-1:-1, :]
        ref_logits = ref_out.logits[:, plen-1:-1, :]
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        tokens = out[:, plen:]
        policy_lp = torch.gather(policy_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ref_lp = torch.gather(ref_logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        mask = (tokens != self.tok.pad_token_id).float()
        kl = ((policy_lp - ref_lp) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return {"ids": out, "mask": (out != self.tok.pad_token_id).long(), "rewards": rew, "kl": kl.mean().item(), "completion_length": comp_len}
    
    def train_step(self, roll):
        self.policy.train()
        ids, mask, rew = roll["ids"], roll["mask"], roll["rewards"]
        with torch.no_grad():
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            old_v = self.v_head(h[:,-1,:]).squeeze(-1)
        
        for _ in range(self.cfg.epochs_per_batch):
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v = self.v_head(h[:,-1,:]).squeeze(-1)
            loss = 0.5 * ((v - rew)**2).mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
        
        self.step += 1
        return {"step": self.step, "reward": rew.mean().item(), "reward_std": rew.std().item(), 
                "value_loss": loss.item(), "kl": roll["kl"], "completion_length": roll["completion_length"]}

# =============== PLOTTING ===============
def plot_results(results, save_path="training_plots.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reward
    ax = axes[0, 0]
    ax.plot(results["s3klq"]["steps"], results["s3klq"]["rewards"], label="S3-KLQ-v2", alpha=0.7)
    ax.plot(results["ppo"]["steps"], results["ppo"]["rewards"], label="PPO", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.set_title("Reward"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # KL
    ax = axes[0, 1]
    ax.plot(results["s3klq"]["steps"], results["s3klq"]["kl"], label="S3-KLQ-v2", alpha=0.7)
    ax.plot(results["ppo"]["steps"], results["ppo"]["kl"], label="PPO", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("KL"); ax.set_title("KL Divergence"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # Value Loss
    ax = axes[0, 2]
    ax.plot(results["s3klq"]["steps"], results["s3klq"]["value_loss"], label="S3-KLQ-v2", alpha=0.7)
    ax.plot(results["ppo"]["steps"], results["ppo"]["value_loss"], label="PPO", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Value Loss"); ax.set_title("Value Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # Reward Std
    ax = axes[1, 0]
    ax.plot(results["s3klq"]["steps"], results["s3klq"]["reward_std"], label="S3-KLQ-v2", alpha=0.7)
    ax.plot(results["ppo"]["steps"], results["ppo"]["reward_std"], label="PPO", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward Std"); ax.set_title("Reward Variance"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # Completion Length
    ax = axes[1, 1]
    ax.plot(results["s3klq"]["steps"], results["s3klq"]["completion_length"], label="S3-KLQ-v2", alpha=0.7)
    ax.plot(results["ppo"]["steps"], results["ppo"]["completion_length"], label="PPO", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Tokens"); ax.set_title("Completion Length"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # Smoothed Reward
    ax = axes[1, 2]
    window = 20
    s3_smooth = np.convolve(results["s3klq"]["rewards"], np.ones(window)/window, mode='valid')
    ppo_smooth = np.convolve(results["ppo"]["rewards"], np.ones(window)/window, mode='valid')
    ax.plot(range(len(s3_smooth)), s3_smooth, label="S3-KLQ-v2", linewidth=2)
    ax.plot(range(len(ppo_smooth)), ppo_smooth, label="PPO", linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward (smoothed)"); ax.set_title("Smoothed Reward"); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plots saved to {save_path}")

# =============== INITIALIZE ===============
print("\n" + "="*60)
print("Initializing 7B Trainers")
print("="*60)

s3klq_trainer = S3KLQTrainer(policy_model, ref_model, tokenizer, compute_reward, config)
print("✓ S3-KLQ-v2 trainer initialized")

print("Loading PPO 7B model...")
ppo_lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
ppo_policy = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
ppo_policy = get_peft_model(ppo_policy, ppo_lora)
ppo_trainer = PPOTrainer(ppo_policy, ref_model, tokenizer, compute_reward, config)
print("✓ PPO trainer initialized")

results = {
    "s3klq": {"steps": [], "rewards": [], "reward_std": [], "kl": [], "value_loss": [], "completion_length": []},
    "ppo": {"steps": [], "rewards": [], "reward_std": [], "kl": [], "value_loss": [], "completion_length": []},
}

def collate_fn(batch):
    return {"prompt": [item["prompt"] for item in batch]}

# =============== TRAIN S3-KLQ-v2 ===============
print("\n" + "="*60)
print(f"Training S3-KLQ-v2 (7B) - {config.max_steps} steps")
print("="*60)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
s3klq_start = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps: break
    rollouts = s3klq_trainer.generate_rollouts(batch["prompt"])
    metrics = s3klq_trainer.train_step(rollouts)
    results["s3klq"]["steps"].append(step)
    results["s3klq"]["rewards"].append(metrics["reward"])
    results["s3klq"]["reward_std"].append(metrics["reward_std"])
    results["s3klq"]["kl"].append(metrics["kl"])
    results["s3klq"]["value_loss"].append(metrics["value_loss"])
    results["s3klq"]["completion_length"].append(metrics["completion_length"])
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['reward']:.3f}, kl={metrics['kl']:.4f}, v_loss={metrics['value_loss']:.3f}, len={metrics['completion_length']:.0f}")

s3klq_time = time.time() - s3klq_start

# =============== TRAIN PPO ===============
print("\n" + "="*60)
print(f"Training PPO (7B) - {config.max_steps} steps")
print("="*60)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
ppo_start = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps: break
    rollouts = ppo_trainer.generate_rollouts(batch["prompt"])
    metrics = ppo_trainer.train_step(rollouts)
    results["ppo"]["steps"].append(step)
    results["ppo"]["rewards"].append(metrics["reward"])
    results["ppo"]["reward_std"].append(metrics["reward_std"])
    results["ppo"]["kl"].append(metrics["kl"])
    results["ppo"]["value_loss"].append(metrics["value_loss"])
    results["ppo"]["completion_length"].append(metrics["completion_length"])
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['reward']:.3f}, kl={metrics['kl']:.4f}, v_loss={metrics['value_loss']:.3f}, len={metrics['completion_length']:.0f}")

ppo_time = time.time() - ppo_start

# =============== SUMMARY ===============
print("\n" + "="*60)
print("EXPERIMENT SUMMARY (7B)")
print("="*60)

print(f"\nS3-KLQ-v2:")
print(f"  Final reward: {np.mean(results['s3klq']['rewards'][-50:]):.3f} ± {np.mean(results['s3klq']['reward_std'][-50:]):.3f}")
print(f"  Max reward: {max(results['s3klq']['rewards']):.3f}")
print(f"  Final KL: {np.mean(results['s3klq']['kl'][-50:]):.4f}")
print(f"  Training time: {s3klq_time/60:.1f} min")

print(f"\nPPO:")
print(f"  Final reward: {np.mean(results['ppo']['rewards'][-50:]):.3f} ± {np.mean(results['ppo']['reward_std'][-50:]):.3f}")
print(f"  Max reward: {max(results['ppo']['rewards']):.3f}")
print(f"  Final KL: {np.mean(results['ppo']['kl'][-50:]):.4f}")
print(f"  Training time: {ppo_time/60:.1f} min")

# =============== PLOT ===============
plot_results(results, "training_plots_7B.png")

# =============== SAVE ===============
with open("results_7B.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Results saved to results_7B.json")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
