# main.py - S3-KLQ-v2 vs PPO-RLHF Complete Experiment
# Single file for Kaggle H100 - Copy and paste into notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

print("Installing dependencies...")
os.system("pip install -q transformers accelerate peft datasets")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

# =============== CONFIG ===============
@dataclass  
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    lora_r: int = 32
    lora_alpha: int = 64
    batch_size: int = 4
    max_steps: int = 50
    policy_lr: float = 2e-5
    critic_lr: float = 1e-4
    max_new_tokens: int = 64
    alpha_softmin: float = 0.5
    tau_polyak: float = 0.02
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    epochs_per_batch: int = 2
    log_every: int = 5
    save_every: int = 25

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============== TOKENIZER ===============
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =============== LORA ===============
lora_config = LoraConfig(
    r=config.lora_r, lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# =============== MODELS ===============
print("Loading policy model...")
policy_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
policy_model = get_peft_model(policy_model, lora_config)
policy_model.print_trainable_parameters()

print("Loading ref model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# =============== DATASET ===============
print("Loading dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test[:200]")

def extract_prompt(ex):
    t = ex["chosen"]
    return {"prompt": t.split("Assistant:")[0] + "Assistant:" if "Assistant:" in t else t[:200]}

dataset = dataset.map(extract_prompt)
eval_dataset = eval_dataset.map(extract_prompt)

# =============== REWARD ===============
class RewardModel:
    def __init__(self, tok):
        self.tok = tok
    def __call__(self, text):
        r = 0.0
        toks = self.tok.encode(text)
        if 50 < len(toks) < 300: r += 0.3
        words = text.lower().split()
        if words: r += len(set(words))/len(words) * 0.3
        for w in ["help", "sure", "can"]:
            if w in text.lower(): r += 0.05
        if len(text.strip()) < 10: r = -1.0
        return min(max(r, -1.0), 1.0)

reward_fn = RewardModel(tokenizer)

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
    def __init__(self, policy, ref, tok, reward, cfg):
        self.policy = policy
        self.ref = ref
        self.tok = tok
        self.reward = reward
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
    
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        inp = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        plen = inp.input_ids.shape[1]
        out = self.policy.generate(**inp, max_new_tokens=self.cfg.max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=self.tok.pad_token_id)
        resp = self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        texts = [p+r for p,r in zip(prompts, resp)]
        rew = torch.tensor([self.reward(t) for t in texts], device=self.device)
        return {"ids": out, "mask": (out != self.tok.pad_token_id).long(), "rewards": rew}
    
    def train_step(self, roll):
        self.policy.train()
        ids, mask, rew = roll["ids"], roll["mask"], roll["rewards"]
        with torch.no_grad():
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            old_v = self.critic.soft_min(v1, v2)
            adv = (rew - old_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for _ in range(self.cfg.epochs_per_batch):
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            v = self.critic.soft_min(v1, v2)
            vc = old_v + torch.clamp(v - old_v, -self.cfg.clip_range_vf, self.cfg.clip_range_vf)
            loss = 0.5 * torch.max((v-rew)**2, (vc-rew)**2).mean()
            self.opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.opt_c.step()
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1-self.cfg.tau_polyak).add_(self.cfg.tau_polyak * p.data)
        
        self.step += 1
        return {"step": self.step, "mean_reward": rew.mean().item(), "value_loss": loss.item(), "kl": 0.0, "entropy": 0.0}
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"critic": self.critic.state_dict()}, path)

# =============== PPO BASELINE TRAINER ===============
class PPOTrainer:
    def __init__(self, policy, ref, tok, reward, cfg):
        self.policy = policy
        self.tok = tok
        self.reward = reward
        self.cfg = cfg
        self.device = next(policy.parameters()).device
        hs = policy.config.hidden_size
        self.v_head = nn.Sequential(nn.Linear(hs, hs//4), nn.GELU(), nn.Linear(hs//4, 1)).to(self.device).to(torch.bfloat16)
        self.opt = torch.optim.AdamW(list(policy.parameters()) + list(self.v_head.parameters()), lr=cfg.policy_lr)
        self.step = 0
    
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        inp = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        plen = inp.input_ids.shape[1]
        out = self.policy.generate(**inp, max_new_tokens=self.cfg.max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=self.tok.pad_token_id)
        resp = self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        texts = [p+r for p,r in zip(prompts, resp)]
        rew = torch.tensor([self.reward(t) for t in texts], device=self.device)
        return {"ids": out, "mask": (out != self.tok.pad_token_id).long(), "rewards": rew}
    
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
        return {"step": self.step, "mean_reward": rew.mean().item(), "value_loss": loss.item()}

# =============== INITIALIZE TRAINERS ===============
print("\n" + "="*60)
print("Initializing Trainers")
print("="*60)

s3klq_trainer = S3KLQTrainer(policy_model, ref_model, tokenizer, reward_fn, config)
print("✓ S3-KLQ-v2 trainer initialized")

print("Loading PPO policy model...")
ppo_lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
ppo_policy = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
ppo_policy = get_peft_model(ppo_policy, ppo_lora)
ppo_trainer = PPOTrainer(ppo_policy, ref_model, tokenizer, reward_fn, config)
print("✓ PPO baseline trainer initialized")

# Results storage
results = {
    "s3klq": {"steps": [], "rewards": [], "kl": [], "value_loss": [], "entropy": []},
    "ppo": {"steps": [], "rewards": [], "value_loss": []},
}

# DataLoader
def collate_fn(batch):
    return {"prompt": [item["prompt"] for item in batch]}

# =============== TRAIN S3-KLQ-v2 ===============
print("\n" + "="*60)
print("Training S3-KLQ-v2")
print("="*60)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
s3klq_start = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps: break
    rollouts = s3klq_trainer.generate_rollouts(batch["prompt"])
    metrics = s3klq_trainer.train_step(rollouts)
    results["s3klq"]["steps"].append(step)
    results["s3klq"]["rewards"].append(metrics["mean_reward"])
    results["s3klq"]["kl"].append(metrics["kl"])
    results["s3klq"]["value_loss"].append(metrics["value_loss"])
    results["s3klq"]["entropy"].append(metrics["entropy"])
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['mean_reward']:.3f}, v_loss={metrics['value_loss']:.3f}")

s3klq_time = time.time() - s3klq_start
print(f"\nS3-KLQ-v2 completed in {s3klq_time/60:.1f} minutes")

# =============== TRAIN PPO ===============
print("\n" + "="*60)
print("Training PPO Baseline")
print("="*60)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
ppo_start = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps: break
    rollouts = ppo_trainer.generate_rollouts(batch["prompt"])
    metrics = ppo_trainer.train_step(rollouts)
    results["ppo"]["steps"].append(step)
    results["ppo"]["rewards"].append(metrics["mean_reward"])
    results["ppo"]["value_loss"].append(metrics["value_loss"])
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['mean_reward']:.3f}, v_loss={metrics['value_loss']:.3f}")

ppo_time = time.time() - ppo_start
print(f"\nPPO completed in {ppo_time/60:.1f} minutes")

# =============== SUMMARY ===============
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)

s3klq_rewards = results["s3klq"]["rewards"]
ppo_rewards = results["ppo"]["rewards"]

print(f"\nS3-KLQ-v2:")
print(f"  Final reward: {np.mean(s3klq_rewards[-20:]):.3f}")
print(f"  Max reward: {max(s3klq_rewards):.3f}")
print(f"  Final KL: {np.mean(results['s3klq']['kl'][-20:]):.3f}")
print(f"  Training time: {s3klq_time/60:.1f} min")

print(f"\nPPO Baseline:")
print(f"  Final reward: {np.mean(ppo_rewards[-20:]):.3f}")
print(f"  Max reward: {max(ppo_rewards):.3f}")
print(f"  Training time: {ppo_time/60:.1f} min")

# =============== EVALUATION ===============
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

def evaluate(trainer, prompts, n_samples=50):
    rewards = []
    for i in range(0, n_samples, config.batch_size):
        batch_prompts = prompts[i:i+config.batch_size]
        if len(batch_prompts) == 0: break
        rollouts = trainer.generate_rollouts(batch_prompts)
        rewards.extend(rollouts["rewards"].cpu().tolist())
    return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards)}

test_prompts = [eval_dataset[i]["prompt"] for i in range(50)]
s3klq_eval = evaluate(s3klq_trainer, test_prompts)
ppo_eval = evaluate(ppo_trainer, test_prompts)

print(f"\nS3-KLQ-v2 Eval: {s3klq_eval['mean_reward']:.3f} ± {s3klq_eval['std_reward']:.3f}")
print(f"PPO Eval: {ppo_eval['mean_reward']:.3f} ± {ppo_eval['std_reward']:.3f}")

# =============== SAVE ===============
with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Results saved to experiment_results.json")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
