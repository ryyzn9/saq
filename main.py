# main.py - S3-KLQ-v2 Complete Experiment
# Single file for Kaggle H100

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
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# =============== MODEL ===============
print("Loading model...")
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

def extract_prompt(ex):
    t = ex["chosen"]
    return {"prompt": t.split("Assistant:")[0] + "Assistant:" if "Assistant:" in t else t[:200]}

dataset = dataset.map(extract_prompt)

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

# =============== CRITIC ===============
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

# =============== TRAINER ===============
class Trainer:
    def __init__(self):
        self.policy = policy_model
        self.ref = ref_model
        self.device = next(policy_model.parameters()).device
        hs = policy_model.config.hidden_size
        self.critic = DoubleCritic(hs, config.alpha_softmin).to(self.device)
        self.critic_target = DoubleCritic(hs, config.alpha_softmin).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters(): p.requires_grad = False
        self.opt_p = torch.optim.AdamW(policy_model.parameters(), lr=config.policy_lr)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        self.step = 0
    
    @torch.no_grad()
    def rollout(self, prompts):
        self.policy.eval()
        inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        plen = inp.input_ids.shape[1]
        out = self.policy.generate(**inp, max_new_tokens=config.max_new_tokens, do_sample=True, temperature=1.0, pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
        texts = [p+r for p,r in zip(prompts, resp)]
        rew = torch.tensor([reward_fn(t) for t in texts], device=self.device)
        return {"ids": out, "mask": (out != tokenizer.pad_token_id).long(), "rewards": rew}
    
    def train_step(self, roll):
        self.policy.train()
        ids, mask, rew = roll["ids"], roll["mask"], roll["rewards"]
        
        with torch.no_grad():
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            old_v = self.critic.soft_min(v1, v2)
            adv = (rew - old_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for _ in range(config.epochs_per_batch):
            h = self.policy(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
            v1, v2 = self.critic(h)
            v = self.critic.soft_min(v1, v2)
            vc = old_v + torch.clamp(v - old_v, -config.clip_range_vf, config.clip_range_vf)
            loss = 0.5 * torch.max((v-rew)**2, (vc-rew)**2).mean()
            
            self.opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.opt_c.step()
            
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1-config.tau_polyak).add_(config.tau_polyak * p.data)
        
        self.step += 1
        return {"step": self.step, "reward": rew.mean().item(), "v_loss": loss.item()}

# =============== RUN ===============
print("\n" + "="*50)
print("Starting S3-KLQ-v2 Training")
print("="*50)

trainer = Trainer()
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                    collate_fn=lambda b: {"prompt": [x["prompt"] for x in b]})

results = []
start = time.time()

for i, batch in enumerate(tqdm(loader, total=config.max_steps)):
    if i >= config.max_steps: break
    roll = trainer.rollout(batch["prompt"])
    m = trainer.train_step(roll)
    results.append(m)
    if i % config.log_every == 0:
        print(f"Step {i}: reward={m['reward']:.3f}, v_loss={m['v_loss']:.3f}")

print(f"\nDone in {(time.time()-start)/60:.1f} min")
print(f"Final reward: {np.mean([r['reward'] for r in results[-10:]]):.3f}")

with open("results.json", "w") as f:
    json.dump(results, f)
print("Saved results.json")
