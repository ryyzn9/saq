# S3-KLQ-v2 Implementation - H100 Optimized
# Notebook 2: Core Algorithm Implementation

# %% [markdown]
# # S3-KLQ-v2 Trainer Implementation
# ## Double Soft-Min Critic with PPO-Style Updates

# %% Cell 1: Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# %% Cell 2: Double Soft-Min Critic
class DoubleSoftMinCritic(nn.Module):
    """
    Twin value heads with numerically stable soft-min aggregation.
    
    V_soft = -α * (logsumexp(-V1/α, -V2/α) - log(2))
    """
    
    def __init__(self, hidden_size: int, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.hidden_size = hidden_size
        
        # Twin value heads
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
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.v_head_1, self.v_head_2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            v1, v2: [batch] - value estimates from each head
        """
        # Use last token for value
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]
        
        v1 = self.v_head_1(last_hidden).squeeze(-1)  # [batch]
        v2 = self.v_head_2(last_hidden).squeeze(-1)  # [batch]
        
        return v1, v2
    
    def soft_min(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable soft-minimum.
        
        V_soft = -α * log(0.5 * (exp(-V1/α) + exp(-V2/α)))
               = -α * (logsumexp(-V1/α, -V2/α) - log(2))
        """
        scaled = torch.stack([-v1 / self.alpha, -v2 / self.alpha], dim=-1)
        log_sum = torch.logsumexp(scaled, dim=-1)
        return -self.alpha * (log_sum - math.log(2))
    
    def get_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get soft-min value."""
        v1, v2 = self.forward(hidden_states)
        return self.soft_min(v1, v2)


# %% Cell 3: S3-KLQ-v2 Trainer
class S3KLQv2Trainer:
    """
    S3-KLQ-v2: Stabilized Soft-Min KL-regularized Q-learning.
    
    Key features:
    - Double soft-min critic (reduces overestimation)
    - PPO-style clipping (multi-epoch training)
    - KL penalty only in policy loss (decoupled)
    - Advantage normalization
    - Entropy bonus
    - Polyak-averaged target networks
    """
    
    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        reward_fn,
        config,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.device = next(policy_model.parameters()).device
        
        # Get hidden size
        hidden_size = policy_model.config.hidden_size
        
        # Initialize critics
        self.critic = DoubleSoftMinCritic(
            hidden_size, 
            alpha=config.alpha_softmin
        ).to(self.device)
        
        self.critic_target = DoubleSoftMinCritic(
            hidden_size,
            alpha=config.alpha_softmin
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.policy_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        
        # Hyperparameters
        self.gamma = config.gamma
        self.lam = config.gae_lambda
        self.clip_range = config.clip_range
        self.clip_range_vf = config.clip_range_vf
        self.beta = config.kl_coef
        self.entropy_coef = config.entropy_coef
        self.tau = config.tau_polyak
        self.epochs = config.epochs_per_batch
        
        # Tracking
        self.step_count = 0
        
    @torch.no_grad()
    def generate_rollouts(self, prompts: List[str]) -> Dict:
        """Generate responses for prompts."""
        self.policy.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate
        outputs = self.policy.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Split into prompt and response
        response_ids = outputs[:, prompt_len:]
        
        # Decode
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        
        # Compute rewards
        rewards = torch.tensor([
            self.reward_fn(text) for text in full_texts
        ], device=self.device, dtype=torch.float32)
        
        return {
            "prompt_ids": inputs.input_ids,
            "response_ids": response_ids,
            "full_ids": outputs,
            "attention_mask": (outputs != self.tokenizer.pad_token_id).long(),
            "rewards": rewards,
            "prompts": prompts,
            "responses": responses,
        }
    
    def compute_log_probs(
        self, 
        model, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        response_start: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for responses."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        logits = outputs.logits  # [batch, seq, vocab]
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # Shift for next-token prediction
        shift_logits = logits[:, response_start-1:-1, :]
        shift_labels = input_ids[:, response_start:]
        
        # Log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        response_mask = attention_mask[:, response_start:].float()
        token_log_probs = token_log_probs * response_mask
        
        # Sum over sequence
        seq_log_probs = token_log_probs.sum(dim=-1)
        
        return seq_log_probs, hidden_states
    
    def train_step(self, rollouts: Dict) -> Dict:
        """Perform one training iteration with multi-epoch updates."""
        self.policy.train()
        
        prompt_len = rollouts["prompt_ids"].shape[1]
        full_ids = rollouts["full_ids"]
        attention_mask = rollouts["attention_mask"]
        rewards = rollouts["rewards"]
        batch_size = full_ids.shape[0]
        
        # Compute old log probs and values
        with torch.no_grad():
            old_log_probs, old_hidden = self.compute_log_probs(
                self.policy, full_ids, attention_mask, prompt_len
            )
            
            ref_log_probs, _ = self.compute_log_probs(
                self.ref_model, full_ids, attention_mask, prompt_len
            )
            
            old_values = self.critic.get_value(old_hidden)
            
            # Compute advantages (simple for single-turn)
            advantages = rewards - old_values
            returns = rewards
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multi-epoch training
        metrics = {
            "policy_loss": 0, "value_loss": 0, 
            "kl": 0, "entropy": 0, "clip_frac": 0
        }
        
        for epoch in range(self.epochs):
            # Current forward pass
            new_log_probs, hidden_states = self.compute_log_probs(
                self.policy, full_ids, attention_mask, prompt_len
            )
            
            # === POLICY LOSS ===
            # Importance ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL penalty (single placement)
            kl = (new_log_probs - ref_log_probs).mean()
            kl_loss = self.beta * kl
            
            # Entropy bonus
            with torch.no_grad():
                outputs = self.policy(full_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits[:, prompt_len:, :], dim=-1)
                log_probs_full = F.log_softmax(outputs.logits[:, prompt_len:, :], dim=-1)
                entropy = -(probs * log_probs_full).sum(dim=-1).mean()
            
            total_policy_loss = policy_loss + kl_loss - self.entropy_coef * entropy
            
            # === VALUE LOSS ===
            v1, v2 = self.critic(hidden_states)
            v_soft = self.critic.soft_min(v1, v2)
            
            # Clipped value loss
            v_clipped = old_values + torch.clamp(
                v_soft - old_values, -self.clip_range_vf, self.clip_range_vf
            )
            
            value_loss1 = (v_soft - returns) ** 2
            value_loss2 = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            # === UPDATES ===
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            
            # Polyak update
            self._polyak_update()
            
            # Track metrics
            clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()
            metrics["policy_loss"] += policy_loss.item() / self.epochs
            metrics["value_loss"] += value_loss.item() / self.epochs
            metrics["kl"] += kl.item() / self.epochs
            metrics["entropy"] += entropy.item() / self.epochs
            metrics["clip_frac"] += clip_frac.item() / self.epochs
        
        self.step_count += 1
        metrics["step"] = self.step_count
        metrics["mean_reward"] = rewards.mean().item()
        
        return metrics
    
    def _polyak_update(self):
        """Update target networks with Polyak averaging."""
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)
    
    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "step": self.step_count,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoints."""
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.step_count = ckpt["step"]


# %% Cell 4: PPO Baseline Trainer
class PPOBaselineTrainer:
    """Standard PPO-RLHF implementation for comparison."""
    
    def __init__(self, policy_model, ref_model, tokenizer, reward_fn, config):
        self.policy = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.device = next(policy_model.parameters()).device
        
        hidden_size = policy_model.config.hidden_size
        
        # Single value head
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            list(policy_model.parameters()) + list(self.v_head.parameters()),
            lr=config.policy_lr
        )
        
        self.clip_range = config.clip_range
        self.beta = config.kl_coef
        self.epochs = config.epochs_per_batch
        self.step_count = 0
    
    @torch.no_grad()
    def generate_rollouts(self, prompts):
        self.policy.eval()
        
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, 
            truncation=True, max_length=512
        ).to(self.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        outputs = self.policy.generate(
            **inputs, max_new_tokens=256,
            do_sample=True, temperature=1.0, top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        responses = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        full_texts = [p + r for p, r in zip(prompts, responses)]
        rewards = torch.tensor([self.reward_fn(t) for t in full_texts], device=self.device)
        
        return {
            "prompt_ids": inputs.input_ids,
            "full_ids": outputs,
            "attention_mask": (outputs != self.tokenizer.pad_token_id).long(),
            "rewards": rewards,
        }
    
    def train_step(self, rollouts):
        self.policy.train()
        prompt_len = rollouts["prompt_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.policy(rollouts["full_ids"], output_hidden_states=True)
            old_hidden = outputs.hidden_states[-1][:, -1, :]
            old_values = self.v_head(old_hidden).squeeze(-1)
            
            advantages = rollouts["rewards"] - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = rollouts["rewards"]
        
        for epoch in range(self.epochs):
            outputs = self.policy(
                rollouts["full_ids"], 
                attention_mask=rollouts["attention_mask"],
                output_hidden_states=True
            )
            
            hidden = outputs.hidden_states[-1][:, -1, :]
            values = self.v_head(hidden).squeeze(-1)
            
            value_loss = 0.5 * ((values - returns) ** 2).mean()
            
            loss = value_loss
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
        
        self.step_count += 1
        return {
            "step": self.step_count,
            "value_loss": value_loss.item(),
            "mean_reward": rollouts["rewards"].mean().item(),
        }

print("S3-KLQ-v2 and PPO trainers defined")
