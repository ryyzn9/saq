# S3-KLQ-v2 vs RLHF - Run Experiments
# Notebook 3: Training Loop (H100 Optimized)

# %% [markdown]
# # Run S3-KLQ-v2 vs PPO-RLHF Experiment
# ## Main Training Loop

# %% Cell 1: Run Setup Script
# %run 01_setup_h100.py

# %% Cell 2: Run Trainer Script  
# %run 02_s3klq_trainer.py

# %% Cell 3: Initialize Trainers
print("=" * 60)
print("Initializing Trainers")
print("=" * 60)

# S3-KLQ-v2 Trainer
s3klq_trainer = S3KLQv2Trainer(
    policy_model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_fn=reward_model,
    config=config,
)
print("✓ S3-KLQ-v2 trainer initialized")

# For PPO, we need a separate model copy
from copy import deepcopy

ppo_policy = load_model_h100(config.model_name, lora_config)
ppo_trainer = PPOBaselineTrainer(
    policy_model=ppo_policy,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_fn=reward_model,
    config=config,
)
print("✓ PPO baseline trainer initialized")

# %% Cell 4: Initialize Logging
import wandb
from datetime import datetime

# Optional: Initialize W&B
USE_WANDB = False  # Set True if you have W&B account

if USE_WANDB:
    wandb.init(
        project="s3klq-vs-rlhf",
        name=f"h100-qwen-{datetime.now().strftime('%Y%m%d_%H%M')}",
        config={
            "model": config.model_name,
            "batch_size": config.batch_size,
            "alpha_softmin": config.alpha_softmin,
            "kl_coef": config.kl_coef,
        }
    )

# Results storage
results = {
    "s3klq": {"steps": [], "rewards": [], "kl": [], "value_loss": [], "entropy": []},
    "ppo": {"steps": [], "rewards": [], "value_loss": []},
}

# %% Cell 5: Create DataLoader
from torch.utils.data import DataLoader

def collate_fn(batch):
    return {"prompt": [item["prompt"] for item in batch]}

train_loader = DataLoader(
    dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

print(f"DataLoader: {len(train_loader)} batches of size {config.batch_size}")

# %% Cell 6: Training Loop - S3-KLQ-v2
print("\n" + "=" * 60)
print("Training S3-KLQ-v2")
print("=" * 60)

from tqdm.auto import tqdm
import time

s3klq_start = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps:
        break
    
    prompts = batch["prompt"]
    
    # Generate rollouts
    rollouts = s3klq_trainer.generate_rollouts(prompts)
    
    # Train step
    metrics = s3klq_trainer.train_step(rollouts)
    
    # Log
    results["s3klq"]["steps"].append(step)
    results["s3klq"]["rewards"].append(metrics["mean_reward"])
    results["s3klq"]["kl"].append(metrics["kl"])
    results["s3klq"]["value_loss"].append(metrics["value_loss"])
    results["s3klq"]["entropy"].append(metrics["entropy"])
    
    if USE_WANDB:
        wandb.log({f"s3klq/{k}": v for k, v in metrics.items()})
    
    # Print progress
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['mean_reward']:.3f}, "
              f"kl={metrics['kl']:.3f}, v_loss={metrics['value_loss']:.3f}")
    
    # Save checkpoint
    if step > 0 and step % config.save_every == 0:
        s3klq_trainer.save(f"checkpoints/s3klq_step_{step}.pt")

s3klq_time = time.time() - s3klq_start
print(f"\nS3-KLQ-v2 training completed in {s3klq_time/60:.1f} minutes")

# %% Cell 7: Training Loop - PPO Baseline
print("\n" + "=" * 60)
print("Training PPO Baseline")
print("=" * 60)

# Reset dataloader
train_loader = DataLoader(
    dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

ppo_start = time.time()

for step, batch in enumerate(tqdm(train_loader, total=config.max_steps)):
    if step >= config.max_steps:
        break
    
    prompts = batch["prompt"]
    
    # Generate rollouts
    rollouts = ppo_trainer.generate_rollouts(prompts)
    
    # Train step
    metrics = ppo_trainer.train_step(rollouts)
    
    # Log
    results["ppo"]["steps"].append(step)
    results["ppo"]["rewards"].append(metrics["mean_reward"])
    results["ppo"]["value_loss"].append(metrics["value_loss"])
    
    if USE_WANDB:
        wandb.log({f"ppo/{k}": v for k, v in metrics.items()})
    
    if step % config.log_every == 0:
        print(f"Step {step}: reward={metrics['mean_reward']:.3f}, "
              f"v_loss={metrics['value_loss']:.3f}")

ppo_time = time.time() - ppo_start
print(f"\nPPO training completed in {ppo_time/60:.1f} minutes")

# %% Cell 8: Save Results
import json

# Save metrics
with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary statistics
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

s3klq_rewards = results["s3klq"]["rewards"]
ppo_rewards = results["ppo"]["rewards"]

print(f"\nS3-KLQ-v2:")
print(f"  Final reward: {np.mean(s3klq_rewards[-50:]):.3f}")
print(f"  Max reward: {max(s3klq_rewards):.3f}")
print(f"  Final KL: {np.mean(results['s3klq']['kl'][-50:]):.3f}")
print(f"  Training time: {s3klq_time/60:.1f} min")

print(f"\nPPO Baseline:")
print(f"  Final reward: {np.mean(ppo_rewards[-50:]):.3f}")
print(f"  Max reward: {max(ppo_rewards):.3f}")
print(f"  Training time: {ppo_time/60:.1f} min")

# %% Cell 9: Evaluation
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

def evaluate(trainer, prompts, n_samples=100):
    """Evaluate model on test prompts."""
    rewards = []
    
    for i in range(0, n_samples, config.batch_size):
        batch_prompts = prompts[i:i+config.batch_size]
        if len(batch_prompts) == 0:
            break
        
        rollouts = trainer.generate_rollouts(batch_prompts)
        rewards.extend(rollouts["rewards"].cpu().tolist())
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": max(rewards),
        "min_reward": min(rewards),
    }

# Get test prompts
test_prompts = [eval_dataset[i]["prompt"] for i in range(100)]

s3klq_eval = evaluate(s3klq_trainer, test_prompts)
ppo_eval = evaluate(ppo_trainer, test_prompts)

print(f"\nS3-KLQ-v2 Eval: {s3klq_eval['mean_reward']:.3f} ± {s3klq_eval['std_reward']:.3f}")
print(f"PPO Eval: {ppo_eval['mean_reward']:.3f} ± {ppo_eval['std_reward']:.3f}")

# %% Cell 10: Save Final Checkpoint
s3klq_trainer.save("checkpoints/s3klq_final.pt")
torch.save(ppo_trainer.policy.state_dict(), "checkpoints/ppo_final.pt")
print("\n✓ Final checkpoints saved")

# Close wandb
if USE_WANDB:
    wandb.finish()

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
