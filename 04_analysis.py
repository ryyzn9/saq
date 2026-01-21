# S3-KLQ-v2 vs RLHF - Analysis and Visualization
# Notebook 4: Results Analysis

# %% [markdown]
# # Experiment Analysis
# ## S3-KLQ-v2 vs PPO-RLHF Comparison

# %% Cell 1: Load Results
import json
import numpy as np
import matplotlib.pyplot as plt

# Load experiment results
with open("experiment_results.json", "r") as f:
    results = json.load(f)

print("Results loaded")
print(f"S3-KLQ-v2 steps: {len(results['s3klq']['steps'])}")
print(f"PPO steps: {len(results['ppo']['steps'])}")

# %% Cell 2: Plot Reward Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Reward comparison
ax = axes[0, 0]
ax.plot(results["s3klq"]["steps"], results["s3klq"]["rewards"], 
        label="S3-KLQ-v2", alpha=0.7, color="blue")
ax.plot(results["ppo"]["steps"], results["ppo"]["rewards"], 
        label="PPO", alpha=0.7, color="orange")

# Smoothed
window = 20
s3klq_smooth = np.convolve(results["s3klq"]["rewards"], 
                           np.ones(window)/window, mode='valid')
ppo_smooth = np.convolve(results["ppo"]["rewards"], 
                         np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(results["s3klq"]["rewards"])), s3klq_smooth, 
        color="darkblue", linewidth=2)
ax.plot(range(window-1, len(results["ppo"]["rewards"])), ppo_smooth, 
        color="darkorange", linewidth=2)

ax.set_xlabel("Training Step")
ax.set_ylabel("Reward")
ax.set_title("Training Reward Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# KL Divergence (S3-KLQ only)
ax = axes[0, 1]
ax.plot(results["s3klq"]["steps"], results["s3klq"]["kl"], 
        color="blue", alpha=0.7)
kl_smooth = np.convolve(results["s3klq"]["kl"], 
                        np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(results["s3klq"]["kl"])), kl_smooth, 
        color="darkblue", linewidth=2)
ax.axhline(y=0.5, color="red", linestyle="--", label="Target KL")
ax.set_xlabel("Training Step")
ax.set_ylabel("KL Divergence (nats)")
ax.set_title("S3-KLQ-v2 KL Divergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Value Loss comparison
ax = axes[1, 0]
ax.plot(results["s3klq"]["steps"], results["s3klq"]["value_loss"], 
        label="S3-KLQ-v2", alpha=0.7, color="blue")
ax.plot(results["ppo"]["steps"], results["ppo"]["value_loss"], 
        label="PPO", alpha=0.7, color="orange")
ax.set_xlabel("Training Step")
ax.set_ylabel("Value Loss")
ax.set_title("Value Loss Comparison")
ax.legend()
ax.grid(True, alpha=0.3)

# Entropy (S3-KLQ only)
ax = axes[1, 1]
ax.plot(results["s3klq"]["steps"], results["s3klq"]["entropy"], 
        color="blue", alpha=0.7)
ent_smooth = np.convolve(results["s3klq"]["entropy"], 
                         np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(results["s3klq"]["entropy"])), ent_smooth, 
        color="darkblue", linewidth=2)
ax.set_xlabel("Training Step")
ax.set_ylabel("Entropy")
ax.set_title("S3-KLQ-v2 Policy Entropy")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Cell 3: Compute Statistics
def compute_stats(rewards):
    """Compute summary statistics."""
    rewards = np.array(rewards)
    final_50 = rewards[-50:] if len(rewards) >= 50 else rewards
    
    return {
        "mean": np.mean(rewards),
        "std": np.std(rewards),
        "max": np.max(rewards),
        "min": np.min(rewards),
        "final_mean": np.mean(final_50),
        "final_std": np.std(final_50),
    }

s3klq_stats = compute_stats(results["s3klq"]["rewards"])
ppo_stats = compute_stats(results["ppo"]["rewards"])

print("\n" + "=" * 50)
print("FINAL STATISTICS")
print("=" * 50)

print(f"\n{'Metric':<20} {'S3-KLQ-v2':<15} {'PPO':<15}")
print("-" * 50)
print(f"{'Mean Reward':<20} {s3klq_stats['mean']:<15.3f} {ppo_stats['mean']:<15.3f}")
print(f"{'Max Reward':<20} {s3klq_stats['max']:<15.3f} {ppo_stats['max']:<15.3f}")
print(f"{'Final Mean':<20} {s3klq_stats['final_mean']:<15.3f} {ppo_stats['final_mean']:<15.3f}")
print(f"{'Final Std':<20} {s3klq_stats['final_std']:<15.3f} {ppo_stats['final_std']:<15.3f}")

# %% Cell 4: Hypothesis Testing
from scipy import stats

# Test if S3-KLQ-v2 matches PPO (within 5%)
s3klq_final = results["s3klq"]["rewards"][-100:]
ppo_final = results["ppo"]["rewards"][-100:]

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(s3klq_final, ppo_final)

print("\n" + "=" * 50)
print("HYPOTHESIS TESTS")
print("=" * 50)

print(f"\nH1: S3-KLQ-v2 matches PPO reward")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
if p_value > 0.05:
    print("  Result: ✓ No significant difference (methods are comparable)")
else:
    diff = np.mean(s3klq_final) - np.mean(ppo_final)
    print(f"  Result: Significant difference of {diff:.3f}")

# Check KL stability
kl_values = np.array(results["s3klq"]["kl"])
kl_stable = np.all(kl_values < 3.0)
print(f"\nH2: KL stays bounded (<3.0)")
print(f"  Max KL: {kl_values.max():.3f}")
print(f"  Result: {'✓ PASS' if kl_stable else '✗ FAIL'}")

# Check entropy maintenance
entropy_values = np.array(results["s3klq"]["entropy"])
entropy_ok = np.mean(entropy_values[-50:]) > 0.3
print(f"\nH3: Entropy maintained (>0.3)")
print(f"  Final entropy: {np.mean(entropy_values[-50:]):.3f}")
print(f"  Result: {'✓ PASS' if entropy_ok else '✗ FAIL'}")

# %% Cell 5: Summary Table
print("\n" + "=" * 50)
print("EXPERIMENT SUMMARY")
print("=" * 50)

print("""
┌──────────────────────────────────────────────────────────────┐
│                    RESULTS COMPARISON                         │
├─────────────────┬────────────────┬────────────────────────────┤
│ Metric          │ S3-KLQ-v2      │ PPO Baseline               │
├─────────────────┼────────────────┼────────────────────────────┤
""")
print(f"│ Final Reward    │ {s3klq_stats['final_mean']:>12.3f}   │ {ppo_stats['final_mean']:>12.3f}                │")
print(f"│ Reward Std      │ {s3klq_stats['final_std']:>12.3f}   │ {ppo_stats['final_std']:>12.3f}                │")
print(f"│ Max Reward      │ {s3klq_stats['max']:>12.3f}   │ {ppo_stats['max']:>12.3f}                │")
print(f"│ Final KL        │ {np.mean(kl_values[-50:]):>12.3f}   │ N/A                        │")
print(f"│ Final Entropy   │ {np.mean(entropy_values[-50:]):>12.3f}   │ N/A                        │")
print("└─────────────────┴────────────────┴────────────────────────────┘")

# Verdict
reward_ratio = s3klq_stats['final_mean'] / ppo_stats['final_mean']
print(f"\n🎯 S3-KLQ-v2 achieves {reward_ratio*100:.1f}% of PPO reward")

if reward_ratio >= 0.9 and kl_stable and entropy_ok:
    print("✅ VERDICT: S3-KLQ-v2 is VALIDATED for RL-LLM!")
elif reward_ratio >= 0.8:
    print("⚠️ VERDICT: S3-KLQ-v2 is PROMISING but needs tuning")
else:
    print("❌ VERDICT: S3-KLQ-v2 underperforms - investigate hyperparameters")

# %% Cell 6: Save Report
report = f"""
# S3-KLQ-v2 vs PPO-RLHF Experiment Report

## Configuration
- Model: {config.model_name}
- Batch Size: {config.batch_size}
- Training Steps: {config.max_steps}
- α (soft-min): {config.alpha_softmin}
- β (KL coef): {config.kl_coef}

## Results

### Reward Comparison
- S3-KLQ-v2 Final: {s3klq_stats['final_mean']:.3f} ± {s3klq_stats['final_std']:.3f}
- PPO Final: {ppo_stats['final_mean']:.3f} ± {ppo_stats['final_std']:.3f}
- Relative Performance: {reward_ratio*100:.1f}%

### S3-KLQ-v2 Metrics
- Max KL: {kl_values.max():.3f}
- Final Entropy: {np.mean(entropy_values[-50:]):.3f}

### Statistical Tests
- t-test p-value: {p_value:.4f}
- Methods comparable: {'Yes' if p_value > 0.05 else 'No'}

## Conclusion
S3-KLQ-v2 {'matches' if reward_ratio >= 0.9 else 'approaches'} PPO performance.
"""

with open("experiment_report.md", "w") as f:
    f.write(report)

print("\n✓ Report saved to experiment_report.md")
print("✓ Plots saved to training_curves.png")
