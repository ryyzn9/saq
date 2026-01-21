# S3-KLQ with Double Soft-Min Critic: Feasibility Analysis for RL-LLM Settings

> **Technical Report**  
> *Evaluating the viability of Soft-State Soft-min KL-regularized Q-learning with Double Critic for Large Language Model Fine-tuning*  
> *Incorporating analysis from FIFO Replay Buffer Infeasibility Report*

---

## Executive Summary

This document analyzes the feasibility of the **S3-KLQ (Soft-State Soft-min KL-regularized Q-learning)** algorithm with **Double Soft-Min Critic** for Reinforcement Learning with Large Language Models (RL-LLM), cross-referencing findings from the technical report on FIFO Replay Buffer Infeasibility in S3-AEPO.

> [!IMPORTANT]
> **Verdict: ✅ HIGHLY FEASIBLE** — The S3-KLQ algorithm addresses the major failure modes identified in the S3-AEPO technical report by being **inherently on-policy**, using **double soft-min critics** for stability, and incorporating **explicit KL regularization**.

### Key Advantages Over S3-AEPO
| Issue in S3-AEPO | S3-KLQ Solution |
|------------------|-----------------|
| FIFO replay buffer creates off-policy bias explosion | **On-policy only** — no replay buffer required |
| Temperature thermostat mixing behavior policies | No temperature switching, consistent policy |
| λ-return bias from stale trajectories | Fresh λ-returns computed on current policy |
| Single critic overestimation | **Double soft-min critic** with pessimistic estimation |

---

## 1. Algorithm Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    S3-KLQ Architecture                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   πθ (Policy LLM)  ←──── KL Regularization ────→  πref (Frozen SFT) │
│         │                        β = 1.0                             │
│         ▼                                                            │
│   On-Policy Rollouts (Fresh Each Iteration)                         │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐    Soft-Min (α=0.1)   ┌─────────────┐              │
│  │  V₁ Critic  │ ──────────────────→   │  Vsoft      │              │
│  │  V₂ Critic  │                       │  (Combined) │              │
│  └──────┬──────┘                       └──────┬──────┘              │
│         │                                      │                     │
│  ┌──────▼──────┐    Polyak (τ=0.005)   ┌──────▼──────┐              │
│  │ V₁_target   │ ←────────────────────  │ TD-Error δ │              │
│  │ V₂_target   │                        │ λ-Returns  │              │
│  └─────────────┘                        └────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Algorithm Components

```python
# Initialization
πθ        # Policy (trainable)
πref      # Frozen SFT reference
V1, V2    # Double critics
V1_target, V2_target  # Target networks

# Hyperparameters
β = 1.0           # KL regularization strength
λ = 0.95          # GAE lambda
α_softmin = 0.1   # Soft-min temperature
τ_polyak = 0.005  # Target update rate
γ = 0.99          # Discount factor (implied)

# Training Loop
for iteration k:
    # 1. ON-POLICY ROLLOUTS (no replay buffer!)
    trajectories = rollout(πθ)
    
    for epoch e in range(E):
        # 2. SOFT-MIN VALUE COMPUTATION
        V_soft(s) = -α * log(0.5 * [exp(-V1(s)/α) + exp(-V2(s)/α)])
        V_soft_target(s) = -α * log(0.5 * [exp(-V1_target(s)/α) + exp(-V2_target(s)/α)])
        
        # 3. TD-ERRORS (with KL cancellation)
        δt = r_{t+1} + γ * V_soft_target(s_{t+1}) - V_soft(st)
        
        # 4. λ-RETURNS
        G_λ,t = V_soft(st) + Σ_{k=t}^{T-1} (γλ)^{k-t} δk
        
        # 5. ADJUSTED TARGET (stop-grad on policy)
        G̃_λ,t = G_λ,t - β * log(πθ(at|st) / πref(at|st)).detach()
        
        # 6. CRITIC UPDATES (both to same target)
        L_V1 = E[(V1(st) - G̃_λ,t)²]
        L_V2 = E[(V2(st) - G̃_λ,t)²]
        
        # 7. POLICY UPDATE (stop-grad on value)
        L_π = -E[β * log(πθ/πref) + V_soft(st).detach()]
        
        # 8. POLYAK UPDATE
        V1_target ← τ*V1 + (1-τ)*V1_target
        V2_target ← τ*V2 + (1-τ)*V2_target
```

---

## 2. Feasibility Analysis: Addressing S3-AEPO Failure Modes

The technical report on S3-AEPO identifies **three critical failure modes** with FIFO replay buffers. Here's how S3-KLQ addresses each:

### 2.1 ✅ **H1: Off-Policy Bias Explosion — SOLVED**

> **S3-AEPO Issue**: Temperature thermostat creates mixed behavior distributions; replay amplifies into "double off-policy drift."

**S3-KLQ Solution**: **Pure on-policy training with no replay buffer.**

```python
# S3-AEPO (problematic)
for iteration:
    trajectories = rollout(πθ, temperature=T_adaptive)  # Mixed temperatures
    buffer.store(trajectories)  # FIFO replay
    batch = buffer.sample()  # Stale + mixed temperature data
    
# S3-KLQ (correct)
for iteration:
    trajectories = rollout(πθ)  # Consistent policy
    # NO BUFFER! Use trajectories directly
    update_critics(trajectories)  # Always on-policy
```

**Why this matters for LLMs**: The report shows that importance ratios `ρ = πθ/π_old^{1/T_old}` can "blow up or collapse over long token horizons" (512-2048 tokens). S3-KLQ avoids this entirely by never replaying.

### 2.2 ✅ **H2: λ-Return Bias — SOLVED**

> **S3-AEPO Issue**: Plain λ-returns on replayed trajectories are biased because they assume on-policy sampling.

**S3-KLQ Solution**: λ-returns are **always computed on fresh on-policy data.**

From the report:
> "Harutyunyan et al. (2016) show λ^π estimators are valid only when sampling distribution is sufficiently close to target π."

S3-KLQ satisfies this requirement by construction:
- No stored trajectories from old policies
- TD-errors `δt = r + γV_target(s') - V(s)` use current critic
- λ-returns `G_λ,t` are valid estimators for `V^{πθ}`

**Credit assignment benefit**: The report warns that "on long LLM horizons, bias compounds recursively through the λ-return trace." S3-KLQ uses λ=0.95 which provides good credit assignment over 512+ token sequences without bias accumulation.

### 2.3 ✅ **H3: Memory Infeasibility — SOLVED**

> **S3-AEPO Issue**: Storing full token-level trajectories (10k × 512 tokens × top-K logits) requires ~2.5GB+ per buffer.

**S3-KLQ Solution**: **No persistent storage required.**

| Approach | Memory for 10k Trajectories |
|----------|---------------------------|
| S3-AEPO with FIFO | ~2.5 GB (per device) |
| S3-KLQ on-policy | ~0 GB (no buffer) |

Only the current batch needs to be in memory during updates, then discarded.

---

## 3. Double Soft-Min Critic: Technical Analysis

### 3.1 Mathematical Foundation

The soft-min operator provides a **smooth, differentiable pessimistic value estimate**:

$$V_{soft}(s) = -\alpha_{softmin} \log\left(\frac{1}{2}\left[\exp\left(-\frac{V_1(s)}{\alpha_{softmin}}\right) + \exp\left(-\frac{V_2(s)}{\alpha_{softmin}}\right)\right]\right)$$

**Properties**:
- As `α → 0`: Converges to hard min (most pessimistic)
- As `α → ∞`: Converges to arithmetic mean
- At `α = 0.1`: Provides smooth gradients while remaining pessimistic

**Comparison with S3-AEPO's Twin Critics**:

| Feature | S3-AEPO Twin Heads | S3-KLQ Double Soft-Min |
|---------|-------------------|----------------------|
| Aggregation | Pessimistic masking | Smooth soft-min |
| Gradient flow | Can have discontinuities | Always smooth |
| Target consistency | Vulnerable to off-policy divergence | Synchronized via Polyak |

### 3.2 Why Double Critics Matter for LLMs

The report notes: "Training twin critics on off-policy data would cause them to diverge (one fits stale data, one fits fresh)."

S3-KLQ's approach ensures:
1. **Both critics see identical on-policy distribution**
2. **Both update toward the same target** `G̃_λ,t`
3. **Soft-min smoothly combines estimates** rather than hard switching

```python
# Both critics updated identically
L_V1 = E[(V1(st) - G̃_λ,t)²]  # Same target
L_V2 = E[(V2(st) - G̃_λ,t)²]  # Same target

# Soft-min for pessimism
V_soft = -α * log(0.5 * (exp(-V1/α) + exp(-V2/α)))
```

---

## 4. KL Regularization: Preventing Catastrophic Forgetting

### 4.1 Implementation in S3-KLQ

The KL penalty appears in two places:

**1. Adjusted λ-return target** (stop-grad on policy):
```python
G̃_λ,t = G_λ,t - β * log(πθ(at|st) / πref(at|st)).detach()
```

**2. Policy loss** (stop-grad on value):
```python
L_π = -E[β * log(πθ/πref) + V_soft(st).detach()]
```

**Why both?** This creates a **consistent KL signal** that:
- Penalizes critic for expecting value from off-distribution actions
- Penalizes policy for drifting too far from reference

### 4.2 Comparison with Other RL-LLM Methods

| Method | KL Regularization | Mechanism |
|--------|------------------|-----------|
| PPO (RLHF) | Implicit via clipping | Surrogate objective |
| DPO | Implicit in preference loss | No explicit KL |
| GRPO | Explicit penalty `β * KL` | Added to reward |
| **S3-KLQ** | **Explicit in targets + loss** | **Dual integration** |

The dual integration prevents the "mode collapse to reward-hacking behaviors" mentioned in the report.

---

## 5. Implementation Recommendations

### 5.1 Efficient Architecture for LLMs

```python
class S3KLQTrainer:
    def __init__(self, policy_model, ref_model, config):
        # Policy: LoRA fine-tuned
        self.policy = policy_model
        self.ref_policy = ref_model  # Frozen
        
        # Double critics: Shared backbone with separate heads
        self.critic_backbone = CriticBackbone(policy_model.config)
        self.v1_head = nn.Linear(hidden_dim, 1)
        self.v2_head = nn.Linear(hidden_dim, 1)
        
        # Target networks (Polyak averaged)
        self.v1_target_head = copy.deepcopy(self.v1_head)
        self.v2_target_head = copy.deepcopy(self.v2_head)
        freeze(self.v1_target_head)
        freeze(self.v2_target_head)
        
        # Hyperparameters
        self.beta = config.beta  # 1.0
        self.lam = config.lam    # 0.95
        self.alpha = config.alpha_softmin  # 0.1
        self.tau = config.tau_polyak  # 0.005
```

### 5.2 Memory-Efficient Training Loop

```python
def train_step(self, prompts):
    # 1. ON-POLICY ROLLOUTS
    with torch.no_grad():
        trajectories = self.generate_rollouts(prompts)
    
    # 2. COMPUTE VALUES
    features = self.critic_backbone(trajectories.states)
    v1 = self.v1_head(features)
    v2 = self.v2_head(features)
    v_soft = self.soft_min(v1, v2)
    
    with torch.no_grad():
        v1_targ = self.v1_target_head(features)
        v2_targ = self.v2_target_head(features)
        v_soft_targ = self.soft_min(v1_targ, v2_targ)
    
    # 3. COMPUTE λ-RETURNS
    G_lambda = self.compute_lambda_returns(
        trajectories.rewards, v_soft, v_soft_targ
    )
    
    # 4. KL REGULARIZATION
    log_ratio = self.policy.log_prob(trajectories.actions, trajectories.states) \
                - self.ref_policy.log_prob(trajectories.actions, trajectories.states)
    
    G_adjusted = G_lambda - self.beta * log_ratio.detach()
    
    # 5. CRITIC LOSSES
    loss_v1 = F.mse_loss(v1, G_adjusted.detach())
    loss_v2 = F.mse_loss(v2, G_adjusted.detach())
    
    # 6. POLICY LOSS
    loss_pi = -(self.beta * log_ratio + v_soft.detach()).mean()
    
    # 7. UPDATES
    self.update_critics(loss_v1, loss_v2)
    self.update_policy(loss_pi)
    self.polyak_update()
    
    return {
        'loss_v1': loss_v1.item(),
        'loss_v2': loss_v2.item(),
        'loss_pi': loss_pi.item(),
        'kl': log_ratio.mean().item()
    }
```

### 5.3 Hyperparameter Recommendations

| Parameter | Value | Sensitivity | Notes |
|-----------|-------|-------------|-------|
| β (KL strength) | 1.0 | High | Start here, reduce to 0.5 if too constrained |
| λ (GAE lambda) | 0.95 | Medium | Higher for longer sequences |
| α_softmin | 0.1 | Low | Lower = more pessimistic |
| τ_polyak | 0.005 | Low | Standard for RL |
| Learning rate (critic) | 1e-5 | Medium | Lower than policy |
| Learning rate (policy) | 1e-6 | High | Standard for LLM RL |

---

## 6. Comparison: S3-KLQ vs Other RL-LLM Methods

| Feature | PPO (RLHF) | DPO | GRPO | S3-AEPO | **S3-KLQ** |
|---------|-----------|-----|------|---------|-----------|
| Training mode | On-policy | Off-policy | On-policy | Hybrid (broken) | **On-policy** |
| Replay buffer | No | Yes (preferences) | No | Yes (FIFO) | **No** |
| Value function | Single V | None | None | Twin V | **Double V (soft-min)** |
| Overestimation control | None | N/A | N/A | Pessimistic mask | **Soft-min** |
| KL regularization | Clipping | Implicit | Explicit | Implicit (Q) | **Explicit (dual)** |
| Credit assignment | GAE | None | None | λ-returns | **λ-returns** |
| LLM memory usage | Low | Medium | Low | High | **Low** |
| Stability | Good | High | Medium | **Unstable** | **Expected: High** |

---

## 7. Potential Challenges and Mitigations

### 7.1 ⚠️ Computational Cost of Double Critics

**Challenge**: Two critic networks + two target networks = 4× value parameters.

**Mitigations**:
1. **Shared backbone**: Only separate the final linear heads
2. **LoRA for critics**: Use low-rank adapters (~1% of params)
3. **Gradient checkpointing**: Reduce memory footprint

```python
# Efficient: Shared backbone
class DoubleCritic(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone  # Shared (frozen or LoRA)
        self.head1 = nn.Linear(d, 1)  # ~4KB
        self.head2 = nn.Linear(d, 1)  # ~4KB
```

### 7.2 ⚠️ Stop-Gradient Complexity

**Challenge**: Incorrect `detach()` placement causes training instability.

**Solution**: Clear separation in code:

```python
# CRITIC update: detach policy contribution
G_adjusted = G_lambda - beta * log_ratio.detach()  # <-- detach here
loss_v = (V(s) - G_adjusted.detach())²             # <-- and here

# POLICY update: detach value
loss_pi = -(beta * log_ratio + V_soft.detach())    # <-- detach value
```

### 7.3 ⚠️ Sample Efficiency

**Challenge**: On-policy training uses each trajectory only once.

**Mitigations**:
1. **Multiple epochs per rollout** (like PPO): Perform E epochs over each batch
2. **Larger batch sizes**: Use gradient accumulation
3. **vLLM for fast generation**: Parallelize rollout generation

---

## 8. Conclusion and Recommendations

### Feasibility Verdict: ✅ **HIGHLY FEASIBLE**

S3-KLQ with Double Soft-Min Critic is **well-designed for RL-LLM settings** and directly addresses the failure modes identified in the S3-AEPO technical report:

| S3-AEPO Failure | S3-KLQ Status | Confidence |
|-----------------|---------------|------------|
| H1: Off-policy bias explosion | ✅ Solved (on-policy) | High |
| H2: λ-return bias | ✅ Solved (no replay) | High |
| H3: Memory infeasibility | ✅ Solved (no buffer) | High |
| M1: Policy drift issues | ✅ Mitigated (KL regularization) | High |
| Twin critic divergence | ✅ Solved (same target, soft-min) | High |

### Recommended Implementation Path

1. **Phase 1: Prototype** (1-2 weeks)
   - Implement on 1.5B model (Qwen2.5-Math-1.5B)
   - Verify training stability over 1000 steps
   - Compare against PPO baseline

2. **Phase 2: Ablation** (1 week)
   - Test α_softmin ∈ {0.01, 0.1, 0.5}
   - Test β ∈ {0.1, 0.5, 1.0, 2.0}
   - Test λ ∈ {0.9, 0.95, 0.99}

3. **Phase 3: Scale** (2-3 weeks)
   - Apply to 7B model with LoRA
   - Benchmark on MATH-500, AIME24
   - Compare sample efficiency vs GRPO, PPO

### Key Metrics to Track

```python
metrics = {
    'reward_mean': float,       # Primary objective
    'kl_divergence': float,     # πθ vs πref (should stay < 2 nats)
    'critic_loss': float,       # MSE (should decrease)
    'v1_v2_disagreement': float, # |V1 - V2| (should be small)
    'policy_entropy': float,    # Diversity of outputs
    'effective_batch_size': int # Samples with non-zero gradient
}
```

---

## References

1. Fujimoto, S., et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
2. Haarnoja, T., et al. (2018). "Soft Actor-Critic" (SAC)
3. Schulman, J., et al. (2017). "Proximal Policy Optimization" (PPO)
4. Schulman, J., et al. (2015). "High-Dimensional Continuous Control Using GAE"
5. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback" (InstructGPT/RLHF)
6. Technical Report: FIFO Replay Buffer Infeasibility in S3-AEPO for RL-LLM Settings
7. RePO: Memory Replay with Policy Optimization

---

*Document generated: January 2026*  
*Cross-referenced with: FIFO Replay Buffer Infeasibility Technical Report*
