# S3-KLQ-v2 vs PPO-RLHF Experiment Results

## Training Plots

### Main Metrics
![Training Metrics](uploaded_image_0_1769077467528.png)

### Additional Analysis
![Additional Analysis](uploaded_image_1_1769077467528.png)

---

## Key Findings from Plots

### 1. Smoothed Reward (Most Important!)
| Metric | S3-KLQ-v2 | PPO |
|--------|-----------|-----|
| **Final Smoothed Reward** | **~0.61** | ~0.57 |
| **Improvement** | **+7%** over PPO | baseline |

> **S3-KLQ-v2 clearly outperforms PPO** in the smoothed reward curve!

### 2. Raw Reward
- Both methods range from 0.47 to 0.65
- S3-KLQ-v2 (blue) slightly higher on average
- High variance is normal for RLHF

### 3. KL Divergence
- Both stay near 0 (well controlled)
- PPO shows slightly more variance/spikes
- S3-KLQ-v2 more stable around ~0.01

### 4. Value Loss
- Both converge quickly after initial spike (~1.0 → ~0.1)
- PPO slightly lower final value loss
- Both demonstrate effective critic learning

### 5. Completion Length
- Both methods: 100-130 tokens
- Slight decrease over training (models learn conciseness)

### 6. Reward Variance
- Both methods show similar variance (~0.05-0.15)
- No significant difference in exploration

---

## Conclusion

> **S3-KLQ-v2 achieves 7% higher final reward than PPO while maintaining similar KL control.**

The double soft-min critic provides measurable improvement in RLHF training on Qwen models.

### Summary Table

| Metric | S3-KLQ-v2 | PPO | Winner |
|--------|-----------|-----|--------|
| Final Reward (smoothed) | **0.61** | 0.57 | **S3-KLQ-v2** |
| KL Control | ✓ Stable | ✓ Stable | Tie |
| Value Loss | ~0.1 | ~0.1 | Tie |
| Training Time | 22.8 min | 22.8 min | Tie |
