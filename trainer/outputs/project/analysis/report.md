# Project Result Summary

Matched-count winners use the mean of their PSNR, SSIM, and LPIPS ranks. PSNR and 
SSIM are ranked high-to-low; LPIPS is ranked low-to-high. Metric-specific deltas remain 
in `matched_comparisons.csv` so a rank tie or disagreement is not hidden.

| Scene | Budget | Best direction | Penalty | Count | PSNR | SSIM | LPIPS | Mean rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| drums | 20% | importance only | 0 | 60,000 | 21.735 | 0.89094 | 0.09262 | 1.00 |
| drums | 40% | importance only | 0 | 120,000 | 22.128 | 0.89649 | 0.08432 | 1.00 |
| ficus | 20% | mask then importance | 0.004 | 60,000 | 22.311 | 0.90839 | 0.06691 | 1.00 |
| ficus | 40% | mask then importance | 0.004 | 120,000 | 22.446 | 0.91097 | 0.06592 | 1.00 |
| lego | 20% | importance only | 0 | 60,000 | 26.169 | 0.93626 | 0.04853 | 1.00 |
| lego | 40% | importance only | 0 | 120,000 | 26.799 | 0.94257 | 0.04032 | 1.00 |

## Excluded or degenerate settings

These settings are retained as negative results but excluded from matched-count ranking 
because they did not contain enough Gaussians to reach the requested budget.

| Scene | Setting | Arm | Status | Available | Target | Reason |
|---|---|---|---|---:|---:|---|
| drums | mask_0p016 | budget_0p20 | overshot_budget | 16 | 60000 | Learned masking retained only 16 Gaussians, below the exact 20% target of 60,000; Gaussians cannot be added back. |
| drums | mask_0p016 | budget_0p40 | overshot_budget | 16 | 120000 | Learned masking retained only 16 Gaussians, below the exact 40% target of 120,000; Gaussians cannot be added back. |
| ficus | mask_0p008 | budget_0p20 | overshot_budget | 39 | 60000 | Learned masking retained only 39 Gaussians, below the exact 20% target of 60,000; Gaussians cannot be added back. |
| ficus | mask_0p008 | budget_0p40 | overshot_budget | 39 | 120000 | Learned masking retained only 39 Gaussians, below the exact 40% target of 120,000; Gaussians cannot be added back. |
| lego | mask_0p016 | budget_0p20 | overshot_budget | 84 | 60000 | Learned masking retained only 84 Gaussians, below the exact 20% target of 60,000; Gaussians cannot be added back. |
| lego | mask_0p016 | budget_0p40 | overshot_budget | 84 | 120000 | Learned masking retained only 84 Gaussians, below the exact 40% target of 120,000; Gaussians cannot be added back. |
| ficus | mask_0p016 | training | collapsed | 0 |  | Learned-mask training progressively closed every Gaussian. |

## Interpretation guide

A combined-method winner supports mask-first pruning for that scene and budget; an 
importance-only winner indicates that learned masking did not improve the final 
quality/count trade-off. Compare the Gaussian diagnostic table across scenes to relate 
that outcome to opacity, scale, visibility, and transmittance. These are associations, 
not causal claims, because the project uses one training seed per setting.
