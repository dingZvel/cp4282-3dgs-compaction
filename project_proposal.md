# Project Proposal: Which Scenes Benefit from Learned-Mask and Importance Pruning?

## Research question

For different types of scenes, what balance between learned-mask pruning and post-training importance pruning gives the best novel-view quality at the same final number of Gaussians? I will test whether dense, self-occluded scenes favour a weaker mask followed by heavier importance pruning, while sparse scenes containing thin or rarely observed structures favour a stronger learned mask followed by lighter importance pruning.

I will use the course Warp 3DGS trainer on the `lego`, `ficus`, and `drums` scenes from NeRF Synthetic. The goal is a scene-dependent rule of thumb for choosing the mask penalty and importance-pruning percentage.

## Motivation and course connection

This project builds on Unit 14, *Compaction of 3DGS* [2], and the material on alpha compositing, differentiable rendering, training, and evaluation. Learned-mask pruning penalizes open trainable masks during training. Importance pruning instead ranks a trained model using each Gaussian's accumulated blending weight over the training views.

The methods use different evidence. Jointly optimized masks let neighbouring Gaussians adapt, but a strong penalty may delete useful structure early. Importance uses post-training visibility and occlusion, but may undervalue Gaussians seen in few cameras. The best division may therefore depend on scene structure.

## Hypothesis about scene types

- **Dense or self-occluded scenes** should favour a weak mask followed by heavier importance pruning because they may contain opaque but hidden Gaussians with low blending weight.
- **Sparse scenes with thin or rarely visible structures** should favour a stronger mask followed by lighter importance pruning because importance may undervalue features covering few pixels or views.
- **Mixed scenes** should favour a balanced combination because they contain both redundant occluded regions and fragile visible details.

I will treat `lego` as dense and self-occluded, `ficus` as dominated by thin irregular structures, and `drums` as mixed. The experiments will test rather than assume that these labels predict the result.

## Method and comparisons

For Gaussian $i$, learned-mask training uses a soft mask $\operatorname{sigmoid}(m_i)$ and a hard forward mask

$$
\chi_i =
\mathbb{1}\!\left[
\operatorname{sigmoid}(m_i) > \tau_{\mathrm{mask}}
\right],
\qquad \tau_{\mathrm{mask}} = 0.01.
$$

The training objective is

$$
L = L_{\mathrm{render}} + \lambda_{\mathrm{mask}}L_{\mathrm{mask}},
\qquad
L_{\mathrm{mask}} =
\frac{1}{M}\sum_{i=1}^{M}\operatorname{sigmoid}(m_i).
$$

A straight-through estimator supplies the sigmoid gradient during back-propagation.

After training, I will compute the Unit 14 importance score

$$
U_i =
\sum_{v \in \mathcal{V}_{\mathrm{train}}}
\sum_{\mathbf{p}}
T_{i,v}(\mathbf{p})\,
\alpha_{i,v}(\mathbf{p}).
$$

This is the blending weight summed over all training pixels and views. The lowest-ranked Gaussians will be removed and the model will first be evaluated immediately, before any further optimization.

I will sweep $\lambda_{\mathrm{mask}} \in \{0.002, 0.004, 0.008, 0.016\}$ and the importance-pruning fraction $r_{\mathrm{importance}} \in \{0, 0.25, 0.50, 0.75\}$. Combinations will be grouped as importance-dominant, balanced, or mask-dominant. Comparisons will use matched final retention budgets of approximately $40\%$ and $20\%$ of the unpruned count; a larger model will be pruned to the smaller count before evaluation.

The baselines are unpruned 3DGS, importance-only pruning, and learned-mask-only pruning. All arms will use the same initialization, seed, resolution, optimizer, densification schedule, and 20,000 training iterations. If time and compute allow, I will also fine-tune every compacted model for 5,000 recovery iterations with densification and further pruning disabled. I will record metrics both immediately after pruning and after recovery. Their difference will show both the immediate safety of each pruning decision and how well the retained Gaussians can adapt. If recovery runs are not feasible, I will evaluate importance pruning without recovery and state clearly that the results measure immediate pruning damage rather than the best quality achievable after fine-tuning. Importance-based pruning followed by recovery has also been explored by LightGaussian [1], although its global-significance score differs from the accumulated blending-weight score used in this project.

## Implementation and experiment plan

I will add mask logits, hard masking of opacity and scale, straight-through gradients, mask inheritance during densification, a per-Gaussian $T_i\alpha_i$ accumulator, exact-count pruning, recovery, and diagnostic logging to the course trainer.

A fixed 10% of training cameras will be held out for validation and excluded from optimization and importance accumulation. Test views will be evaluated only after selection. I will report the GPU, runtime, and peak memory.

## Evaluation and analysis

At each matched Gaussian count, I will report PSNR, SSIM, LPIPS, model size, rendering speed, training time, and peak GPU memory. I will also report per-view quality differences and magnified crops from the views where each direction performs best and worst.

To explain the result, I will compare removed populations by opacity, scale, pixel coverage, visible-view count, transmittance, importance, and location. I will measure overlap between removal sets and relate occlusion, coverage, and thin-structure prevalence to the best combination.

The project succeeds if these measurements support a repeatable relationship between scene properties and the preferred pruning direction at matched size. The hypothesis fails if no direction consistently fits the proposed scene categories, if differences are within measurement variation, or if the apparent rule does not transfer beyond the validation scene. A negative result remains useful if it shows that scene type alone is insufficient and identifies which Gaussian-level statistics are more predictive.

## References

1. Z. Fan et al. “LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS.” *NeurIPS*, 2024. [arXiv:2311.17245](https://arxiv.org/abs/2311.17245).
2. W. Tsang. “Unit 14: Compaction of 3DGS.” [CP4282 course notes](https://weitsang.github.io/gaussian-splatting-notes/unit-14/).
