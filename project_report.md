# Scene-Dependent Compaction of 3D Gaussian Splatting

## Final report

## Abstract

This project investigated whether the best combination of learned-mask pruning and post-training
importance pruning depends on scene type. A Warp-based 3D Gaussian Splatting (3DGS) trainer was
extended with trainable hard masks, a straight-through mask gradient, an accumulated blending-
weight importance score, exact-count pruning, and diagnostic logging. Experiments used the Drums,
Ficus, and Lego scenes from NeRF Synthetic. Five mask settings were trained for each scene and the
successful models were evaluated under four relative importance-pruning fractions and two matched
final budgets: 120,000 Gaussians (40% retention) and 60,000 Gaussians (20% retention).

The initial hypothesis was that learned masks should generally outperform post-training importance
pruning because the mask variables are optimized jointly with the rendering objective and can learn
which Gaussians the representation can discard. Scene structure was expected to determine the size
of this advantage and the appropriate balance between the two pruning directions.

The result is scene-dependent, but not in the simple way originally hypothesized. Ficus benefits
from a mild mask penalty of 0.004 followed by importance pruning. Relative to importance pruning
alone, the combined method gains 0.879 dB at 40% retention and 1.036 dB at 20% retention, while
also improving SSIM and LPIPS. Drums and Lego favour importance pruning alone at both budgets.
For Lego, even the weakest mask penalty loses 1.468 dB at 40% retention and 2.123 dB at 20%
retention. The penalty sweep also exposes a brittle transition: penalty 0.016 leaves only 16
Gaussians for Drums, collapses Ficus to zero, and leaves 84 for Lego; Ficus already collapses to
39 Gaussians at 0.008. Consequently, the main conclusion is not that a semantic scene category
directly determines a fixed method combination. Instead, mask pressure is highly scene-sensitive,
and a useful mask penalty must be selected from the training dynamics or a target retention
constraint. Importance pruning is the safer default; a mild learned mask can help, but its benefit
must be validated per scene. In two of the three tested scenes, skipping learned-mask training both
improves matched-budget quality and avoids the additional full training runs required to tune the
scene-specific mask penalty by trial and error.

## 1. Research question

The project asked:

> For different types of scenes, what balance between learned-mask pruning and post-training
> importance pruning gives the best novel-view quality at the same final number of Gaussians?

The primary initial hypothesis was that learned masks should generally do a better job than
post-training importance pruning. Because the mask variables are trained jointly with the rendering
objective, they were expected to learn which Gaussians could be discarded without damaging the
representation, whereas importance pruning relies on a contribution score computed after training.

A secondary scene-structure hypothesis classified Lego as dense and self-occluded, Ficus as sparse
with thin and irregular structure, and Drums as mixed. It predicted that dense scenes would favour
a weak mask followed by stronger importance pruning, sparse scenes would tolerate a stronger mask,
and mixed scenes would prefer a balanced combination.

The experiments support the broader claim that the best direction is scene-dependent. They do not
support the expectation that learned masks usually outperform importance pruning: only Ficus gains,
while Drums and Lego perform better when the learned-mask step is skipped. They also do not support
the proposed monotonic mapping from scene labels to mask strength. In particular, Ficus benefits
from a *mild* mask penalty but is the first scene to collapse as the penalty grows.

## 2. Methods

### 2.1 Learned-mask training

Each Gaussian (i) has a trainable mask logit (m_i), soft mask
(sigma(m_i)), and hard forward mask

$$
\chi_i = \mathbb{1}\!\left[\sigma(m_i) > \tau_{\mathrm{mask}}\right],
\qquad \tau_{\mathrm{mask}}=0.01.
$$

The renderer uses the hard mask in the forward pass. A straight-through estimator supplies the
sigmoid derivative to the mask logit during back-propagation. The training objective is

$$
L = L_{\mathrm{render}} + \lambda_{\mathrm{mask}}L_{\mathrm{mask}},
\qquad
L_{\mathrm{mask}} = \frac{1}{M}\sum_{i=1}^{M}\sigma(m_i).
$$

Mask state is inherited during densification, optimizer state is reset for reused slots, and a
final cleanup removes Gaussians whose learned masks are closed.

### 2.2 Importance pruning

After training, each surviving Gaussian is assigned the Unit 14 accumulated blending-weight score

$$
U_i = \sum_{v\in\mathcal V_{\mathrm{train}}}\sum_{\mathbf p}
T_{i,v}(\mathbf p)\,\alpha_{i,v}(\mathbf p).
$$

Gaussians are sorted by (U_i), with stable index order for ties, and the lowest-scoring entries
are removed. The implementation supports both a pruning fraction and an exact retained count.
There is no recovery fine-tuning in the reported experiment, so the results measure immediate
pruning damage rather than the best quality achievable after adaptation.

### 2.3 Data and controlled protocol

The experiment used Drums, Ficus, and Lego from NeRF Synthetic. For each scene, 10 camera-diverse
views were deterministically selected from the original 100 training views for validation; the
remaining 90 were used for optimization and importance accumulation. The original 200 test views
were untouched until final evaluation.

All training arms used the same seed, initialization procedure, resolution, optimizer,
densification schedule, capacity, and 20,000 iterations. The mask sweep was

$$
\lambda_{\mathrm{mask}}\in\{0, 0.002, 0.004, 0.008, 0.016\},
$$

where zero denotes the importance-only baseline. Post-training pruning used relative removal
fractions of 0%, 25%, 50%, and 75%. The decisive comparisons used exact final counts:

- 120,000 Gaussians, or 40% of the 300,000-Gaussian unpruned baseline;
- 60,000 Gaussians, or 20% of the baseline.

A model that learned fewer Gaussians than a target was recorded as `overshot_budget` and excluded
from that matched-count ranking. Adding Gaussians back would not be a fair comparison, while
forcing every other model down to tens of Gaussians would no longer test the proposed budgets.

Quality was measured over all 200 test views using PSNR, SSIM, and LPIPS. The reported winner is
the mean of its three metric ranks; PSNR and SSIM are maximized, while LPIPS is minimized.

## 3. Results

### 3.1 Matched-count comparison

| Scene | Retention | Best direction | Mask penalty | Count | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|
| Drums | 20% | Importance only | 0 | 60,000 | 21.735 | 0.89094 | 0.09262 |
| Drums | 40% | Importance only | 0 | 120,000 | 22.128 | 0.89649 | 0.08432 |
| Ficus | 20% | Mask then importance | 0.004 | 60,000 | 22.311 | 0.90839 | 0.06691 |
| Ficus | 40% | Mask then importance | 0.004 | 120,000 | 22.446 | 0.91097 | 0.06592 |
| Lego | 20% | Importance only | 0 | 60,000 | 26.169 | 0.93626 | 0.04853 |
| Lego | 40% | Importance only | 0 | 120,000 | 26.799 | 0.94257 | 0.04032 |

The winner is consistent across the two budgets for every scene. This makes the scene-dependent
result stronger than a single-budget reversal: Ficus prefers the combined method, while Drums and
Lego prefer importance alone at both 40% and 20% retention.

### 3.2 Difference from importance-only pruning

| Scene | Retention | Best mask setting | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|---:|---:|
| Drums | 20% | 0.008 | −0.371 dB | −0.01701 | +0.01655 |
| Drums | 40% | 0.002 | −0.505 dB | −0.00852 | +0.00943 |
| Ficus | 20% | 0.004 | **+1.036 dB** | **+0.00713** | **−0.01090** |
| Ficus | 40% | 0.004 | **+0.879 dB** | **+0.00527** | **−0.00689** |
| Lego | 20% | 0.002 | −2.123 dB | −0.02412 | +0.02504 |
| Lego | 40% | 0.002 | −1.468 dB | −0.01454 | +0.01436 |

For Drums and Lego, even the best mask-trained alternative is worse in all three metrics. Lego is
especially sensitive: increasing the penalty monotonically worsens the matched result. At 20%
retention, penalties 0.002, 0.004, and 0.008 lose 2.123, 2.348, and 3.039 dB respectively against
importance alone.

Ficus is different. Penalty 0.004 wins every metric at both budgets. However, the improvement is
not created solely by post-training importance pruning. Before importance pruning, the 0.004
Ficus model already achieves 22.461 dB compared with 21.600 dB for the no-mask model. At 40%
retention it preserves almost all of that quality (22.446 dB), and at 20% it loses only 0.150 dB
from its own unpruned result. The correct interpretation is therefore:

1. mild mask regularization improves the Ficus training trajectory;
2. importance pruning preserves that improved representation very effectively;
3. the combination is better than importance-only pruning at the same final count.

This experiment does not isolate whether the gain comes from deleting a particular final set of
Gaussians during training or from regularizing how the remaining population is learned.

**Practical implication.** The initial expectation that a jointly learned pruning signal would
usually beat a post-training heuristic is rejected for two of the three scenes. Drums and Lego
achieve better PSNR, SSIM, and LPIPS with importance-only pruning at both matched budgets. The
importance-only route is also cheaper operationally: it trains one unmasked model and then scores
and sorts its Gaussians, whereas evaluating learned masking requires a separate training run for
each candidate value of \(\lambda_{\mathrm{mask}}\). Because the useful range is scene-dependent
and ends in a sharp collapse, selecting \(\lambda_{\mathrm{mask}}\) currently requires trial and
error. The additional computation is justified by the Ficus gain, but not by the Drums or Lego
results.

### 3.3 Robustness of importance-only pruning

Importance-only pruning is a strong baseline. Removing 75% of the unmasked 300,000-Gaussian model
and retaining 75,000 Gaussians changes PSNR by only:

- Drums: 22.155 to 21.951 dB, a loss of 0.204 dB;
- Ficus: 21.600 to 21.396 dB, a loss of 0.204 dB;
- Lego: 26.778 to 26.485 dB, a loss of 0.293 dB.

At the stricter exact 20% budget, importance-only pruning loses 0.419 dB on Drums, 0.325 dB on
Ficus, and 0.609 dB on Lego relative to each unpruned baseline. This stability explains why
mask-first training must offer a real representational benefit to win: importance ranking already
removes a large redundant population with little immediate damage.

### 3.4 Penalty instability and degenerate runs

| Scene | Penalty | Final count | Outcome |
|---|---:|---:|---|
| Drums | 0.016 | 16 | Severe budget overshoot |
| Ficus | 0.008 | 39 | Severe budget overshoot |
| Ficus | 0.016 | 0 | Complete collapse at approximately iteration 12,300 |
| Lego | 0.016 | 84 | Severe budget overshoot |

Ficus is not simply a scene that “likes stronger masking.” It is the most sensitive scene. With
penalty 0.016, its active population falls from 299,910 at iteration 5,300 to 5,504 at iteration
7,000, 87 at iteration 8,000, five at iteration 9,000, and zero at approximately iteration
12,300. Drums and Lego survive the same penalty only in a formal sense: 16 and 84 Gaussians are
not meaningful 20% or 40% budget models.

The sweep therefore identifies a sharp, scene-dependent phase transition. The useful region is
not well resolved by the coarse doubling sequence. On Ficus it lies below 0.008, whereas Drums and
Lego remain at the 300,000-Gaussian capacity through 0.008 and then collapse at 0.016.

### 3.5 What kinds of Gaussians were removed?

Importance pruning generally removes Gaussians with low accumulated contribution, but the
diagnostics show that mask training changes the population being ranked. For the importance-only
40% models, the median removed Gaussian is visible in roughly 44–45 training views and contributes
to 116–118 pixels. For the 0.008 mask-trained Drums and Lego models, the median removed Gaussian is
visible in only 28 and 32 views and contributes to 53 and 62 pixels. Its median transmittance-
weighted contribution is also lower. This is consistent with stronger mask training producing a
population in which many surviving entries are highly opaque but less broadly useful.

Opacity makes this contrast especially clear. At the 40% budget, the median opacity of Gaussians
removed from the importance-only models is 0.076 for Drums, 0.093 for Ficus, and 0.062 for Lego.
For penalty 0.008 it is approximately 0.999 for Drums and Lego. High opacity is therefore not
sufficient evidence that a Gaussian is globally useful: an opaque Gaussian can still have low
accumulated blending weight because it covers few pixels or views, or is occluded in the relevant
camera set.

The learned-mask event logs require caution. They contain millions of deletion *events*, not
millions of unique final Gaussians. Densification can reuse a slot after it is pruned, and the same
slot may be deleted again later. These data are useful for describing training dynamics, but they
cannot directly provide a Jaccard overlap between two unique final removal sets without persistent
Gaussian lineage identifiers.

## 4. Conclusions by scene

### Drums

Importance pruning alone is the best direction. Mask training degrades all three matched metrics,
and no tested penalty produces a useful final compaction level: penalties through 0.008 finish at
the 300,000 capacity, while 0.016 collapses to 16. Drums behaves more like a scene where the
post-training contribution score is sufficient than the proposed “mixed scene needs a balanced
combination” hypothesis predicted.

### Ficus

Ficus is the only scene that benefits from the combined method. Penalty 0.004 improves the learned
representation and importance pruning preserves the improvement down to 60,000 Gaussians. But the
scene is also the least tolerant of additional mask pressure. Thus the useful rule is not “thin
scenes prefer strong masks”; it is “this thin, irregular scene benefits from mild regularization,
but its penalty must be controlled carefully.”

### Lego

Importance pruning alone wins decisively. Learned masking reduces quality even when the final
training count remains 300,000, and stronger penalties cause progressively larger losses before
eventual collapse. The dense/self-occluded label is compatible with importance-dominant pruning,
but the learned mask does not add value at the tested settings.

## 5. How the experiment failed scientifically

The project produced usable findings, but several parts of the original experimental ambition
were not achieved.

### 5.1 The learned-mask advantage and scene taxonomy hypotheses mostly failed

The primary expectation that jointly trained masks would generally make better pruning decisions
than post-training importance ranking failed on Drums and Lego. Only Ficus benefits from the
combined method. The secondary taxonomy also failed: the proposed ordering—stronger masking for
sparse scenes and a balanced method for mixed scenes—does not hold. Ficus benefits from the
combined method but collapses earliest, while Drums does not prefer a balanced combination.

**Fix:** replace semantic labels with measurable predictors. Candidate features include opacity
histograms, visible-view counts, pixel coverage, transmittance, depth complexity, and the slope of
the active-count curve during a short mask warm-up. Fit or validate a decision rule on more scenes
rather than assigning a method from an informal category. Use importance pruning as the default;
run a short mask pilot first, and expand the \(\lambda_{\mathrm{mask}}\) search only when validation
quality improves without a rapid decline in active count.

### 5.2 The penalty sweep was too coarse near the transition

The values 0.002, 0.004, 0.008, and 0.016 cover a broad range but miss the narrow interval between
“no final count reduction” and “nearly total collapse.”

**Fix:** use a two-stage sweep. First run short pilot jobs to locate the transition, then sample it
densely—for example 0.004, 0.005, 0.006, 0.007, 0.008 for Ficus and 0.008, 0.010, 0.012, 0.014,
0.016 for Drums and Lego. Better still, control the mask penalty dynamically to track a target
retention curve rather than treating the penalty as an open-loop constant.

### 5.3 Low penalties did not themselves compact the final model

Most penalties finish at the 300,000-Gaussian capacity. Mask deletion events occur during
training, but densification refills freed slots. The successful Ficus combination is therefore a
regularized training trajectory followed by importance compaction, not a clean sequence in which
the mask alone reaches an intermediate target.

**Fix:** stop densification before the final mask-pruning phase, introduce a mask-only refinement
window, or enforce a target active-count schedule. Report both the active count over time and the
unique final count after cleanup.

### 5.4 Only one seed was tested

The same seed makes arms comparable, but one seed cannot establish that the Ficus gain or the
ordering of close Drums configurations is repeatable. The Ficus gain is large and consistent
across metrics, yet it could still include initialization-specific effects.

**Fix:** repeat at least the baseline and leading mask settings for three to five seeds. Report
mean and standard deviation, and bootstrap the paired per-view metric differences. A confidence
interval over the 200 common test views would distinguish consistent gains from a few favourable
cameras.

### 5.5 Recovery fine-tuning was omitted

The proposal made recovery optional, and the implemented experiment used the stated fallback:
immediate evaluation after importance pruning. The results therefore answer which deletion is
safest immediately, not which compact model can recover the best.

**Fix:** fine-tune every matched model for the same 5,000 iterations with densification and further
pruning disabled. Record metrics both before and after recovery. The difference separates pruning
safety from adaptation capacity.

### 5.6 Cost and qualitative evidence are incomplete

The experiment records Slurm runtime and peak host memory in logs and produces all test renders,
but the final aggregate table does not yet report model file size, GPU peak memory, rendering FPS,
or selected worst/best-view crops. These were promised in the proposal. Nevertheless, the protocol
already exposes an important training-cost asymmetry. Importance pruning can be applied to one
trained baseline, while the learned-mask study requires another full training run for every tested
penalty. The five-setting sweep therefore multiplies training work, and the cliff-like,
scene-dependent response makes this search trial-and-error rather than a reusable fixed choice. In
Drums and Lego, that additional computation produces lower matched-budget quality than skipping the
mask step.

**Fix:** add a fixed-camera rendering benchmark with warm-up, collect GPU memory through
`nvidia-smi`, store PLY byte size, aggregate `sacct` runtime and memory, and automatically select
views with the largest paired metric differences for qualitative figures. Report the cumulative
GPU-hours of the complete \(\lambda_{\mathrm{mask}}\) search separately from the cost of training
and importance-pruning a single baseline.

### 5.7 The diagnostic deletion records are event-level

Slot reuse means learned-mask deletion logs do not identify a unique, persistent Gaussian
population. Direct overlap with the final importance-pruned set is therefore unavailable.

**Fix:** assign a persistent lineage ID when a Gaussian is initialized, cloned, or split. Log the
lineage and parent ID on every deletion. Then compute removal-set overlap, survival by generation,
and spatial maps without double-counting reused slots.

### 5.8 The scene sample is too small for a general rule

Three synthetic object-centric scenes can demonstrate a counterexample, but cannot establish a
general scene-selection policy.

**Fix:** extend to all NeRF Synthetic scenes and at least one real forward-facing or 360-degree
dataset. Reserve scenes—not merely views—for testing any learned rule of thumb.

## 6. Engineering failures and implemented fixes

The automation itself failed in three instructive ways. All were repaired without discarding
completed evaluations.

### 6.1 A collapsed sweep point blocked dependent arrays

Ficus at penalty 0.016 raised an error when every Gaussian closed. Slurm then marked the original
post-processing dependency as `DependencyNeverSatisfied`.

**Implemented fix:** the collapse is now a structured negative result. Post-processing skips the
missing PLY, models below a matched target receive `overshot_budget`, and analysis ranks only
eligible exact-count arms. The permanently blocked jobs were replaced while all successful
training outputs were preserved.

### 6.2 Tiny models overflowed the tile-pair heuristic

The original tile-pair buffer used (16N). For models containing 16–84 large Gaussians, individual
Gaussians covered enough tiles to exceed that small buffer.

**Implemented fix:** the automatic capacity is now

$$
C_{\mathrm{pairs}} = \max(16N,\,16N_{\mathrm{tiles}})
$$

per buffered view. At 256×256 resolution this supplies a minimum of 4,096 records. Only the three
failed tasks were rerun; completed CSVs were skipped.

### 6.3 `--skip-existing` skipped required metadata

The zero-pruning importance scorer created `importance_0p00.ply` before the generic arm writer ran.
The existence check consequently skipped both the PLY and its JSON metadata, causing final
analysis to fail even though all 78 metric CSVs existed.

**Implemented fix:** PLY creation and metadata creation now have independent existence checks. A
10-second CPU backfill generated the missing JSON without rerendering any view, after which the
analysis completed successfully.

## 7. Final conclusion

Importance pruning should be the default compaction method for these experiments. It retains
quality surprisingly well even after removing 75% of the unmasked population, and it wins on
Drums and Lego at both exact budgets. Learned-mask training is not uniformly beneficial and has a
narrow stability range. Nevertheless, it should not be discarded: on Ficus, a mild penalty of
0.004 followed by importance pruning gives a large, consistent improvement across PSNR, SSIM, and
LPIPS at both 40% and 20% retention.

The initial hypothesis—that a pruning signal learned jointly with rendering would generally make
better decisions than a post-training importance score—is therefore mostly rejected by these
experiments. In two of three scenes, the learned-mask step lowers quality and also requires extra
full training runs to find a usable, scene-specific penalty. Its benefit on Ficus shows that the
idea can work, but not that its additional computation and trial-and-error search should be the
default workflow.

The most defensible rule from this project is therefore:

> Start with importance pruning. Add a mild learned-mask regularizer only when a short pilot shows
> improved validation quality without a rapid decline in active count. Select or control the mask
> penalty per scene, and treat active-count collapse as a failed configuration rather than as
> successful compaction. If the pilot provides no improvement, skip learned-mask training and avoid
> a full penalty sweep.

This conclusion is narrower than both the original learned-mask advantage hypothesis and the
semantic scene-type hypothesis, but it is better supported by the evidence. A follow-up with
adaptive penalties, multiple seeds, recovery fine-tuning, persistent Gaussian lineage, and more
scenes would determine whether observable training statistics can predict when the combined method
will outperform importance pruning alone.

## 8. Reproducibility artifacts

The aggregate results are stored under `trainer/outputs/project/analysis/`:

- `all_results.csv`: all 78 evaluated arms;
- `matched_comparisons.csv`: exact-count comparisons and metric deltas;
- `excluded_settings.csv`: six budget overshoots and the complete Ficus collapse;
- `gaussian_diagnostics.csv`: opacity, scale, visibility, coverage, importance, and transmittance
  summaries;
- `report.md`: automatically generated concise winner summary.

The main implementation entry points are `trainer/3dgs_trainer.py`,
`trainer/importance_prune.py`, `trainer/scripts/postprocess_model.py`,
`trainer/scripts/analyze_project_results.py`, and `trainer/slurm/`.

## References

1. Z. Fan et al. “LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+
   FPS.” *NeurIPS*, 2024. <https://arxiv.org/abs/2311.17245>
2. W. Tsang. “Unit 14: Compaction of 3DGS.” *CP4282 Gaussian Splatting course notes*.
   <https://weitsang.github.io/gaussian-splatting-notes/unit-14/>
