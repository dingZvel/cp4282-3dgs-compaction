# Presentation Script: Scene-Dependent 3DGS Compaction

Estimated length: 13–15 minutes, excluding optional questions

## Slide 1 — When learned masks help—and when they collapse

**Target time: 45 seconds**

Good morning. My project investigates compaction in 3D Gaussian Splatting, specifically whether
we should combine learned-mask pruning during training with importance pruning after training.

The central question is not just whether we can remove Gaussians. We already know that we can. The
question is whether different scenes benefit from different combinations of these two methods when
we compare them at exactly the same final number of Gaussians.

I tested three NeRF Synthetic scenes: Drums, Ficus, and Lego. In total, I trained 15 configurations
and evaluated 78 compacted models. Every final metric is averaged over the same 200 untouched test
views.

The short conclusion is in the title: a learned mask helps one scene, but stronger mask pressure
can also cause the entire representation to collapse.

**Transition:** First, I will explain how I made the comparison fair.

---

## Slide 2 — Which scenes benefit from combining the two pruning directions?

**Target time: 105 seconds**

The research question is: at the same final Gaussian count, does mask-first training improve
novel-view quality over importance pruning alone?

My initial hypothesis was that learned masks should generally do a better job than post-training
importance pruning. The mask variables are optimized jointly with the rendering objective, so I
expected them to learn which Gaussians the representation could discard, rather than relying only
on a contribution score computed after training.

I also expected scene structure to control the size of that advantage. A dense, self-occluded
scene might still favour importance pruning; a scene with thin or rarely visible structures might
benefit more from learned-mask optimization; and a mixed scene might prefer a balanced combination.

The experiment has five steps.

First, I train every model for 20,000 iterations with the same seed, resolution, optimizer, and
densification schedule.

Second, I sweep the learned-mask penalty from no mask through 0.002, 0.004, 0.008, and 0.016.

Third, after training, I score each surviving Gaussian using its accumulated blending contribution
over the training views. This is the sum of transmittance times alpha, or \(T_i\alpha_i\), over
pixels and cameras.

Fourth, I prune the models to exact common budgets: 120,000 Gaussians for 40 percent retention and
60,000 for 20 percent retention. Exact counts are important because otherwise a smaller model
could appear better simply because it kept more Gaussians.

Finally, I evaluate PSNR, SSIM, and LPIPS on all 200 test views. Higher PSNR and SSIM are better;
lower LPIPS is better.

**Point to:** the five-step pipeline, especially the exact-budget step.

**Transition:** With that protocol, the matched-count results show a clear split between scenes.

---

## Slide 3 — The winning direction depends on the scene

**Target time: 105 seconds**

This chart compares all eligible mask settings at the same final count. I can switch between the
40 and 20 percent budgets, and between PSNR, SSIM, and LPIPS.

Let me begin at 40 percent retention using PSNR.

For Drums, the importance-only point is the best. Every mask-trained alternative is to its left,
which means lower PSNR.

Ficus is different. The 0.004 mask penalty is clearly the best. It reaches 22.45 dB, compared with
21.57 dB for importance pruning alone.

For Lego, importance-only pruning wins by a large margin. The mask-trained models become worse as
the penalty increases.

If I switch to 20 percent retention, the same scene-level ordering remains. Ficus with penalty
0.004 still wins, while Drums and Lego still favour importance pruning alone.

The conclusion is also consistent across SSIM and LPIPS, not just PSNR. Ficus 0.004 wins all three
metrics at both budgets. The importance-only model wins all three metrics for Drums and Lego.

So the method direction is genuinely scene-dependent, but only one of the three scenes benefits
from the combination.

**Interaction:** briefly switch from 40% to 20%, then from PSNR to LPIPS. Return to PSNR before
advancing.

**Transition:** The next slide shows the size of the difference relative to the importance-only
baseline.

---

## Slide 4 — Only Ficus gains from mask-first training

**Target time: 120 seconds**

These bars show the PSNR change relative to importance-only pruning at the same final Gaussian
count. Values to the right are improvements; values to the left are regressions.

For Drums, even the best mask alternative is about 0.50 dB worse at 40 percent retention and 0.37
dB worse at 20 percent.

For Lego, the difference is much larger. Even the weakest eligible mask penalty loses about 1.47
dB at 40 percent and 2.12 dB at 20 percent. Stronger mask penalties perform even worse.

Ficus is the only positive result. Penalty 0.004 gains 0.88 dB at 40 percent retention and 1.04 dB
at 20 percent. It also improves SSIM and lowers LPIPS.

There is one important interpretation detail. The Ficus advantage already exists before
importance pruning. The unpruned 0.004 mask-trained model scores about 22.46 dB, compared with
21.60 dB for the unmasked model. After pruning to 120,000 Gaussians it still scores 22.45 dB, and
after pruning to 60,000 it scores 22.31 dB.

So the learned mask appears to improve the Ficus training trajectory, while importance pruning
preserves that improved representation. This does not prove that learned-mask deletion alone found
a better final subset.

The practical result is less favourable to the initial hypothesis. In two of the three scenes,
Drums and Lego, the better choice was to skip learned-mask training altogether. Importance-only
pruning produced better matched-budget quality and avoided the extra computation of retraining
several mask variants while tuning lambda by trial and error. A parameter being learned does not
guarantee a better pruning decision when its penalty changes the optimization trajectory or is
poorly calibrated for the scene.

**Point to:** the positive Ficus bar, followed by the Ficus finding card.

**Transition:** Before looking at the penalty failure, these representative test views show what
the average metric differences look like in the rendered images.

---

## Slide 5 — What the metric differences look like

**Target time: 120 seconds**

This slide shows representative test cameras from each scene at both evaluated budgets: 40 percent,
or 120,000 Gaussians, in the upper row of each card; and 20 percent, or 60,000 Gaussians, in the
lower row. Every row compares the ground truth, importance-only pruning, and the best eligible
mask-trained alternative.

For each scene and budget, I selected the camera whose PSNR difference is closest to that
scene-budget pair's 200-view mean difference. These are therefore typical examples rather than the
most dramatic successes or failures.

At 40 percent retention, Drums view 89 falls from 20.38 to 19.90 dB with mask 0.008, a loss of
0.48 dB. Ficus view 119 rises from 20.81 to 21.68 dB with mask 0.004, a gain of 0.87 dB. Lego view
189 falls from 26.05 to 24.58 dB with mask 0.002, a loss of 1.48 dB.

At 20 percent retention, Drums view 22 reaches 22.29 dB with importance-only pruning. The best mask alternative shown here,
penalty 0.008 followed by importance pruning, reaches 21.91 dB. The mask-trained result loses some
of the fine hardware and cymbal detail, matching the average negative result for Drums.

For Ficus view 71, importance-only pruning reaches 20.53 dB, while mask 0.004 followed by
importance pruning reaches 21.57 dB. The improvement is about 1.04 dB, almost exactly the mean
improvement over all test cameras. The combined model better preserves the plant structure and
produces a cleaner overall reconstruction.

For Lego view 14, importance-only pruning reaches 26.98 dB, while the weakest mask alternative,
penalty 0.002, reaches 24.85 dB. The loss is about 2.12 dB and is visible around the fine mechanical
structure, tracks, and object boundaries.

Across both budgets, the images support the same metric-level conclusion: the mask-first direction
is helpful for Ficus, but it introduces visible degradation for Drums and especially Lego. The
direction of the result is stable as the budget changes, while the size of the gap changes.

**Point to:** first compare the upper 40 percent rows across scenes, then the lower 20 percent rows;
within each row move left to right from ground truth, to importance-only, to mask plus importance.

**Transition:** The next result explains why choosing the mask penalty is difficult even when a
mild setting helps.

---

## Slide 6 — The useful range ends at a cliff

**Target time: 90 seconds**

This chart shows the final number of active Gaussians after training as the mask penalty increases.
The dashed lines are the intended 40 and 20 percent budgets.

For Drums and Lego, penalties up to 0.008 finish at the full 300,000-Gaussian capacity. At 0.016,
they suddenly collapse to only 16 and 84 Gaussians.

Ficus is more sensitive. It already collapses to 39 Gaussians at penalty 0.008, and at 0.016 it
closes every Gaussian. That run reaches zero at about iteration 12,300.

This means the mask penalty is not a reliable open-loop control for final model size. The sweep
does not give a smooth progression from 300,000 to 120,000 to 60,000. It stays near capacity and
then falls off a cliff.

This also changes the original scene hypothesis. Ficus does benefit from a mild learned mask, but
it does not tolerate a stronger mask. In fact, it is the first scene to become unstable.

The practical fix would be an adaptive penalty or target-count controller. Another option is a
two-stage sweep: use short pilot runs to locate the transition, then sample that narrow interval
more densely.

**Point to:** the 120k and 60k target lines, then the Ficus drop at 0.008.

**Transition:** This instability matters because importance pruning alone is already a very strong
baseline.

---

## Slide 7 — Importance pruning removes 75% with less than 0.3 dB loss

**Target time: 75 seconds**

This slope chart compares the unmasked model at 300,000 Gaussians with the same model after
importance pruning to 75,000 Gaussians.

That is a four-times reduction in Gaussian count.

Drums loses only 0.20 dB. Ficus also loses about 0.20 dB. Lego loses about 0.29 dB. All three
remain within 0.3 dB of their unpruned PSNR.

At the stricter exact 20 percent budget of 60,000 Gaussians, importance-only pruning still loses
only 0.42 dB on Drums, 0.33 dB on Ficus, and 0.61 dB on Lego.

This explains why learned-mask training has a high bar to clear. Importance ranking already finds
a large redundant population and removes it with little immediate damage. A learned mask must
improve the representation itself—not merely remove some Gaussians—to outperform this baseline.

That is what appears to happen for Ficus at 0.004, but not for Drums or Lego.

**Point to:** the small vertical change in all three slope lines.

**Transition:** The negative and failed runs also revealed several limitations in both the science
and the implementation.

---

## Slide 8 — Negative results improved the protocol

**Target time: 95 seconds**

There are four main scientific limitations.

First, the penalty sweep is too coarse near the collapse transition. A denser or adaptive sweep is
needed.

Second, I used one seed. The Ficus gain is large and consistent across metrics, but multiple seeds
are required to establish repeatability and confidence intervals.

Third, I did not run recovery fine-tuning. These results measure immediate pruning damage. A fair
follow-up should fine-tune every compact model for the same number of iterations, with additional
pruning and densification disabled.

Fourth, three synthetic scenes are not enough to establish a general rule. The experiment should
be extended to more synthetic scenes and real captures, ideally holding out entire scenes when
testing a method-selection rule.

There were also two engineering failures.

The zero-Gaussian Ficus run originally blocked the downstream Slurm dependency. I fixed this by
recording collapse as a structured negative result and excluding undersized models from matched-
budget ranking.

The tiny surviving models overflowed a tile-pair buffer because the old capacity heuristic became
too small. I added a resolution-aware minimum.

The key point is that completed work was preserved: only failed or missing steps were rerun.

**Point to:** scientific limitations on the left, then engineering fixes on the right.

**Transition:** These results lead to a more cautious decision rule than the original hypothesis.

---

## Slide 9 — Importance first; mask only with evidence

**Target time: 100 seconds**

The final recommendation is to start with importance pruning. It is stable, simple, and performs
very well across all three scenes. In this experiment, learned masks justified their additional
training and scene-specific lambda search only for Ficus.

Then run a short learned-mask pilot. Monitor both validation quality and the active Gaussian count.

If validation improves and the population remains stable, as it did for Ficus with penalty 0.004,
then combine mask training with importance pruning.

If quality falls or the active count begins collapsing, as it did for Drums, Lego, and the stronger
Ficus settings, stay with importance pruning alone. This avoids spending full training runs on a
trial-and-error lambda sweep that may produce lower quality or an unusable collapsed model.

So the main conclusion is not that one semantic scene label directly tells us which method to use.
A scene label is not enough. The penalty should be selected from measured training dynamics and a
target retention constraint.

The next steps are an adaptive penalty, multiple seeds, equal recovery fine-tuning, persistent
Gaussian lineage for unique deletion-set comparisons, and evaluation on more scenes.

To summarize in one sentence: importance pruning is the safe default; a mild learned mask can
help, but only when the scene provides evidence that it improves the representation without
triggering collapse.

Thank you.

**Point to:** the decision flow from importance pruning, to pilot, to the two outcomes.

---

## Optional questions and answers

### Why were the collapsed models excluded from the matched-budget ranking?

They contained fewer Gaussians than the requested 60,000 or 120,000. We cannot add Gaussians back,
and forcing every other model down to 16, 39, or 84 Gaussians would answer a different question.
The collapsed settings are retained as negative results instead of being silently discarded.

### Why use accumulated \(T_i\alpha_i\) as importance?

It measures how much a Gaussian contributes to alpha compositing across training pixels and views.
It therefore incorporates visibility, opacity, transmittance, and projected coverage rather than
using opacity alone.

### Does Ficus prove that learned-mask pruning is better for thin structures?

No. It is one positive scene under one seed. It shows that mild mask regularization can improve the
Ficus training trajectory, but it does not establish a general rule for thin structures. More
scenes and seeds are required.

### Why is there no recovery step?

Recovery was optional in the proposal and was omitted to keep the initial experiment tractable.
The present result measures immediate pruning safety. Equal recovery runs are a high-priority
follow-up because different retained populations may have different adaptation capacity.

### Why can learned-mask deletion-event counts exceed the final population?

Densification can reuse a pruned slot, and that new Gaussian can be pruned again later. The logs
count deletion events, not unique Gaussian identities. Persistent lineage IDs are needed for a
unique removal-set overlap analysis.
