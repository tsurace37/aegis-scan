# aegis-scan

Open-source detection of data poisoning and backdoor attacks in ML classification pipelines, validated across a healthcare-imaging benchmark and a general-purpose benchmark.

**Status: all 8 pipeline stages implemented and tested.** Dataset loading, synthetic poison injection, classifier training, activation extraction, detection, fusion, evaluation, and reporting are all below, end to end, on the synthetic dataset -- confirming against the real healthcare/benchmark loaders on a machine with normal internet access is the next step.

This project is the empirical companion to [*Toward Automated Detection of Data Poisoning and Backdoor Attacks in Healthcare Imaging AI*](https://doi.org/10.5281/zenodo.22431042) (Zenodo, DOI 10.5281/zenodo.22431042), which specifies the methodology this code implements.

## Why

Healthcare organizations are deploying AI-enabled diagnostic tools faster than they can independently verify those models haven't been tampered with. Governance frameworks like ISO/IEC 42001 tell you an organization *has a policy* for reviewing its AI systems; they don't tell you whether a specific deployed model was actually tested against data poisoning or backdoor triggers. This project is the missing technical-testing layer: it combines two established detection techniques -- spectral signature analysis ([Tran et al., 2018](https://arxiv.org/abs/1811.00636)) and activation clustering ([Chen et al., 2018](https://arxiv.org/abs/1811.03728)) -- and reports results against [MITRE ATLAS](https://atlas.mitre.org/) technique IDs and the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)'s *Measure* function, so findings are legible to a security team without translation.

## Pipeline

1. **Load datasets** -- a healthcare-imaging benchmark and a non-healthcare benchmark, in a common format, to test cross-sector generalization. *(implemented)*
2. **Inject synthetic poison** -- stamp a backdoor trigger onto a subset of images at a configurable rate, which also produces the ground-truth labels used at step 7. *(implemented)*
3. **Train a classifier** -- a compact ResNet on the poisoned data. *(implemented)*
4. **Extract activations** -- forward hooks capture intermediate-layer activations. *(implemented)*
5. **Detect** -- spectral signature analysis and activation clustering, run independently on those activations. *(implemented)*
6. **Fuse scores** -- combine both methods into a per-sample and model-level risk score. *(implemented)*
7. **Evaluate** -- score detection accuracy (TPR/FPR) against step 2's ground truth. *(implemented)*
8. **Report** -- map findings to MITRE ATLAS and NIST AI RMF. *(implemented)*

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Load PneumoniaMNIST (a lightweight chest X-ray benchmark standing in for the full ChestX-ray14 dataset during development) and poison 5% of it:

```bash
aegis-scan inject --dataset healthcare --rate 0.05 --out data/poisoned_healthcare_5pct.npz
```

Do the same against CIFAR-10, the non-healthcare benchmark:

```bash
aegis-scan inject --dataset benchmark --rate 0.05 --out data/poisoned_benchmark_5pct.npz
```

Each run prints how many samples were actually poisoned and saves a `.npz` file containing the (possibly) altered images, their (possibly flipped) labels, and the `poison_mask` ground truth -- which samples were really altered -- for later stages to train on and evaluate against.

Train a classifier on the poisoned output, then extract its layer-3 activations for stage 05's detectors to consume next:

```bash
aegis-scan train --data data/poisoned_healthcare_5pct.npz --epochs 10 --out models/healthcare_5pct.pt
aegis-scan extract-activations --model models/healthcare_5pct.pt --data data/poisoned_healthcare_5pct.npz --layer layer3 --out data/activations_healthcare_5pct.npz
```

`train` prints the final epoch's loss/accuracy and saves a checkpoint (weights plus the `in_channels`/`num_classes` needed to reload it). `extract-activations` reloads that checkpoint, runs every sample back through it, and saves the named layer's per-sample activation vectors -- pass any layer name from `SmallResNet` (`layer1`, `layer2`, `layer3` by default, or a deeper path like `layer3.conv2`).

Score those activations with both detection methods:

```bash
aegis-scan detect --data data/poisoned_healthcare_5pct.npz --activations data/activations_healthcare_5pct.npz --out data/detect_healthcare_5pct.npz
```

`detect` runs spectral signature analysis and activation clustering independently, per class, on the activations -- both methods are blind to stage 02's `poison_mask`; neither one sees it. It saves each sample's spectral outlier score and its activation-clustering minority-cluster flag, for stage 06 to combine and stage 07 to score against the ground truth.

Combine both detectors' outputs into one score:

```bash
aegis-scan fuse --detect data/detect_healthcare_5pct.npz --out data/fuse_healthcare_5pct.npz
```

`fuse` prints a mean fused score plus two model-level statistics: the fraction of samples flagged by *either* detector (higher recall, more permissive), and the fraction flagged by *both* (the headline risk number -- two differently-motivated methods agreeing on the same sample is much stronger evidence than either alone). Still entirely blind to the ground truth; stage 07 is where that finally gets checked.

Finally, check how accurate all of that actually was, against the ground truth that every stage up to this point was never shown:

```bash
aegis-scan evaluate --inject data/poisoned_healthcare_5pct.npz --detect data/detect_healthcare_5pct.npz --fuse data/fuse_healthcare_5pct.npz --out data/evaluate_healthcare_5pct.json
```

`evaluate` is the one command in the whole pipeline allowed to look at `poison_mask`. `--detect` and `--fuse` are both optional (pass whichever outputs you have -- at least one is expected), and it prints TPR/FPR/precision for the two boolean flags (clustering, agreement) plus AUROC/average precision/top-k recall for the two continuous scores (spectral, fused), then saves the full report as JSON.

Last, translate those numbers into a report a security or compliance reviewer can read without learning this project's internals:

```bash
aegis-scan report --evaluate data/evaluate_healthcare_5pct.json --dataset healthcare_5pct --out data/report_healthcare_5pct.md
```

`report` runs no new analysis -- everything in it traces back to a number `evaluate` already computed against ground truth. It picks the strongest available evidence (the fused score, if present) for a one-sentence headline finding, then maps the attack being tested for to MITRE ATLAS technique IDs ([AML.T0020](https://atlas.mitre.org/) Poison Training Data, AML.T0059 Erode Dataset Integrity, AML.T0018 Manipulate AI Model) and the specific NIST AI RMF subcategory this kind of testing satisfies (MEASURE 2.7: "AI system security and resilience... are evaluated and documented"), and saves the result as a self-contained Markdown report.

## Test

```bash
pytest
```

## Design notes

- **Why CIFAR-10 and not CIFAR-10-C:** CIFAR-10-C is a corruption-robustness benchmark (blur, noise, weather), which is a different question from backdoor detection. The papers this project builds on both benchmark against plain CIFAR-10 with an injected trigger, so that's what's used here.
- **Why PneumoniaMNIST for now:** it's the same imaging modality (chest X-ray) and binary framing as the eventual ChestX-ray14 target, but small enough to iterate on quickly while stages 3-8 are being built. Swapping in the full benchmark later only means writing a new loader with the same output shape -- nothing downstream changes.
- **Why only non-target-class samples are eligible for poisoning:** stamping a trigger on a sample that's already the target class doesn't test whether the trigger caused a misclassification, since its label doesn't actually change. This matches how the backdoor-attack literature sets up the experiment.
- **Why a hand-written `SmallResNet` instead of `torchvision.models.resnet18`:** torchvision's ResNets assume 224x224 ImageNet-sized input and downsample 4x in the stem alone -- applied to a 28x28 PneumoniaMNIST image, that leaves almost nothing for the residual blocks to work with. `models/resnet.py` adapts to whatever image size and channel count it's given instead.
- **Why forward hooks instead of changing the model's `forward()`:** stage 04 needs a trained model's intermediate activations without permanently altering how that model behaves as a classifier. A forward hook attaches for the duration of one inference pass and is removed immediately after, so `SmallResNet` stays an ordinary classifier the rest of the time -- and the same extraction code will work against a different architecture later without changes.
- **Why detection runs per class, not on the whole dataset at once:** both methods look for "this class secretly contains an unnatural sub-pattern" -- spectral signatures find a one-directional fingerprint within a class's activations, activation clustering looks for a distinguishable minority sub-population within a class. Pooling every class together would mostly just rediscover the classes themselves, not a poisoning signal.
- **Why activation clustering only flags a cluster below `max_minority_fraction` (35% by default):** k-means with k=2 always returns *some* split, even for a class with no real sub-population -- run it on one uniform blob and it still cuts it roughly in half. Only trusting the smaller cluster when it's an actual minority (comfortably above this project's tested 1-10% poisoning rates, but well below an even 50/50 split) avoids mistaking that artifact for a finding.
- **Why fusion ranks spectral scores by per-class percentile instead of using the raw scores:** spectral signature scores aren't comparable across classes -- each class gets its own SVD and its own scale. Converting to a 0-1 rank within each class first puts every class on the same footing before averaging with the clustering flag, without assuming anything about how many samples are actually poisoned.
- **Why "flagged by both detectors" is the headline risk number, not "flagged by either":** the two methods have different blind spots, so agreement between them is much less likely to happen by chance than either one alone. Testing this on a synthetic 10%-poisoned run bore it out directly: the "both" agreement flag had zero false positives (though it caught only 58 of the 100 poisoned samples), while ranking all samples by the continuous fused score put 99 of the 100 poisoned samples in the top 100 -- precision and recall trade off differently depending on which output you read.
- **Why `evaluate` rescales spectral scores by per-class percentile before scoring them, instead of using the raw scores directly:** a single global AUROC computed on raw spectral scores would be meaningless, for the same reason fusion can't average them directly -- each class gets its own SVD and its own arbitrary scale in stage 05. Reusing `fuse.py`'s `percentile_rank_per_class` here (rather than duplicating that logic) means the spectral numbers reported by `evaluate` are on the exact same footing as the ones that went into the fused score.
- **Why `evaluate`'s metrics return NaN instead of raising when a class is undefined:** precision is undefined when nothing was flagged (0/0), and AUROC/average precision are undefined when every sample -- or no sample -- is truly poisoned (there's no "other class" to rank against). Both are real situations a synthetic run at an extreme poisoning rate can hit; NaN propagates that "not applicable here" signal instead of forcing a crash or a misleading 0.
- **Why `top_k_recall` is reported alongside AUROC/average precision:** AUROC and AP are the right metrics for comparing detectors in the abstract, but neither answers a concrete question a reviewer will actually ask: "if I flag as many samples as are truly poisoned, how many do I actually catch?" `top_k_recall` answers exactly that, in the same terms used for the manual sanity checks run during development (e.g. "99 of the 100 poisoned samples were in the top 100 by fused score").
- **Why `evaluate --detect` and `--fuse` are both optional (but at least one is expected):** stage 07 is meant to be run against whatever's available -- just stage 05's raw outputs, just stage 06's fused ones, or (typically) both side by side. Requiring both would make it impossible to spot-check stage 05 in isolation before fusion is even run.
- **Why the MITRE ATLAS / NIST AI RMF mapping was verified against primary sources, not summarized from blog posts:** the technique IDs (AML.T0020, AML.T0059, AML.T0018) were checked directly against MITRE ATLAS's own technique data, and the MEASURE 2.7 wording was quoted directly from NIST AI 100-1 -- because a wrong citation in a security report is worse than an absent one, and secondhand summaries of these frameworks have been found to drift from the primary text elsewhere in this project already.
- **Why `report` doesn't compute a single overall risk score:** collapsing TPR, FPR, AUROC, and top-k recall into one number would hide exactly the nuance stage 06/07's own testing surfaced -- that "flagged by both detectors" and "the continuous fused score" trade off precision and recall very differently. The report shows every metric that was supplied and states a headline finding in plain language, but leaves risk tolerance (how much residual FPR/FNR is acceptable) to the reader, since that's an organizational policy decision this tool has no basis to make for them.

Apache-2.0. See `LICENSE`.

## Author

Suresh Tamang -- Graduate Researcher, Artificial Intelligence, University of the Cumberlands.
