# aegis-scan

Open-source detection of data poisoning and backdoor attacks in ML classification pipelines, validated across a healthcare-imaging benchmark and a general-purpose benchmark.

**Status: all 8 pipeline stages implemented and tested, with published results.** Dataset loading, synthetic poison injection, classifier training, activation extraction, detection, fusion, evaluation, and reporting all run end to end against the real PneumoniaMNIST and CIFAR-10 loaders -- a full 40-configuration grid (2 datasets x 4 poison rates x 5 seeds) has been run and its results are reported in the paper below.

This project is the open-source implementation accompanying [*Fusing Spectral Signatures and Activation Clustering for Backdoor Detection in Healthcare Imaging Models: Method, Implementation, and Evaluation*](https://arxiv.org/abs/2609.14290) (arXiv:2609.14290, cs.CR/cs.LG; also on Zenodo, DOI [10.5281/zenodo.22732363](https://doi.org/10.5281/zenodo.22732363)), which this code implements and whose results were produced by this pipeline.

## Why

Healthcare organizations are deploying AI-enabled diagnostic tools faster than they can independently verify those models haven't been tampered with. Governance frameworks like ISO/IEC 42001 tell you an organization *has a policy* for reviewing its AI systems; they don't tell you whether a specific deployed model was actually tested against data poisoning or backdoor triggers. This project is the missing technical-testing layer: it combines two established detection techniques -- spectral signature analysis ([Tran et al., 2018](https://arxiv.org/abs/1811.00636)) and activation clustering ([Chen et al., 2018](https://arxiv.org/abs/1811.03728)) -- and reports results against [MITRE ATLAS](https://atlas.mitre.org/) technique IDs and the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)'s *Measure* function, so findings are legible to a security team without translation.

## Why the name

An *aegis* was the shield carried by Zeus and Athena in Greek mythology -- famously said to bear the head of Medusa on it to ward off attackers -- and the word has since come to mean protection or defense more generally (as in "under the aegis of"). It's also the name of the U.S. Navy's Aegis Combat System, the shield-and-radar defense system that protects ships from incoming missiles. `aegis-scan` is the same idea applied to an ML pipeline: it scans a model's training data and activations for the hidden threats a defensive shield would need to catch.

## Pipeline

1. **Load datasets** -- a healthcare-imaging benchmark and a non-healthcare benchmark, in a common format, to test cross-sector generalization; plus a generic `custom` loader for any other dataset already packaged into that same format. *(implemented)*
2. **Inject synthetic poison** -- stamp a backdoor trigger onto a subset of images at a configurable rate, which also produces the ground-truth labels used at step 7. *(implemented)*
3. **Train a classifier** -- a compact ResNet on the poisoned data. *(implemented)*
4. **Extract activations** -- forward hooks capture intermediate-layer activations. *(implemented)*
5. **Detect** -- spectral signature analysis and activation clustering, run independently on those activations. *(implemented)*
6. **Fuse scores** -- combine both methods into a per-sample and model-level risk score. *(implemented)*
7. **Evaluate** -- score detection accuracy (TPR/FPR) against step 2's ground truth. *(implemented)*
8. **Report** -- map findings to MITRE ATLAS, NIST AI RMF, and (flagged unverified) ISO/IEC 42001. *(implemented)*

## Install

```bash
pip install aegis-scan-ai
```

This installs the `aegis-scan` command (all commands below use that name -- only the PyPI package name is `aegis-scan-ai`; the CLI, repo, and everything else stays `aegis-scan`. The PyPI name `aegis-scan` itself was already taken by an unrelated project when this one went to publish).

To install from source instead (e.g. to run the test suite or make changes):

```bash
git clone https://github.com/tsurace37/aegis-scan
cd aegis-scan
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

Or run the pipeline against your own dataset, once it's packaged into the same shape (see **Custom datasets** below):

```bash
aegis-scan inject --dataset custom --data-path data/my_dataset.npz --rate 0.05 --out data/poisoned_custom_5pct.npz
```

Train a classifier on the poisoned output, then extract its layer-3 activations for stage 05's detectors to consume next:

```bash
aegis-scan train --data data/poisoned_healthcare_5pct.npz --epochs 10 --out models/healthcare_5pct.pt
aegis-scan extract-activations --model models/healthcare_5pct.pt --data data/poisoned_healthcare_5pct.npz --layer layer3 --out data/activations_healthcare_5pct.npz
```

`train` prints the final epoch's loss/accuracy and saves a checkpoint (weights, `in_channels`/`num_classes`, and which architecture was used, all needed to reload it). It defaults to `SmallResNet` (`--arch small_resnet`); pass `--arch resnet18` to train a CIFAR-adapted `torchvision.models.resnet18` instead -- see **Architectures** below. `extract-activations` reloads that checkpoint, runs every sample back through it, and saves the named layer's per-sample activation vectors -- pass any layer name from the architecture it was trained with (`layer1`, `layer2`, `layer3` by default for `small_resnet`; `layer1`-`layer4` for `resnet18`), or a deeper path like `layer3.conv2`.

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

`report` runs no new analysis -- everything in it traces back to a number `evaluate` already computed against ground truth. It picks the strongest available evidence (the fused score, if present) for a one-sentence headline finding, then maps the attack being tested for to MITRE ATLAS technique IDs ([AML.T0020](https://atlas.mitre.org/) Poison Training Data, AML.T0059 Erode Dataset Integrity, AML.T0018 Backdoor ML Model), the specific NIST AI RMF subcategory this kind of testing satisfies (MEASURE 2.7: "AI system security and resilience... are evaluated and documented"), and a related ISO/IEC 42001 Annex A control (A.6.2.4, AI system verification and validation) -- the last one explicitly labeled unverified in the report, since ISO/IEC 42001's own text is paywalled and wasn't consulted (see **Design notes** below) -- and saves the result as a self-contained report.

The output format is picked from `--out`'s file extension: anything ending in `.html` renders a styled, self-contained HTML file (open it in any browser, then File > Print > Save as PDF -- no PDF library or system dependency required); any other extension (`.md`, or none) renders Markdown, unchanged from before. Both formats show exactly the same numbers and section order -- only the presentation differs.

```bash
aegis-scan report --evaluate data/evaluate_healthcare_5pct.json --dataset healthcare_5pct --out data/report_healthcare_5pct.html
```

## Custom datasets

Every stage from 02 onward already works on any dataset, because they're all written against one shape (`PoisonableDataset`: float32 images in `(N, C, H, W)`, values in `[0, 1]`, plus integer labels) -- stage 01's loaders are the only dataset-specific code in the pipeline. `load_custom_dataset` is the generic escape hatch: preprocess your own data once into a `.npz` file with `images` and `labels` arrays in that shape, then point `inject` at it:

```bash
aegis-scan inject --dataset custom --data-path data/my_dataset.npz --rate 0.05 --out data/poisoned_custom_5pct.npz
```

`--data-path` is required (and only used) when `--dataset custom` is chosen; the built-in loaders ignore it. A malformed file (missing keys, wrong shape, wrong dtype) fails immediately with a specific message rather than a confusing crash several stages later -- `PoisonableDataset`'s own validation catches it.

This intentionally stops at "already-formatted npz," not a general-purpose image/label folder importer: writing that conversion script once for your own data source is a small one-time cost, and keeping the loader itself simple means every downstream stage's behavior stays exactly as tested, whatever produced the npz.

## Architectures

`train` supports more than one model architecture via `--arch`:

- `small_resnet` (default) -- the hand-written, compact ResNet this project's own published results were trained with. Adapts to whatever image size and channel count it's given (28x28 grayscale chest X-rays, 32x32 RGB CIFAR-10, or anything else), via an adaptive average pool instead of assuming a fixed input size.
- `resnet18` -- `torchvision.models.resnet18`, with its stem adapted for small images (a 3x3 stride-1 first conv and no initial max-pool, instead of the stock 7x7 stride-2 conv + max-pool that assumes 224x224 ImageNet-sized input and would otherwise shrink a 28x28-32x32 image to almost nothing before the first residual block). This is the standard CIFAR-ResNet stem adaptation, not a bespoke one.

Both are looked up through one registry (`models/registry.py`), so `train`/`load_checkpoint` aren't hardwired to either -- adding a third architecture later means registering a new build function there, nothing else. A checkpoint records which architecture trained it, so `load_checkpoint` always reconstructs the right one automatically; checkpoints saved before this registry existed have no such record and are assumed to be `small_resnet` (the only option that existed at the time), so every checkpoint this project has already produced -- including the ones behind the paper's published results -- keeps loading unchanged.

## Test

```bash
pytest
```

## Releasing (publishing a new version to PyPI)

Publishing is automated via GitHub Actions using PyPI's "trusted publishing" (OIDC) -- there is no API token stored anywhere, in this repo or elsewhere. This needs a one-time setup step before the first release:

1. On [pypi.org](https://pypi.org), under your account's **Publishing** settings, add a new "pending trusted publisher" with: PyPI project name `aegis-scan-ai`, repository owner `tsurace37`, repository name `aegis-scan`, workflow filename `publish.yml`, and environment name `pypi`. (The project doesn't need to already exist on PyPI for this -- a pending publisher creates it on the first successful publish.)
2. Bump the `version` field in `pyproject.toml`.
3. Create a new GitHub Release (Releases -> Draft a new release -> tag it, e.g. `v0.1.0`) and publish it.
4. That triggers `.github/workflows/publish.yml`, which builds the package and publishes it to PyPI automatically. Check the Actions tab if it doesn't show up on PyPI within a few minutes.

## Design notes

- **Why CIFAR-10 and not CIFAR-10-C:** CIFAR-10-C is a corruption-robustness benchmark (blur, noise, weather), which is a different question from backdoor detection. The papers this project builds on both benchmark against plain CIFAR-10 with an injected trigger, so that's what's used here.
- **Why PneumoniaMNIST for now:** it's the same imaging modality (chest X-ray) and binary framing as the eventual ChestX-ray14 target, but small enough to iterate on quickly while stages 3-8 are being built. Swapping in the full benchmark later only means writing a new loader with the same output shape -- nothing downstream changes.
- **Why only non-target-class samples are eligible for poisoning:** stamping a trigger on a sample that's already the target class doesn't test whether the trigger caused a misclassification, since its label doesn't actually change. This matches how the backdoor-attack literature sets up the experiment.
- **Why `small_resnet` is still the default now that `resnet18` also exists:** it's the architecture this project's own published results were trained and validated with, so keeping it the default means every existing command and checkpoint keeps behaving exactly as documented in the paper unless `--arch` is passed explicitly. `resnet18` exists to prove stage 03/04 aren't hardwired to one bespoke network (see **Architectures** above), not to replace the validated default.
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
- **Why the ISO/IEC 42001 mapping is explicitly marked unverified, unlike the two mappings above:** ISO/IEC 42001 is a paywalled standard, so the control this report points to (A.6.2.4, "AI system verification and validation") was identified from public secondary sources describing the standard's Annex A control list, not from the standard's own wording -- no clause text is quoted anywhere in this project. The report says so directly rather than implying a verification that wasn't actually done; treat it as a pointer to check against your own copy, not a citation you can rely on as-is.
- **Why `report` doesn't compute a single overall risk score:** collapsing TPR, FPR, AUROC, and top-k recall into one number would hide exactly the nuance stage 06/07's own testing surfaced -- that "flagged by both detectors" and "the continuous fused score" trade off precision and recall very differently. The report shows every metric that was supplied and states a headline finding in plain language, but leaves risk tolerance (how much residual FPR/FNR is acceptable) to the reader, since that's an organizational policy decision this tool has no basis to make for them.

Apache-2.0. See `LICENSE`.

## Author

Suresh Tamang -- Graduate Researcher, Artificial Intelligence, University of the Cumberlands.
