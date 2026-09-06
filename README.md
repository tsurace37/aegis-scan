# aegis-scan

Open-source detection of data poisoning and backdoor attacks in ML classification pipelines, validated across a healthcare-imaging benchmark and a general-purpose benchmark.

**Status: early development.** Stages 01-04 (dataset loading, synthetic poison injection, classifier training, activation extraction) are implemented and tested below. Stages 05-08 (detection, fusion, evaluation, and reporting) are in progress.

This project is the empirical companion to [*Toward Automated Detection of Data Poisoning and Backdoor Attacks in Healthcare Imaging AI*](https://doi.org/10.5281/zenodo.22431042) (Zenodo, DOI 10.5281/zenodo.22431042), which specifies the methodology this code implements.

## Why

Healthcare organizations are deploying AI-enabled diagnostic tools faster than they can independently verify those models haven't been tampered with. Governance frameworks like ISO/IEC 42001 tell you an organization *has a policy* for reviewing its AI systems; they don't tell you whether a specific deployed model was actually tested against data poisoning or backdoor triggers. This project is the missing technical-testing layer: it combines two established detection techniques -- spectral signature analysis ([Tran et al., 2018](https://arxiv.org/abs/1811.00636)) and activation clustering ([Chen et al., 2018](https://arxiv.org/abs/1811.03728)) -- and reports results against [MITRE ATLAS](https://atlas.mitre.org/) technique IDs and the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)'s *Measure* function, so findings are legible to a security team without translation.

## Pipeline

1. **Load datasets** -- a healthcare-imaging benchmark and a non-healthcare benchmark, in a common format, to test cross-sector generalization. *(implemented)*
2. **Inject synthetic poison** -- stamp a backdoor trigger onto a subset of images at a configurable rate, which also produces the ground-truth labels used at step 7. *(implemented)*
3. **Train a classifier** -- a compact ResNet on the poisoned data. *(implemented)*
4. **Extract activations** -- forward hooks capture intermediate-layer activations. *(implemented)*
5. **Detect** -- spectral signature analysis and activation clustering, run independently on those activations. *(not yet implemented)*
6. **Fuse scores** -- combine both methods into a per-sample and model-level risk score. *(not yet implemented)*
7. **Evaluate** -- score detection accuracy (TPR/FPR) against step 2's ground truth. *(not yet implemented)*
8. **Report** -- map findings to MITRE ATLAS and NIST AI RMF. *(not yet implemented)*

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

## License

Apache-2.0. See `LICENSE`.

## Author

Suresh Tamang -- Graduate Researcher, Artificial Intelligence, University of the Cumberlands.
