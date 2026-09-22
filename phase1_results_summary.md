# Phase 1 grid results — for Table 1

Full 40-config grid (2 datasets x 4 poison rates x 5 seeds), run via
`tooling/run_grid_healthcare.sh` and `tooling/run_grid_cifar.sh`.

## 1. Table rows (markdown)

| Dataset | Rate | ASR | Clean acc | Spectral AUROC | Clustering TPR | Fused AUROC | Agreement precision |
|---|---|---|---|---|---|---|---|
| PneumoniaMNIST | 0% | 0.029 ± 0.033 | 0.866 ± 0.030 | n/a | n/a | n/a | n/a |
| PneumoniaMNIST | 1% | 0.989 ± 0.017 | 0.869 ± 0.019 | 0.986 ± 0.000 | 0.996 ± 0.010 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| PneumoniaMNIST | 5% | 0.999 ± 0.001 | 0.862 ± 0.027 | 0.940 ± 0.001 | 0.998 ± 0.002 | 0.999 ± 0.001 | 1.000 ± 0.000 |
| PneumoniaMNIST | 10% | 1.000 ± 0.000 | 0.756 ± 0.162 | 0.894 ± 0.004 | 0.995 ± 0.004 | 0.996 ± 0.004 | 1.000 ± 0.000 |
| CIFAR-10 | 0% | 0.039 ± 0.026 | 0.779 ± 0.008 | n/a | n/a | n/a | 0.000 (n=2/5) |
| CIFAR-10 | 1% | 0.917 ± 0.029 | 0.768 ± 0.013 | 0.934 ± 0.004 | 0.916 ± 0.012 | 0.968 ± 0.005 | 1.000 ± 0.000 |
| CIFAR-10 | 5% | 0.965 ± 0.008 | 0.762 ± 0.020 | 0.811 ± 0.005 | 0.960 ± 0.005 | 0.963 ± 0.005 | 1.000 ± 0.000 |
| CIFAR-10 | 10% | 0.972 ± 0.004 | 0.762 ± 0.018 | 0.545 ± 0.007 | 0.000 ± 0.000 | 0.545 ± 0.007 | n/a |

**Note:** at 10% poisoning, CIFAR-10's activation clustering collapses to
0% TPR and spectral drops to near-chance AUROC (0.545), despite ASR
confirming the backdoor is learned (97.2%). Consistent with the
documented `max_minority_fraction` constraint in clustering.py -- worth
reporting as a real limitation, not a bug.

## LaTeX rows

```
CIFAR-10 & 0\% & 0.039 $\pm$ 0.026 & 0.779 $\pm$ 0.008 & n/a & n/a & n/a & 0.000 $\pm$ 0.000 (n=2/5) \\
CIFAR-10 & 1\% & 0.917 $\pm$ 0.029 & 0.768 $\pm$ 0.013 & 0.934 $\pm$ 0.004 & 0.916 $\pm$ 0.012 & 0.968 $\pm$ 0.005 & 1.000 $\pm$ 0.000 \\
CIFAR-10 & 5\% & 0.965 $\pm$ 0.008 & 0.762 $\pm$ 0.020 & 0.811 $\pm$ 0.005 & 0.960 $\pm$ 0.005 & 0.963 $\pm$ 0.005 & 1.000 $\pm$ 0.000 \\
CIFAR-10 & 10\% & 0.972 $\pm$ 0.004 & 0.762 $\pm$ 0.018 & 0.545 $\pm$ 0.007 & 0.000 $\pm$ 0.000 & 0.545 $\pm$ 0.007 & n/a \\
PneumoniaMNIST & 0\% & 0.029 $\pm$ 0.033 & 0.866 $\pm$ 0.030 & n/a & n/a & n/a & n/a \\
PneumoniaMNIST & 1\% & 0.989 $\pm$ 0.017 & 0.869 $\pm$ 0.019 & 0.986 $\pm$ 0.000 & 0.996 $\pm$ 0.010 & 1.000 $\pm$ 0.000 & 1.000 $\pm$ 0.000 \\
PneumoniaMNIST & 5\% & 0.999 $\pm$ 0.001 & 0.862 $\pm$ 0.027 & 0.940 $\pm$ 0.001 & 0.998 $\pm$ 0.002 & 0.999 $\pm$ 0.001 & 1.000 $\pm$ 0.000 \\
PneumoniaMNIST & 10\% & 1.000 $\pm$ 0.000 & 0.756 $\pm$ 0.162 & 0.894 $\pm$ 0.004 & 0.995 $\pm$ 0.004 & 0.996 $\pm$ 0.004 & 1.000 $\pm$ 0.000 \\
```

## 2. Commit SHA

```
6bf3af4c46ede67d48f3a3bbb2cec801a348fccb
```

This commit includes two fixes made while running the grid:
- `spectral.py`: switched numpy's default SVD driver (`gesdd`) to
  scipy's `gesvd` -- `gesdd` reliably raised `MemoryError` on Windows
  for a class-sized matrix as large as CIFAR-10's (~5000x8192
  activations), even with ample RAM free.
- `make_table.py`: fixed a crash in the stdev computation whenever a
  metric is legitimately `NaN` (e.g. precision/AUROC undefined at 0%
  poisoning).

## 3. Seed list

Seeds 0-4, unmodified default from `tooling/_pipeline_common.sh`.

## 4. Sample assurance report (healthcare, 5% poison, seed 0)

```markdown
# aegis-scan Assurance Report

**Dataset:** healthcare
**Generated:** 2026-09-07T12:18:13.107608+00:00
**Poison rate (ground truth):** 4.992%

## Finding

The fused detector (spectral signature analysis + activation clustering combined) achieved AUROC=1.000 (excellent) and caught 100.0% of truly poisoned samples when flagging exactly as many samples as were actually poisoned.

## Detection metrics

- **Activation clustering:** TPR=100.0%, FPR=0.0%, precision=100.0% (tp=235, fp=0, fn=0, tn=4473)
- **Both detectors agree:** TPR=61.7%, FPR=0.0%, precision=100.0% (tp=145, fp=0, fn=90, tn=4473)
- **Spectral signature analysis:** AUROC=0.941 (excellent), average precision=0.312, top-k recall=31.1%
- **Fused score (spectral + clustering combined):** AUROC=1.000 (excellent), average precision=1.000, top-k recall=100.0%

## MITRE ATLAS mapping

The attack this report tests for maps to the following ATLAS techniques:

- **[AML.T0020] Poison Training Data** -- The primary technique aegis-scan is built around: stage 02 simulates this attack directly (modifying a subset of training images and their labels), and stages 05-07 are the detection and validation layer for it.
- **[AML.T0059] Erode Dataset Integrity** -- A poisoned dataset is a targeted special case of this broader technique: the altered subset isn't random corruption, it's crafted to survive training and re-emerge as a backdoor -- which is why detecting it needs more than basic data-quality checks.
- **[AML.T0018] Manipulate AI Model** -- The end state of a successful poisoning attack: a persistent, hidden change in the trained model's behavior. aegis-scan doesn't inspect model weights directly the way this technique's 'Poison AI Model' sub-technique (AML.T0018.000) describes -- it infers the same outcome indirectly, from how poisoned training data reshapes a layer's activations.

## NIST AI RMF mapping

**MEASURE 2.7:** "AI system security and resilience -- as identified in the MAP function -- are evaluated and documented."

This report is exactly that evaluation and documentation, scoped to one named, testable security risk (data poisoning / backdoor attacks): quantified detection performance -- TPR, FPR, AUROC -- measured against a known ground truth, not a policy statement that testing happened somewhere. A governance framework can point an auditor here as the technical evidence MEASURE 2.7 asks an organization to produce.
```
