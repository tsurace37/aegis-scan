# Phase 1 tooling

`measure_asr.py` and `run_grid.sh` are NOT part of the aegis-scan
repository. They're a companion script for this paper, written against
the public API as of commit `7c9c063`. Copy both into your aegis-scan
checkout (`tooling/` subfolder, or wherever) before running.

## Why measure_asr.py exists

`aegis-scan evaluate` (stage 07) scores the four detectors against the
ground-truth poison mask. It never touches a held-out test set. Nothing
in the existing pipeline currently tells you whether the backdoor was
actually learned -- and without that, a detection AUROC number doesn't
establish what the paper claims it establishes. `measure_asr.py` closes
that gap: clean accuracy on an untriggered test set, and attack success
rate (fraction of triggered, non-target-class test images the model now
classifies as the target class).

## Setup

```bash
git clone https://github.com/tsurace37/aegis-scan
cd aegis-scan
pip install -e . --break-system-packages   # or your usual venv workflow
pip install medmnist torchvision --break-system-packages
cp /path/to/measure_asr.py /path/to/run_grid.sh tooling/  # this pair of files
mkdir -p tooling && mv measure_asr.py run_grid.sh tooling/  # if not already there
chmod +x tooling/run_grid.sh
```

Confirm the CLI is on your PATH: `aegis-scan --help` should list all
eight subcommands (inject, train, extract-activations, detect, fuse,
evaluate, report, plus whatever else is registered).

## Run

```bash
./tooling/run_grid.sh
```

This runs the full 2 x 4 x 5 = 40-configuration grid (both datasets,
0%/1%/5%/10% poisoning, 5 seeds) end to end: inject, train, extract,
detect, fuse, evaluate, measure_asr, report -- eight commands per
configuration, so 320 commands total. It prints its own progress and
ends by printing the commit SHA for the paper's reproducibility
statement.

**Time.** PneumoniaMNIST (~4.7k train images) is fast even on CPU.
CIFAR-10 (50k images, 10 epochs, 20 configurations) is not -- budget for
hours, not minutes, without a GPU. If time is tight, run PneumoniaMNIST
first and treat it as the paper's primary, complete result; CIFAR-10 can
follow. The paper already frames CIFAR-10 as the cross-sector
generalization check, not the headline claim.

**If a run fails partway**, `run_grid.sh` has `set -e` so it stops
rather than silently skipping a step. Delete that configuration's `runs/<tag>/`
directory and re-run just that iteration by hand, or comment out the
completed prefixes in the `SEEDS`/`RATES` arrays and re-run the script.

## After it finishes

```bash
python3 make_table.py runs/
```

Prints a "matched keys" block first -- **read it**. If `asr` or
`clean_acc` show up under "NOT FOUND," a `measure_asr.py` step failed or
didn't write where expected; check `runs/*/asr.json` exists before
trusting the table. Then it prints the LaTeX table rows: paste them over
the `---` placeholders in Table 1 in `main.tex`.

## What to send back

1. The pasted table rows (or just the `runs/` directory -- I can run
   `make_table.py` on it).
2. The commit SHA `run_grid.sh` printed at the end.
3. One `runs/<tag>/assurance_report.md` -- the real stage-08 output, to
   paste into the paper's appendix in place of the current TODO.
4. Confirmation of the seed list used (0-4, or whatever you changed it to).

Send those four things and the rest of the paper closes in one pass.
