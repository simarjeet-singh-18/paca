# paca_rl — Bandit-based Connection Selection for PaCA PEFT

Implementation accompanying *"Learning What to Tune: Bandit-based Connection
Selection for Efficient Fine-Tuning"* (Tripathi & Diddigi). It extends PaCA
(Woo et al., ICLR 2025) so that the subset of pre-trained connections updated
during fine-tuning is **learned** via a multi-armed bandit rather than fixed at
random.

> NOTE ON PROVENANCE: this folder was assembled from three sources — the driver
> scripts (from the `paca_2` repo), the modified PEFT library (`paca_modded/peft`),
> and the design docs (`docs/`). The directory layout follows the paths referenced
> inside those docs (`paca_rl/paca_modded/peft/...`, `paca_rl/scripts/...`). It is
> a reasonable reconstruction, not a verbatim copy of a single upstream repo.

## Structure

```
paca_rl/
├── README.md
├── requirements.txt
├── paca_modded/
│   └── peft/                       # Modified HuggingFace PEFT (install from source)
│       └── src/peft/tuners/paca/   # <-- the actual algorithm lives here
│           ├── config.py           #     PacaConfig: arms, UCB/TS, gradient-chain params
│           ├── layer.py            #     per-layer UCB / Thompson arm selection
│           ├── model.py            #     PacaThompsonManager, PacaGradientChainManager
│           └── bnb.py
├── scripts/                        # Experiment drivers (one folder per method)
│   ├── plain/        # full fine-tuning baseline
│   ├── lora/         # LoRA baseline
│   ├── paca/         # original (fixed random) PaCA
│   ├── random/       # R-PaCA (new random subset each epoch)
│   ├── ucb/          # UCB-PaCA           (+ arm_tests/, alpha_tests/, rank_tests/, tpt/)
│   ├── thompson/     # TS-PaCA (Gaussian) (+ arms/, fixed_epoch/, rank/, tpt/)
│   └── gradient_chain/  # Gradient-PaCA   (+ ablation/, arms/, rank/, tpt/)
│       # each folder has: cifar100.py, flowers102.py, caltech101.py, svhn.py (+ pets.py = extra, off-paper)
└── docs/                           # Design write-ups (documentation, not source)
```

## Install & run

```bash
# 1) install the modified PEFT from source (editable)
cd paca_modded/peft && pip install -e . && cd ../..

# 2) install the rest
pip install -r requirements.txt

# 3) run an experiment (downloads the dataset to ./data on first run)
python scripts/gradient_chain/cifar100.py
python scripts/thompson/cifar100_gaussian.py   # if present; see folder
python scripts/ucb/cifar100.py
```

Backbone: ImageNet-1K pre-trained ViT (12 encoder blocks, patch 16).
Datasets in the paper: CIFAR-100, Flowers-102, Caltech-101, SVHN.

## Known code-vs-paper discrepancies (verified by reading the source)

These are documented so they can be reconciled; the code is a faithful *superset*
of the paper with a few divergences:

1. UCB exploration term. Paper Alg.1 line 6: `alpha * sqrt(log t / (N_i + 1))`.
   Code (layer.py:176, model.py:83, model.py:473): `ucb_alpha * sqrt(2*log t / n_i)`
   with an `n_i>0 -> inf` guard. (Textbook UCB1: factor of 2, no +1 smoothing.)
2. Reward normalization. Paper Eq.7 reward = A_after - A_before (raw). The library
   uses the raw reward, but the Gaussian-TS *driver scripts* feed `reward/10.0`.
3. Gradient-chain starts. Paper: "K distinct" starting indices for diversity.
   Code: `torch.randperm(...)[:num_arms]` (random; distinct only by construction).
4. Chain hop guards: `next = argmax(...) % r` wrap-around and a fallback-to-0 branch
   (model.py build_gradient_chains) are not in the paper's Eq.6.
5. Extra, off-paper: horizontal & vertical arm distributions, Beta Thompson Sampling,
   and Oxford-Pets (`pets.py`) runs. The paper reports only random+gradient-chain arms,
   Gaussian TS, and the four datasets above.

Matches confirmed: Gaussian-TS posterior update = Alg.2 verbatim; reward Eq.7 at the
library level; UCB mean = incremental mean; gradient-chain sensitivity = mean-activation
proxy (Eq.4) + batch-averaged |grad| (Eq.5) + argmax hop (Eq.6).

NOTE: the stale `paca_modded/peft/build/` copy from the original archive was removed;
`src/` is the source of record.
