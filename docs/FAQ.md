# Frequently Asked Questions

## General

**Q: What's the difference between EDG and EDG++?**

A: EDG uses naive distillation (all samples). EDG++ adds a pretrained reliability estimator to filter unreliable ED predictions, preventing negative transfer.

**Q: Do I need to run pretraining?**

A: No. We provide pretrained models. Just download them with `bash scripts/download_pretrained_models.sh`.

## Installation

**Q: PyTorch Geometric installation fails**

A: Check CUDA compatibility. Visit https://pytorch-geometric.readthedocs.io/ for version-specific instructions.

**Q: CUDA out of memory**

A: Reduce batch size in training scripts or use gradient accumulation.

## Training

**Q: How to reproduce paper results?**

A: Run `bash scripts/reproduce_qm9_results.sh` or `bash scripts/reproduce_rmd17_results.sh`.

**Q: What hyperparameters should I use?**

A: See [docs/DISTILLATION.md](DISTILLATION.md) for recommended settings per dataset.

**Q: Training loss becomes NaN**

A: Reduce learning rate or `weight_ED`. Check if ED features are loaded correctly.

**Q: No improvement over baseline**

A: Try different `alpha_std_all` values (0.3-0.7). Ensure `--use_evaluator` is enabled for EDG++.

## Data

**Q: Where to download datasets?**

A: Run `bash scripts/download_datasets.sh all`. QM9 and rMD17 auto-download on first use.

**Q: How to generate ED features for custom datasets?**

A: See [docs/PRETRAIN.md](PRETRAIN.md) section "Generating ED Features for New Datasets".

## Results

**Q: How to parse evaluation results?**

A: See [experiments/README.md](../experiments/README.md) for code examples.
