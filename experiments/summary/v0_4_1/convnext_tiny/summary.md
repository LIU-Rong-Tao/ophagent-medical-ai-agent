# Experiment Summary

## Overview

- Project: `OphAgent`
- Version: `v0.4.1`
- Stage: `Vision Baseline`
- Dataset: `APTOS2019`
- Task: `5-class diabetic retinopathy classification`
- Backbone: `convnext_tiny`
- Input size: `224`
- Seed: `42`
- Checkpoint: `experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth`

## Test Metrics

- Test accuracy: `0.8136363636363636`
- Macro precision: `0.7079157090490819`
- Macro recall: `0.6554590935354048`
- Macro F1: `0.6496387745585863`
- Weighted F1: `0.8093088458126397`

## Training Config

- Batch size: `32`
- Number of epochs: `10`
- Learning rate: `0.0001`
- Pretrained: `True`

## Training Curve Summary

- Logged epochs: `10`
- Best epoch: `9`
- Best validation accuracy: `0.8365758754863813`
- Final validation accuracy: `0.830739299610895`
- Final train loss: `0.2944651996949687`
- Final validation loss: `0.6123521923727822`

## Intended Use

Research and engineering demo only. Not for clinical diagnosis.
