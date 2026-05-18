# Experiment Summary

## Overview

- Project: `OphAgent`
- Version: `v0.5.0`
- Stage: `Vision Baseline`
- Dataset: `APTOS2019`
- Task: `5-class diabetic retinopathy classification`
- Backbone: `swin_tiny_patch4_window7_224.ms_in1k`
- Input size: `224`
- Seed: `42`
- Checkpoint: `experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/checkpoints/swin_tiny_patch4_window7_224.ms_in1k_best.pth`

## Test Metrics

- Test accuracy: `0.8290909090909091`
- Macro precision: `0.7040503281050817`
- Macro recall: `0.6342182193316326`
- Macro F1: `0.6567072753814438`
- Weighted F1: `0.820152057515222`

## Training Config

- Batch size: `32`
- Number of epochs: `10`
- Learning rate: `0.0001`
- Pretrained: `True`

## Training Curve Summary

- Logged epochs: `10`
- Best epoch: `10`
- Best validation accuracy: `0.8346303501945526`
- Final validation accuracy: `0.8346303501945526`
- Final train loss: `0.2569509624736383`
- Final validation loss: `0.5392497052534428`

## Intended Use

Research and engineering demo only. Not for clinical diagnosis.
