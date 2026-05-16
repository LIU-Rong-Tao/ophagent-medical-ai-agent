# v0.4.1 Backbone Comparison

## Protocol

- Dataset: APTOS2019
- Task: 5-class diabetic retinopathy classification
- Image size: 224
- Seed: 42
- Learning rate: 1e-4
- Batch size: 32
- Epochs: 10
- Evaluation split: test

## Results

| Backbone | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Best Val Acc |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.8136 | 0.7079 | 0.6555 | 0.6496 | 0.8093 | 0.8366 |
| Swin-Tiny | 0.8291 | 0.7041 | 0.6342 | 0.6567 | 0.8202 | 0.8346 |

## Notes

Swin-Tiny was added as the second backbone baseline under the same APTOS2019 training and evaluation protocol.

Compared with ConvNeXt-Tiny, Swin-Tiny achieved slightly higher test accuracy, macro F1, and weighted F1 in this single-seed experiment.

This is not yet a full benchmark. Future versions should include more backbones and multiple seeds.
