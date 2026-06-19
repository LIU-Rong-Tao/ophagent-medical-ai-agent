# v0.7.1 External DR Direct Inference Summary

## 版本定位

本结果使用 v0.7.0 冻结的 APTOS-trained checkpoints，直接在 IDRiD_data / MESSIDOR2 test split 上推理。

本阶段不使用外部 train / val 训练，不根据外部结果重新选择排序信号或复核预算。

## 样本规模

- IDRiD_data / convnext_tiny: 103 records
- IDRiD_data / retfound_mae_cfp_official_like: 103 records
- IDRiD_data / swin_tiny: 103 records
- IDRiD_data / vit_b_imagenet: 103 records
- IDRiD_data / vit_b_official_like: 103 records
- IDRiD_data / vit_l_official_like: 103 records
- MESSIDOR2 / convnext_tiny: 526 records
- MESSIDOR2 / retfound_mae_cfp_official_like: 526 records
- MESSIDOR2 / swin_tiny: 526 records
- MESSIDOR2 / vit_b_imagenet: 526 records
- MESSIDOR2 / vit_b_official_like: 526 records
- MESSIDOR2 / vit_l_official_like: 526 records

## 分类迁移指标

- IDRiD_data / convnext_tiny / accuracy: 0.3204 (n=103)
- IDRiD_data / convnext_tiny / macro_f1: 0.3056 (n=103)
- IDRiD_data / convnext_tiny / qwk: 0.5835 (n=103)
- IDRiD_data / convnext_tiny / weighted_f1: 0.3465 (n=103)
- IDRiD_data / retfound_mae_cfp_official_like / accuracy: 0.3301 (n=103)
- IDRiD_data / retfound_mae_cfp_official_like / macro_f1: 0.2708 (n=103)
- IDRiD_data / retfound_mae_cfp_official_like / qwk: 0.5809 (n=103)
- IDRiD_data / retfound_mae_cfp_official_like / weighted_f1: 0.3172 (n=103)
- IDRiD_data / swin_tiny / accuracy: 0.4563 (n=103)
- IDRiD_data / swin_tiny / macro_f1: 0.3289 (n=103)
- IDRiD_data / swin_tiny / qwk: 0.5723 (n=103)
- IDRiD_data / swin_tiny / weighted_f1: 0.4294 (n=103)
- IDRiD_data / vit_b_imagenet / accuracy: 0.3107 (n=103)
- IDRiD_data / vit_b_imagenet / macro_f1: 0.2514 (n=103)
- IDRiD_data / vit_b_imagenet / qwk: 0.4964 (n=103)
- IDRiD_data / vit_b_imagenet / weighted_f1: 0.3307 (n=103)
- IDRiD_data / vit_b_official_like / accuracy: 0.3689 (n=103)
- IDRiD_data / vit_b_official_like / macro_f1: 0.3029 (n=103)
- IDRiD_data / vit_b_official_like / qwk: 0.5415 (n=103)
- IDRiD_data / vit_b_official_like / weighted_f1: 0.3752 (n=103)
- IDRiD_data / vit_l_official_like / accuracy: 0.3495 (n=103)
- IDRiD_data / vit_l_official_like / macro_f1: 0.3016 (n=103)
- IDRiD_data / vit_l_official_like / qwk: 0.5648 (n=103)
- IDRiD_data / vit_l_official_like / weighted_f1: 0.3509 (n=103)
- MESSIDOR2 / convnext_tiny / accuracy: 0.5875 (n=526)
- MESSIDOR2 / convnext_tiny / macro_f1: 0.3160 (n=526)
- MESSIDOR2 / convnext_tiny / qwk: 0.3771 (n=526)
- MESSIDOR2 / convnext_tiny / weighted_f1: 0.4914 (n=526)
- MESSIDOR2 / retfound_mae_cfp_official_like / accuracy: 0.5646 (n=526)
- MESSIDOR2 / retfound_mae_cfp_official_like / macro_f1: 0.3280 (n=526)
- MESSIDOR2 / retfound_mae_cfp_official_like / qwk: 0.3786 (n=526)
- MESSIDOR2 / retfound_mae_cfp_official_like / weighted_f1: 0.4972 (n=526)
- MESSIDOR2 / swin_tiny / accuracy: 0.6065 (n=526)
- MESSIDOR2 / swin_tiny / macro_f1: 0.3270 (n=526)
- MESSIDOR2 / swin_tiny / qwk: 0.4789 (n=526)
- MESSIDOR2 / swin_tiny / weighted_f1: 0.5233 (n=526)
- MESSIDOR2 / vit_b_imagenet / accuracy: 0.5817 (n=526)
- MESSIDOR2 / vit_b_imagenet / macro_f1: 0.2793 (n=526)
- MESSIDOR2 / vit_b_imagenet / qwk: 0.3232 (n=526)
- MESSIDOR2 / vit_b_imagenet / weighted_f1: 0.4738 (n=526)
- MESSIDOR2 / vit_b_official_like / accuracy: 0.6046 (n=526)
- MESSIDOR2 / vit_b_official_like / macro_f1: 0.2717 (n=526)
- MESSIDOR2 / vit_b_official_like / qwk: 0.2554 (n=526)
- MESSIDOR2 / vit_b_official_like / weighted_f1: 0.4838 (n=526)
- MESSIDOR2 / vit_l_official_like / accuracy: 0.6046 (n=526)
- MESSIDOR2 / vit_l_official_like / macro_f1: 0.3053 (n=526)
- MESSIDOR2 / vit_l_official_like / qwk: 0.4334 (n=526)
- MESSIDOR2 / vit_l_official_like / weighted_f1: 0.5226 (n=526)

## 解释边界

- 这是 direct external inference，不是外部数据重训。
- 外部分类性能无论好坏都应报告。
- 若分类迁移表现严重不足，后续 review prioritization 只能作为 failure analysis，不能强称 protocol 泛化成功。
- IDRiD_data 内部 train/test 重复不影响本阶段，因为本阶段不使用 IDRiD train。
