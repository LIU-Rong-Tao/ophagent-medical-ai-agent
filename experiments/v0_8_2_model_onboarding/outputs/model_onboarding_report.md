# v0.8.2 Model Onboarding Report

## 1. Scope

This report validates registry-driven onboarding for timm classifier models.

## 2. Validation Summary

### swin_tiny

- arch: `swin_tiny_patch4_window7_224`
- strict_load_ok: `True`
- prediction_valid: `True`
- n: 1100
- accuracy: 0.829091
- macro_f1: 0.656707
- qwk: 0.898186
- n_error: 188

### vit_b_imagenet

- arch: `vit_base_patch16_224`
- strict_load_ok: `True`
- prediction_valid: `True`
- n: 1100
- accuracy: 0.818182
- macro_f1: 0.646211
- qwk: 0.875670
- n_error: 200

## 3. Forward-Cost Summary

### swin_tiny

- n_runs: 5
- mean_ms_per_image_median: 0.611608
- mean_ms_per_image_mean: 0.611633
- mean_ms_per_image_std: 0.001659
- mean_ms_per_image_cv: 0.002712
- total_forward_ms_median: 672.769
- images_per_second_median: 1635.034
- peak_mem_max_mb: 612.779
- checkpoint_mb: 105.063
- latency_cv_gt_10pct: `False`
- latency_has_run_outlier: `False`
- n_latency_outlier_any: 0
- latency_has_mad_sensitive_flag: `True`
- n_latency_mad_sensitive_flag: 2

### vit_b_imagenet

- n_runs: 5
- mean_ms_per_image_median: 1.106972
- mean_ms_per_image_mean: 1.106926
- mean_ms_per_image_std: 0.000946
- mean_ms_per_image_cv: 0.000855
- total_forward_ms_median: 1217.670
- images_per_second_median: 903.365
- peak_mem_max_mb: 594.687
- checkpoint_mb: 327.369
- latency_cv_gt_10pct: `False`
- latency_has_run_outlier: `False`
- n_latency_outlier_any: 0
- latency_has_mad_sensitive_flag: `False`
- n_latency_mad_sensitive_flag: 0

## 4. Boundary

- Current script only supports registry-declared timm classifier models.
- Cost is forward-only latency and excludes image decode / transform / dataloader / service overhead.
- Non-timm foundation or embedding models should use a separate onboarding script.