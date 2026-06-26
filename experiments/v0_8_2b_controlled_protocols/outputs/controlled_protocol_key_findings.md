# v0.8.2b 受控协议评测结果

## 1. 当前定位

本报告只汇总 controlled_protocols.yaml 中预定义的协议。全组合搜索只作为 exploratory screening，不作为主结论。

## 2. 最佳主协议

- protocol: `dense_convnext_retfound`
- family: `dense_baseline`
- scouts: ``
- experts: `convnext_tiny+retfound_mae_cfp_official_protocol`
- budget: 1.0
- policy/signal: `dense`
- accuracy: 0.854545
- macro_f1: 0.718665
- qwk: 0.910170
- n_error: 160
- ms_per_image: 4.003712

## 3. 主协议 cost-performance frontier

- `dense_convnext` / dense / budget=1.0: Acc=0.813636, ms/image=0.486144, n_error=205
- `dense_swin` / dense / budget=1.0: Acc=0.829091, ms/image=0.611608, n_error=188
- `convnext_to_retfound` / high_entropy / budget=0.2: Acc=0.847273, ms/image=1.180663, n_error=168
- `swin_to_retfound` / low_confidence / budget=0.3: Acc=0.848182, ms/image=1.665126, n_error=167
- `swin_to_retfound` / low_margin / budget=0.3: Acc=0.849091, ms/image=1.665126, n_error=166
- `swin_to_retfound` / high_entropy / budget=0.3: Acc=0.850000, ms/image=1.665126, n_error=165
- `convnext_swin_to_retfound` / disagreement_then_uncertainty / budget=0.2: Acc=0.850909, ms/image=1.801266, n_error=164
- `convnext_swin_to_retfound` / disagreement_then_uncertainty / budget=0.3: Acc=0.853636, ms/image=2.153022, n_error=161
- `dense_convnext_retfound` / dense / budget=1.0: Acc=0.854545, ms/image=4.003712, n_error=160

## 4. 解释边界

- 本结果不是全局最优组合搜索。
- candidate ranking score 不作为最终证据。
- ViT-B 和 GreenScout 当前主要作为 screening/ablation 对照。
- Multi-scout routing 用于验证 scout 不必只有一个。