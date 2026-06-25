# v0.8.0d Scout Ablation 关键结果

## 1. 实验设置

本轮实验比较 RETFound-Green 与 ConvNeXt-Tiny 作为 scout 时的稀疏专家调用效果。

模型池包括：

- Scout 候选：RETFound-Green linear probe、ConvNeXt-Tiny
- Expert 候选：ConvNeXt-Tiny、RETFound-MAE official-protocol

budget 表示进入专家模型通道的样本比例，即 expert-call equivalent budget；不等同于真实端到端推理成本下降比例。

## 2. Dense / Single Baselines

| method | accuracy | macro_f1 | qwk | n_error |
| --- | --- | --- | --- | --- |
| retfound_green_linear_probe | 0.7791 | 0.6394 | 0.8653 | 243 |
| convnext_tiny | 0.8136 | 0.6496 | 0.8615 | 205 |
| retfound_mae_cfp_official_protocol | 0.8482 | 0.7054 | 0.9129 | 167 |
| experts_only_average | 0.8545 | 0.7187 | 0.9102 | 160 |
| all_three_average | 0.8455 | 0.7112 | 0.9063 | 170 |

## 3. Best Sparse Policy at 30/40/50% Budget

| setting | budget | policy | accuracy | macro_f1 | qwk | n_error | random_acc_p975 | above_random_p975_acc | oracle_accuracy | gap_to_oracle_acc | expert_correctable_total | selected_expert_correctable | expert_correctable_capture_recall | rescued_scout_errors | expert_induced_errors | net_error_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_convnext_scout_to_retfound_only | 0.3000 | high_entropy | 0.8464 | 0.7024 | 0.9080 | 169 | 0.8336 | True | 0.8891 | 0.0427 | 83 | 67 | 0.8072 | 67 | 31 | 36 |
| C_green_scout_to_retfound_only | 0.3000 | high_entropy | 0.8418 | 0.7054 | 0.9114 | 174 | 0.8109 | True | 0.8927 | 0.0509 | 125 | 93 | 0.7440 | 93 | 24 | 69 |
| A_green_scout_to_convnext_retfound_avg | 0.3000 | high_entropy | 0.8409 | 0.7059 | 0.9056 | 175 | 0.8118 | True | 0.8882 | 0.0473 | 120 | 88 | 0.7333 | 88 | 20 | 68 |
| B_green_scout_to_convnext_only | 0.3000 | high_entropy | 0.8191 | 0.6744 | 0.8754 | 199 | 0.8009 | True | 0.8791 | 0.0600 | 110 | 82 | 0.7455 | 82 | 38 | 44 |
| D_convnext_scout_to_retfound_only | 0.4000 | high_entropy | 0.8500 | 0.7124 | 0.9119 | 165 | 0.8382 | True | 0.8891 | 0.0391 | 83 | 79 | 0.9518 | 79 | 39 | 40 |
| A_green_scout_to_convnext_retfound_avg | 0.4000 | high_entropy | 0.8464 | 0.7109 | 0.9075 | 169 | 0.8200 | True | 0.8882 | 0.0418 | 120 | 102 | 0.8500 | 102 | 28 | 74 |
| C_green_scout_to_retfound_only | 0.4000 | low_margin | 0.8418 | 0.7005 | 0.9117 | 174 | 0.8182 | True | 0.8927 | 0.0509 | 125 | 105 | 0.8400 | 105 | 36 | 69 |
| B_green_scout_to_convnext_only | 0.4000 | low_margin | 0.8191 | 0.6705 | 0.8723 | 199 | 0.8045 | True | 0.8791 | 0.0600 | 110 | 97 | 0.8818 | 97 | 53 | 44 |
| A_green_scout_to_convnext_retfound_avg | 0.5000 | low_confidence | 0.8527 | 0.7200 | 0.9102 | 162 | 0.8282 | True | 0.8882 | 0.0355 | 120 | 114 | 0.9500 | 114 | 33 | 81 |
| D_convnext_scout_to_retfound_only | 0.5000 | low_confidence | 0.8518 | 0.7120 | 0.9140 | 163 | 0.8409 | True | 0.8891 | 0.0373 | 83 | 83 | 1.0000 | 83 | 41 | 42 |
| C_green_scout_to_retfound_only | 0.5000 | low_confidence | 0.8482 | 0.7116 | 0.9142 | 167 | 0.8255 | True | 0.8927 | 0.0445 | 125 | 118 | 0.9440 | 118 | 42 | 76 |
| B_green_scout_to_convnext_only | 0.5000 | low_margin | 0.8209 | 0.6716 | 0.8686 | 197 | 0.8082 | True | 0.8791 | 0.0582 | 110 | 105 | 0.9545 | 105 | 59 | 46 |

## 4. 当前边界

- 本结果仍是 prediction-level sparse routing simulation。
- expert-call equivalent budget 不等于真实 wall-clock、吞吐量或显存收益。
- DR-specific risk events 只适用于当前 APTOS DR 五级分级任务，不应直接泛化到所有眼科疾病。
- Oracle same-budget 为事后上界，不可部署。
