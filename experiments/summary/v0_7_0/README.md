# v0.7.0 External DR Protocol Freeze and Dataset Precheck

v0.7.0 是外部 DR 验证前的协议冻结和数据预检版本。

这一版不做外部模型推理，目标是先确认外部数据、checkpoint 和审计口径是否可用，避免看见外部结果后再调整目标事件或排序信号。

## 数据规模

| Dataset | Split | Images |
|---|---|---:|
| IDRiD_data | test | 103 |
| MESSIDOR2 | test | 526 |

当前外部数据没有患者 ID、双眼关系或就诊级元数据，因此后续分析按 image-level 进行。

## 已生成产物

| File | Description |
|---|---|
| `external_dr_dataset_inventory.csv` | 外部数据集清单 |
| `external_dr_class_distribution.csv` | 外部数据类别分布 |
| `external_dr_data_quality.csv` | 基础数据质量检查 |
| `external_dr_overlap_audit.csv` | APTOS 与外部数据 MD5 重叠审计 |
| `external_dr_duplicate_exclusion_manifest.csv` | IDRiD 内部重复样本记录 |
| `external_dr_precheck_table.csv` | 外部数据预检汇总表 |
| `external_dr_precheck_summary.md` | 外部数据预检文字总结 |
| `checkpoint_manifest.csv` | frozen checkpoint manifest |

## 预检结果

- 未发现 APTOS 与 IDRiD_data / MESSIDOR2 test split 的 MD5 重叠。
- IDRiD 内部发现 1 组 train/test MD5 重复且标签冲突：
  - `IDRiD_data/test/dsevereDR/IDRiD_064.png`
  - `IDRiD_data/train/anoDR/IDRiD_118.png`
  - MD5：`7dc007745435fcba18b2d390312a40b4`
- v0.7.1 外部直接推理只使用 IDRiD_data / MESSIDOR2 test split，不使用 IDRiD train split，因此该重复问题不阻塞 v0.7.1。
- 如果后续进入目标域重训或微调，应排除训练侧重复样本，或按 MD5 group 重新划分，确保重复图像不跨 split。

## Checkpoints

v0.7.0 记录了 6 个 APTOS-trained frozen checkpoints：

- `convnext_tiny`
- `swin_tiny`
- `vit_b_imagenet`
- `vit_b_official_like`
- `vit_l_official_like`
- `retfound_mae_cfp_official_like`

RETFound 相关表述使用：

```text
RETFound-MAE-CFP initialized ViT-L under the OphAgent official-like unified training protocol.
```

不写成 strict official RETFound fine-tuning reproduction。

## Protocol deviation: learned_logistic external baseline

v0.7.0 原协议计划保留 `learned_logistic` 作为外部预设监督式 baseline（非 primary comparator）。

当前 v0.7.1/v0.7.1b 已完成 primary gate-only comparison，但尚未实现 APTOS-frozen `learned_logistic` 的外部推理。因此，外部 `learned_logistic` 记录为预设监督式 baseline（非 primary comparator）缺失 / protocol deviation。

该偏差不影响 v0.7.1b 的 primary comparison：

```text
gated_severe_prob_mass_only vs random_gate_only_expected
```

后续如果补充外部 `learned_logistic` baseline，必须读取 APTOS 侧 frozen artifact，包括 feature order、missing value policy、StandardScaler 参数、logistic coefficients 和 intercept；不能在 IDRiD_data / MESSIDOR2 上重新拟合、重新标准化或重新选特征。

## 与后续版本关系

- v0.7.0：协议冻结、数据预检、重叠审计。
- v0.7.1：使用 frozen checkpoints 直接推理 IDRiD_data / MESSIDOR2。
- v0.7.1b：补 random gate-only、image-clustered bootstrap CI 和 seed sensitivity。
