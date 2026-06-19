# v0.7.0 外部 DR 数据预检查结果

本文件由 `scripts/precheck_v070_external_dr_datasets.py` 自动生成。

本阶段只检查数据结构、质量、重复风险和 grade-based proxy 承接条件；不运行模型推理，不给出外部泛化结论。

## 核心摘要

| dataset | test_split_exists | test_total_images | event_sample_size_grade_3_or_4 | structurally_eligible | hash_check_performed | duplicate_check_passed_for_direct_external_test | cross_dataset_md5_overlap_rows | external_internal_cross_split_md5_overlap_rows | cross_dataset_overlap_rows_involving_dataset | n_unexpected_class_dirs | n_unreadable_images | all_5_classes_present_test | all_5_classes_present_warning | patient_or_eye_grouping_metadata_observed | analysis_unit | statistical_adequacy_pending |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDRiD_data | True | 103 | 32 | True | True | True | 0 | 1 | 0 | 0 | 0 | True | False | False | image_level_only | True |
| MESSIDOR2 | True | 526 | 34 | True | True | True | 0 | 0 | 0 | 0 | 0 | True | False | False | image_level_only | True |

## 类别分布

### APTOS2019

| split | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| test | 542 | 111 | 300 | 58 | 89 |
| train | 1010 | 207 | 559 | 108 | 164 |
| val | 253 | 52 | 140 | 27 | 42 |

### IDRiD_data

| split | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| test | 34 | 5 | 32 | 19 | 13 |
| train | 107 | 16 | 108 | 59 | 39 |
| val | 27 | 4 | 28 | 15 | 10 |

### MESSIDOR2

| split | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| test | 306 | 81 | 105 | 23 | 11 |
| train | 568 | 151 | 193 | 41 | 19 |
| val | 143 | 38 | 49 | 11 | 5 |

## 数据质量摘要

| dataset | split | split_exists | n_images | n_unreadable_images | min_width | min_height | median_width | median_height | observed_modes | observed_channels | n_unexpected_class_dirs | unexpected_class_dirs | analysis_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| APTOS2019 | train | True | 2048 | 0 | 256 | 256 | 256.0 | 256.0 | RGB | 3 | 0 |  | image_level_only |
| APTOS2019 | val | True | 514 | 0 | 256 | 256 | 256.0 | 256.0 | RGB | 3 | 0 |  | image_level_only |
| APTOS2019 | test | True | 1100 | 0 | 256 | 256 | 256.0 | 256.0 | RGB | 3 | 0 |  | image_level_only |
| IDRiD_data | train | True | 329 | 0 | 3408 | 3408 | 3416.0 | 3416.0 | RGB | 3 | 0 |  | image_level_only |
| IDRiD_data | val | True | 84 | 0 | 3412 | 3412 | 3416.0 | 3416.0 | RGB | 3 | 0 |  | image_level_only |
| IDRiD_data | test | True | 103 | 0 | 3412 | 3412 | 3416.0 | 3416.0 | RGB | 3 | 0 |  | image_level_only |
| MESSIDOR2 | train | True | 972 | 0 | 904 | 904 | 1384.0 | 1384.0 | RGB | 3 | 0 |  | image_level_only |
| MESSIDOR2 | val | True | 246 | 0 | 904 | 904 | 1384.0 | 1384.0 | RGB | 3 | 0 |  | image_level_only |
| MESSIDOR2 | test | True | 526 | 0 | 904 | 904 | 1384.0 | 1384.0 | RGB | 3 | 0 |  | image_level_only |

## 重复与重叠审计

- hash_check_performed: `True`
- overlap rows: `236`

| overlap_key_type | overlap_key | n_rows | datasets | splits | overlap_type | hash_check_performed |
| --- | --- | --- | --- | --- | --- | --- |
| filename | IDRiD_001.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_002.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_003.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_004.png | 2 | IDRiD_data | test;val | cross_split_within_dataset | True |
| filename | IDRiD_005.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_006.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_007.png | 2 | IDRiD_data | test;val | cross_split_within_dataset | True |
| filename | IDRiD_008.png | 2 | IDRiD_data | test;val | cross_split_within_dataset | True |
| filename | IDRiD_009.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_010.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_011.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_012.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_013.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_014.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_015.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_016.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_017.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_018.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_019.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_020.png | 2 | IDRiD_data | test;val | cross_split_within_dataset | True |
| filename | IDRiD_021.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_022.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_023.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_024.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_025.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_026.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_027.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_028.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |
| filename | IDRiD_029.png | 2 | IDRiD_data | test;val | cross_split_within_dataset | True |
| filename | IDRiD_030.png | 2 | IDRiD_data | test;train | cross_split_within_dataset | True |

## 解释边界

- `structurally_eligible=True` 只表示 test split、图像读取和 grade 3/4 proxy 存在初步承接条件。
- `duplicate_check_passed_for_direct_external_test=True` 只表示 hash 检查已运行，且未发现 APTOS 与外部数据之间的跨数据集 md5 重叠。
- `cross_dataset_md5_overlap_rows` 用于判断 APTOS 与外部数据是否存在图像级重复。
- `external_internal_cross_split_md5_overlap_rows` 用于记录外部数据集内部 train/val/test 是否存在图像级重复；该字段主要影响后续目标数据重训实验。
- `event_sample_size_grade_3_or_4` 只是外部 test 中 grade 3/4 样本数，不等于最终 dangerous event 数。
- `statistical_adequacy_pending=True` 表示统计充分性必须等 v0.7.1 推理结果和危险事件数量出来后再判断。
- `all_5_classes_present_test=False` 是警告条件，不是自动阻断条件。
- 若无患者或双眼元数据，后续只能声明 image-level analysis。
- 无 DME、病灶级或临床终点标签时，只能称为 grade-only proxy，不能称为真实 VTDR 临床终点。

## IDRiD 内部重复图像确认

人工检查确认，以下两张图像内容完全一致：

- `IDRiD_data/test/dsevereDR/IDRiD_064.png`
- `IDRiD_data/train/anoDR/IDRiD_118.png`

二者 md5 相同，且位于不同 split 和不同标签目录，属于外部数据集内部的跨 split 重复与标签冲突。

该问题不影响 v0.7.1 的 APTOS frozen checkpoint direct external validation，因为 v0.7.1 不使用 IDRiD train / val 训练模型。但该问题会影响后续 v0.7.2 的目标数据重训或 protocol reproduction，因此后续若使用 IDRiD train 训练，应排除 `IDRiD_data/train/anoDR/IDRiD_118.png`，或重新按 md5 group 划分数据。

