# v0.7.0 外部 DR 验证协议冻结与数据预检查

本目录用于保存 v0.7.0 的协议冻结和外部 DR 数据 precheck 结果。

## 目录用途

v0.7.0 不进行新的方法搜索，也不根据外部数据结果调整方法。

本阶段只做：

1. 冻结 v0.6.x 得到的审计协议；
2. 明确 primary targets、review budget、主排序信号和 diagnostic baselines；
3. 检查 IDRiD / MESSIDOR2 是否适合承接 direct external validation；
4. 为 v0.7.1 的 frozen checkpoint external validation 做准备。

## 关键文档

- `notes/v0.7.0_external_dr_protocol_freeze.md`

## 预期输出

后续 precheck 脚本应生成：

- `external_dr_dataset_inventory.csv`
- `external_dr_class_distribution.csv`
- `external_dr_protocol_feasibility.csv`
- `external_dr_precheck_summary.md`

## 边界说明

v0.7.0 不是最终外部验证结果。

真正的 direct external validation 应在 v0.7.1 中完成，并且必须使用 v0.7.0 已冻结的协议。
