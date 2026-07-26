from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.model_hub_demo import _model_flag
from app.model_hub_review import FAULT_SCENARIO, NORMAL_SCENARIOS


ROOT = Path(__file__).resolve().parents[1]


def test_workstation_has_two_read_only_tasks_and_one_fault_scenario() -> None:
    assert set(NORMAL_SCENARIOS) == {
        "APTOS · 冻结 validation 资产",
        "青光眼 · 冻结 validation 资产",
    }
    assert FAULT_SCENARIO == "故障门禁 · 离线资产请求原图推理"
    assert all(
        "test" not in str(configuration).lower()
        for configuration in NORMAL_SCENARIOS.values()
    )


def test_model_hub_reuses_existing_clinical_workspace_entry() -> None:
    clinical_source = (ROOT / "app/model_hub_clinical.py").read_text(
        encoding="utf-8"
    )
    demo_source = (ROOT / "app/model_hub_demo.py").read_text(encoding="utf-8")

    assert "render_offline_case_review_workstation" in clinical_source
    assert "离线病例审阅工作台" in clinical_source
    assert "历史路由回放" in clinical_source
    assert "render_clinical_workspace(data)" in demo_source


def test_workstation_exposes_review_actions_and_research_boundaries() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")

    for text in (
        "病例队列",
        "眼底图像",
        "输入与任务检查",
        "可用模型与资格",
        "单模型输出",
        "模型输出错误风险审计",
        "Scout / Expert 研究模拟",
        "接受模型输出",
        "修改输出",
        "标记不确定",
        "加入复核队列",
        "保存并进入下一例",
        "结构化报告",
        "完整工具调用轨迹",
    ):
        assert text in source
    assert "route_eligible=false" in source
    assert "不提供诊断、治疗或患者分流建议" in source
    assert "http://" not in source
    assert "https://" not in source


def test_report_schema_excludes_source_case_key_and_test_content() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")

    report_body = source[source.index("def _review_report") :]
    assert '"case_alias": case.alias' in report_body
    assert '"source_case_key"' not in report_body
    assert '"test_content_used": False' in report_body
    assert '"external_network_used": False' in report_body


def test_overview_tolerates_legacy_models_without_online_flag() -> None:
    models = pd.DataFrame({"model_id": ["a", "b"]})

    result = _model_flag(models, "task_inference_ready")

    assert result.tolist() == [False, False]
