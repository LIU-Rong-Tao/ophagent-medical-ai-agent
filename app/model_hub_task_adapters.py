"""Task Profile adapters used by the Model Hub tool layer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.model_hub_index import build_task_profile_index
from app.orchestration_contracts import ConfiguredTaskAdapter, TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    PROJECT_ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols"
    / "model_hub_task_profiles_v2.json"
)
MISSING_CLINICAL_VALUE_PREFIXES = ("未提供", "未采集", "未知", "不详")
MISSING_CLINICAL_VALUES = {
    "",
    "未记录",
    "暂无",
    "无资料",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not provided",
}


def _is_present_clinical_value(value: Any) -> bool:
    if not isinstance(value, (str, int, float)):
        return False
    normalized = str(value).strip()
    if normalized.casefold() in MISSING_CLINICAL_VALUES:
        return False
    return not normalized.startswith(MISSING_CLINICAL_VALUE_PREFIXES)


@dataclass(frozen=True)
class ClinicalFieldSpec:
    """Allowlisted, task-specific clinical context shown to a doctor."""

    label: str
    metadata_keys: tuple[str, ...]
    missing_value: str
    note: str
    missing_status: str = "missing"

    def view(self, metadata: dict[str, Any]) -> dict[str, str]:
        for key in self.metadata_keys:
            value = metadata.get(key)
            if _is_present_clinical_value(value):
                return {
                    "label": self.label,
                    "value": str(value).strip(),
                    "status": "provided",
                    "status_label": "已提供",
                    "note": self.note,
                }
        return {
            "label": self.label,
            "value": self.missing_value,
            "status": self.missing_status,
            "status_label": (
                "当前模型未评估"
                if self.missing_status == "not_assessed"
                else "未提供"
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class ClinicalAssistProfile:
    """Clinical presentation profile injected through the Task Adapter."""

    result_heading: str
    evidence_label: str
    model_scope: str
    review_prompt: str
    fields: tuple[ClinicalFieldSpec, ...]

    def view(
        self,
        metadata: dict[str, Any],
        *,
        image_count: int,
    ) -> dict[str, Any]:
        return {
            "result_heading": self.result_heading,
            "current_evidence": (
                f"{self.evidence_label} · {max(image_count, 0)} 张"
            ),
            "quality_boundary": (
                "已通过文件与技术可读性检查；未完成临床级图像质量判定"
            ),
            "model_scope": self.model_scope,
            "review_prompt": self.review_prompt,
            "fields": tuple(field.view(metadata) for field in self.fields),
        }


_DR_CLINICAL_PROFILE = ClinicalAssistProfile(
    result_heading="糖尿病视网膜病变影像分级提示",
    evidence_label="彩色眼底照片（CFP）",
    model_scope=(
        "本次结果仅基于单张 CFP 的 DR 影像分级；不评估视力、"
        "黄斑厚度或液体，也不替代双眼散瞳检查。"
    ),
    review_prompt=(
        "请结合眼别、双眼眼底检查和视力复核。若视力下降、"
        "怀疑黄斑受累或影像与临床不一致，需补充黄斑 OCT 或人工阅片。"
    ),
    fields=(
        ClinicalFieldSpec(
            label="眼别与双眼对应",
            metadata_keys=("眼别", "laterality"),
            missing_value="未提供眼别，无法与对侧眼比较",
            note="分级结果需对应到具体眼别，并结合双眼情况。",
        ),
        ClinicalFieldSpec(
            label="最佳矫正视力",
            metadata_keys=("最佳矫正视力", "BCVA", "bcva"),
            missing_value="未提供",
            note="用于判断影像分级与视觉功能是否一致。",
        ),
        ClinicalFieldSpec(
            label="黄斑 OCT",
            metadata_keys=("黄斑 OCT", "macular_oct"),
            missing_value="未提供",
            note="当前 CFP 模型未评估黄斑厚度、液体或牵拉改变。",
        ),
        ClinicalFieldSpec(
            label="糖尿病与既往治疗信息",
            metadata_keys=(
                "糖尿病病程与 HbA1c",
                "diabetes_context",
                "既往眼底治疗",
            ),
            missing_value="病程、HbA1c 与既往眼底治疗均未提供",
            note="这些信息用于临床综合判断，不参与本次模型推理。",
        ),
    ),
)


_GLAUCOMA_CLINICAL_PROFILE = ClinicalAssistProfile(
    result_heading="青光眼相关眼底影像提示",
    evidence_label="彩色眼底照片（CFP）",
    model_scope=(
        "本次结果仅基于 CFP；单张眼底照片不能确认青光眼，"
        "也不能反映眼压、视野功能或视网膜神经纤维层厚度。"
    ),
    review_prompt=(
        "请结合眼压、视盘结构、OCT RNFL/GCC 和标准自动视野复核；"
        "资料不足时应标记为待补充，而不是据 CFP 单独下结论。"
    ),
    fields=(
        ClinicalFieldSpec(
            label="眼压（含测量时间）",
            metadata_keys=("眼压", "IOP", "iop"),
            missing_value="未提供",
            note="需结合降眼压用药、角膜厚度和测量条件解释。",
        ),
        ClinicalFieldSpec(
            label="杯盘比 / 视盘结构",
            metadata_keys=("杯盘比", "CDR", "cdr"),
            missing_value="当前模型未输出结构化杯盘比",
            note="应由医生阅片或经获准的结构量化工具确认。",
            missing_status="not_assessed",
        ),
        ClinicalFieldSpec(
            label="OCT RNFL / GCC",
            metadata_keys=("OCT RNFL / GCC", "oct_rnfl_gcc"),
            missing_value="未提供",
            note="用于补充视神经结构性损伤证据。",
        ),
        ClinicalFieldSpec(
            label="标准自动视野",
            metadata_keys=("标准自动视野", "visual_field"),
            missing_value="未提供",
            note="用于补充功能性损伤及分期信息。",
        ),
    ),
)


def _clinical_profile_for(spec: TaskSpec) -> ClinicalAssistProfile:
    configured = {
        "aptos_dr_5class": _DR_CLINICAL_PROFILE,
        "glaucoma_3class": _GLAUCOMA_CLINICAL_PROFILE,
    }.get(spec.task_id)
    if configured is not None:
        return configured
    return ClinicalAssistProfile(
        result_heading=f"{spec.report_label}影像提示",
        evidence_label=spec.modality,
        model_scope=(
            f"当前模型只分析已登记的 {spec.modality} 输入；"
            "未提供的临床资料不会被推断或补写。"
        ),
        review_prompt="请结合任务相关临床检查与人工阅片复核。",
        fields=(),
    )


@dataclass(frozen=True)
class ModelHubTaskAdapter:
    """Inject task-specific labels, risk semantics, and tool assets."""

    configured: ConfiguredTaskAdapter
    risk_proxy_name: str
    risk_proxy_definition: str
    tool_contract: dict[str, Any]
    clinical_assist: ClinicalAssistProfile

    @property
    def spec(self) -> TaskSpec:
        return self.configured.spec

    def validate_metadata(
        self,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        return self.configured.validate_metadata(metadata)

    def risk_summary(
        self,
        probabilities: tuple[float, ...],
    ) -> dict[str, Any]:
        result = self.configured.risk_summary(probabilities)
        return {
            **result,
            "name": self.risk_proxy_name,
            "definition": self.risk_proxy_definition,
        }

    def as_legacy_contract(self) -> dict[str, Any]:
        """Conservative compatibility projection for existing UI helpers."""

        return {
            "task_name": self.spec.report_label,
            "class_order": list(range(len(self.spec.label_space))),
            "class_labels": list(self.spec.label_space),
            "modality": self.spec.modality,
            "risk_proxy": self.spec.risk_semantics,
            **self.tool_contract,
        }


def load_task_adapters(
    path: Path = PROFILE_PATH,
) -> dict[str, ModelHubTaskAdapter]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = build_task_profile_index(PROJECT_ROOT).set_index("task_id")
    adapters: dict[str, ModelHubTaskAdapter] = {}
    for row in payload.get("profiles", []):
        if not isinstance(row, dict):
            continue
        task_id = str(row["task_id"])
        profile = profiles.loc[task_id]
        spec = TaskSpec(
            task_id=task_id,
            dataset_id=str(profile["dataset_id"]),
            modality=str(profile["modality"]),
            label_space=tuple(
                str(value)
                for value in json.loads(str(profile["label_space"]))
            ),
            primary_metric=str(profile["primary_metric"]),
            risk_semantics=str(profile["risk_semantics"]),
            report_label=str(profile["report_label"]),
            risk_positive_class_ids=tuple(
                int(value)
                for value in json.loads(
                    str(profile["risk_positive_class_ids"])
                )
            ),
            adaptation_type=str(profile["adaptation_type"]),
        )
        adapters[task_id] = ModelHubTaskAdapter(
            configured=ConfiguredTaskAdapter(spec),
            risk_proxy_name=str(row["risk_proxy_name"]),
            risk_proxy_definition=str(row["risk_proxy_definition"]),
            tool_contract=dict(row["tool_contract"]),
            clinical_assist=_clinical_profile_for(spec),
        )
    return adapters


TASK_ADAPTERS = load_task_adapters()


def task_adapter_for(task_id: str) -> ModelHubTaskAdapter:
    try:
        return TASK_ADAPTERS[task_id]
    except KeyError as exc:
        raise KeyError(f"unregistered Task Profile: {task_id}") from exc


TASK_CONTRACTS = {
    task_id: adapter.as_legacy_contract()
    for task_id, adapter in TASK_ADAPTERS.items()
}
