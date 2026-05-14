"""
OphAgent v0.2.2
结构化 findings 数据定义。

注意：
这里不是病灶检测结果，也不是临床诊断。
这里只是把模型预测结果、CAM 关注区域等信息，
整理成后续 report generation 可以消费的结构化中间表示。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Finding:
    """
    单条结构化 finding。

    字段说明：
    - name: finding 的内部名称，便于程序处理
    - display_name: 给用户展示的中文名称
    - description: 中文描述，必须避免临床确诊式 wording
    - confidence: 与该 finding 相关的置信度，通常来自分类模型置信度
    - evidence: 支撑该 finding 的证据来源，例如 prediction / CAM
    """

    name: str
    display_name: str
    description: str
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class CaseFindings:
    """
    单张眼底图像对应的结构化 findings。

    这是 classification / CAM 到 report generation 之间的桥梁。
    """

    image_path: str
    prediction: str
    raw_class: str
    confidence: float
    topk_predictions: List[Dict[str, float]] = field(default_factory=list)
    cam_method: Optional[str] = None
    cam_target_layer: Optional[str] = None
    cam_output_path: Optional[str] = None
    findings: List[Finding] = field(default_factory=list)
    disclaimer: str = (
        "本结果由模型自动生成，仅用于研究与展示，不构成临床诊断。"
        "具体判断需结合眼科医生意见。"
    )

    def to_dict(self) -> Dict:
        """转换为普通 dict，便于保存为 JSON 或传给 report provider。"""
        return {
            "image_path": self.image_path,
            "prediction": self.prediction,
            "raw_class": self.raw_class,
            "confidence": self.confidence,
            "topk_predictions": self.topk_predictions,
            "cam_method": self.cam_method,
            "cam_target_layer": self.cam_target_layer,
            "cam_output_path": self.cam_output_path,
            "findings": [
                {
                    "name": f.name,
                    "display_name": f.display_name,
                    "description": f.description,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
            "disclaimer": self.disclaimer,
        }
