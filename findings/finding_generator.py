"""
OphAgent v0.2.2
从分类结果和 CAM 信息生成结构化 findings。

重要原则：
1. 不做真实病灶检测
2. 不输出“发现病灶”“临床确诊”等表述
3. 只描述模型预测和模型关注区域可能相关的视觉特征
"""

from typing import Dict, List, Optional

from findings.finding_schema import CaseFindings, Finding


_DR_LABEL_TO_CN = {
    "No DR": "未见明显糖尿病视网膜病变倾向",
    "Mild DR": "轻度糖尿病视网膜病变倾向",
    "Moderate DR": "中度糖尿病视网膜病变倾向",
    "Severe DR": "重度糖尿病视网膜病变倾向",
    "Proliferative DR": "增殖期糖尿病视网膜病变倾向",
}


# 不同 DR 分级对应的可能相关视觉线索。
#
# 注意：
# 这里只是基于 DR 分类等级的一般性视觉模式总结，
# 不是病灶检测结果，也不是临床确诊。
#
# wording 必须保持：
# - “样改变”
# - “相关视觉线索”
# - “可能相关”
#
# 避免：
# - “发现病灶”
# - “存在 hemorrhage”
# - “检测到渗出”
_DR_LABEL_TO_VISUAL_CUES = {
    "No DR": [
        "未见明显与糖尿病视网膜病变相关的视觉模式",
    ],
    "Mild DR": [
        "少量微血管瘤样改变",
        "轻微局灶性微血管异常",
    ],
    "Moderate DR": [
        "微血管瘤样改变",
        "点片状出血样视觉线索",
        "硬性渗出样高反射区域",
        "局灶性视网膜微血管异常",
    ],
    "Severe DR": [
        "较广泛的出血样视觉线索",
        "静脉串珠样改变",
        "明显的视网膜微血管异常",
        "棉絮斑样改变",
    ],
    "Proliferative DR": [
        "新生血管相关视觉线索",
        "玻璃体或视网膜前出血相关视觉模式",
        "纤维增殖相关改变",
    ],
}


def get_visual_cues(prediction: str) -> List[str]:
    """根据 DR 分类结果返回可能相关的视觉线索。"""
    return _DR_LABEL_TO_VISUAL_CUES.get(prediction, [])


def _build_main_finding(prediction: str, confidence: float) -> Finding:
    """
    根据分类预测生成主 finding。

    这里不判断真实病灶，只把模型分类结果转成结构化描述。
    """

    display_name = _DR_LABEL_TO_CN.get(prediction, prediction)
    visual_cues = get_visual_cues(prediction)
    visual_cues_text = "、".join(visual_cues)

    description = (
        f"分类模型输出为“{display_name}”。"
        f"该结果表示模型在当前图像上更倾向于该类别。"
    )

    if visual_cues:
        description += (
            f"结合该类别常见眼底影像模式，"
            f"模型关注区域可能与以下视觉线索相关："
            f"{visual_cues_text}。"
        )

    description += (
        "上述描述仅用于模型解释与研究展示，"
        "不能直接作为临床诊断依据。"
    )

    return Finding(
        name="classification_tendency",
        display_name=display_name,
        description=description,
        confidence=float(confidence),
        evidence=["classification_prediction"],
    )


def _build_visual_cue_finding(prediction: str, confidence: float) -> Optional[Finding]:
    """
    根据 DR 分类等级生成可能相关视觉线索 finding。

    这不是病灶检测结果，而是将分类标签对应的常见视觉模式显式展示出来，
    方便后续 report provider 生成更有医学上下文的摘要。
    """

    visual_cues = get_visual_cues(prediction)

    if not visual_cues:
        return None

    description = (
        "根据当前分类结果对应的一般眼底影像学模式，"
        "模型结果可能涉及以下视觉线索："
        + "、".join(visual_cues)
        + "。这些线索仅表示与该分类等级相关的可能视觉模式，"
        "并不代表系统已经完成独立病灶检测。"
    )

    return Finding(
        name="possible_visual_cues",
        display_name="可能相关视觉线索",
        description=description,
        confidence=float(confidence),
        evidence=["classification_label_prior"],
    )


def _build_cam_finding(
    confidence: float,
    cam_method: Optional[str],
    cam_target_layer: Optional[str],
    cam_output_path: Optional[str],
) -> Optional[Finding]:
    """
    根据 CAM 输出生成可解释性 finding。

    注意：
    CAM 只能说明模型关注区域，不能说明那里一定存在具体病灶。
    """

    if cam_output_path is None:
        return None

    method_text = cam_method or "CAM"
    layer_text = cam_target_layer or "未指定层"

    description = (
        f"模型使用 {method_text} 在 {layer_text} 上生成关注区域热力图。"
        f"该热力图用于展示模型决策时较关注的图像区域。"
        f"若关注区域与上述可能视觉线索空间上重叠，"
        f"可作为模型解释的参考，但不能单独作为病灶定位或疾病诊断依据。"
    )

    return Finding(
        name="model_attention_region",
        display_name="模型关注区域",
        description=description,
        confidence=float(confidence),
        evidence=["cam_heatmap"],
    )


def generate_case_findings(
    image_path: str,
    prediction: str,
    raw_class: str,
    confidence: float,
    topk_predictions: Optional[List[Dict[str, float]]] = None,
    cam_method: Optional[str] = None,
    cam_target_layer: Optional[str] = None,
    cam_output_path: Optional[str] = None,
) -> CaseFindings:
    """
    生成单张图像的结构化 findings。

    参数通常来自：
    - infer_classifier 输出
    - gradcam 输出
    """

    findings: List[Finding] = []

    findings.append(
        _build_main_finding(
            prediction=prediction,
            confidence=confidence,
        )
    )

    visual_cue_finding = _build_visual_cue_finding(
        prediction=prediction,
        confidence=confidence,
    )

    if visual_cue_finding is not None:
        findings.append(visual_cue_finding)

    cam_finding = _build_cam_finding(
        confidence=confidence,
        cam_method=cam_method,
        cam_target_layer=cam_target_layer,
        cam_output_path=cam_output_path,
    )

    if cam_finding is not None:
        findings.append(cam_finding)

    return CaseFindings(
        image_path=image_path,
        prediction=prediction,
        raw_class=raw_class,
        confidence=float(confidence),
        topk_predictions=topk_predictions or [],
        cam_method=cam_method,
        cam_target_layer=cam_target_layer,
        cam_output_path=cam_output_path,
        findings=findings,
    )
