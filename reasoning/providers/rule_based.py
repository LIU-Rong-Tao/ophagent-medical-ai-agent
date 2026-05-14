"""
Rule-based report provider。

这是 v0.2.2 的默认 provider。
它负责把 structured findings 转成简洁、可读的中文摘要。
"""

from findings.finding_schema import CaseFindings
from findings.finding_generator import get_visual_cues
from reasoning.providers.base import BaseReportProvider


class RuleBasedReportProvider(BaseReportProvider):
    """基于模板的中文 clinical-style summary 生成器。"""

    def generate_report(self, case_findings: CaseFindings) -> str:
        """根据 CaseFindings 生成简洁中文摘要。"""

        visual_cues = get_visual_cues(case_findings.prediction)

        lines = []

        lines.append("## OphAgent 眼底图像分析摘要")
        lines.append("")
        lines.append(
            f"当前眼底图像的模型预测结果为 **{case_findings.prediction}** "
            f"（置信度：{case_findings.confidence:.4f}）。"
        )
        lines.append(
            "该结果表示模型在当前输入图像上更倾向于该糖尿病视网膜病变分级，"
            "不能直接等同于临床诊断。"
        )

        if visual_cues:
            lines.append("")
            lines.append("结合该类别的一般眼底影像学表现，模型关注区域可能与以下视觉线索相关：")
            for cue in visual_cues:
                lines.append(f"- {cue}")

        if case_findings.cam_output_path:
            lines.append("")
            lines.append(
                f"系统已生成 CAM 热力图，用于展示模型在分类过程中较关注的图像区域。"
            )
            lines.append(
                f"CAM 输出路径：{case_findings.cam_output_path}"
            )

        lines.append("")
        lines.append(
            "需要强调的是，当前系统并未进行独立病灶检测；"
            "上述视觉线索仅为基于分类结果和 CAM 关注区域的解释性总结。"
        )
        lines.append(case_findings.disclaimer)

        return "\n".join(lines)
