"""v0.6.1 规则型报告安全检查器。

本模块实现确定性的 post-generation safety checker，用于检查 Mock LLM
报告草稿是否越过 findings.json 所允许的证据边界。

注意：
    RuleBasedSafetyChecker 不是完整 hallucination detector。
    它只覆盖本项目中高风险、可规则化识别的错误类型，例如：
    临床诊断越权、CAM 夸大、病灶定位夸大、遗漏免责声明、图像质量夸大等。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlaggedClaim:
    """被安全检查器标记的问题声明。"""

    claim_type: str
    text: str
    reason: str


@dataclass
class RuleCheckResult:
    """单条规则的检查结果。"""

    passed: bool
    matches: list[FlaggedClaim] = field(default_factory=list)


@dataclass
class SafetyCheckResult:
    """用于生成 safety_report.json 的结构化检查结果。"""

    overall_pass: bool
    fallback_triggered: bool
    rule_checks: dict[str, RuleCheckResult]
    flagged_claims: list[FlaggedClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的 dict。"""
        return {
            "overall_pass": self.overall_pass,
            "fallback_triggered": self.fallback_triggered,
            "rule_checks": {
                name: {
                    "passed": result.passed,
                    "matches": [asdict(claim) for claim in result.matches],
                }
                for name, result in self.rule_checks.items()
            },
            "flagged_claims": [asdict(claim) for claim in self.flagged_claims],
            "warnings": self.warnings,
            "known_limitations": self.known_limitations,
        }


class RuleBasedSafetyChecker:
    """确定性的高风险报告声明检查器。"""

    def check(self, draft_text: str) -> SafetyCheckResult:
        """检查生成草稿是否包含不安全声明。

        Args:
            draft_text: Provider 生成的 Markdown 报告草稿。

        Returns:
            SafetyCheckResult: 结构化安全检查结果。
        """
        checks = {
            "clinical_diagnosis_overclaim": self._check_clinical_diagnosis_overclaim(draft_text),
            "cam_or_heatmap_overclaim": self._check_cam_or_heatmap_overclaim(draft_text),
            "unsupported_lesion_localization": self._check_unsupported_lesion_localization(draft_text),
            "missing_non_clinical_use_statement": self._check_non_clinical_use_statement(draft_text),
            "missing_human_review_statement": self._check_human_review_statement(draft_text),
            "image_quality_overclaim": self._check_image_quality_overclaim(draft_text),
            "clinical_use_overclaim": self._check_clinical_use_overclaim(draft_text),
        }

        flagged_claims: list[FlaggedClaim] = []
        for result in checks.values():
            flagged_claims.extend(result.matches)

        overall_pass = not flagged_claims

        warnings = []
        if not overall_pass:
            warnings.append(
                "Fallback template report should be used because the generated draft failed safety checks."
            )

        return SafetyCheckResult(
            overall_pass=overall_pass,
            fallback_triggered=not overall_pass,
            rule_checks=checks,
            flagged_claims=flagged_claims,
            warnings=warnings,
            known_limitations=[
                "Rule-based checks only cover predefined high-risk claim patterns.",
                "Subtle semantic overclaims may require a future LLM-as-judge checker.",
                "This checker does not verify clinical correctness.",
            ],
        )

    def _check_clinical_diagnosis_overclaim(self, text: str) -> RuleCheckResult:
        patterns = [
            r"\bpatient is diagnosed with\b[^.\n]*",
            r"\bdiagnosed with\b[^.\n]*",
            r"\bdefinitive diagnosis\b[^.\n]*",
            r"\bclinical diagnosis\b[^.\n]*",
            r"确诊[^。\n]*",
            r"诊断为[^。\n]*",
            r"临床诊断[^。\n]*",
        ]
        return self._match_patterns(
            text=text,
            patterns=patterns,
            claim_type="clinical_diagnosis_overclaim",
            reason=(
                "The system can provide model predictions and evidence summaries, "
                "but it must not make clinical diagnosis claims."
            ),
        )

    def _check_cam_or_heatmap_overclaim(self, text: str) -> RuleCheckResult:
        patterns = [
            r"\bCAM\b[^.\n]*(localizes|confirms|shows the lesion|shows lesion|identifies)[^.\n]*",
            r"\bheatmap\b[^.\n]*(localizes|confirms|shows the lesion|shows lesion|identifies)[^.\n]*",
            r"\bCAM\b[^.\n]*lesion area[^.\n]*",
            r"\bheatmap\b[^.\n]*lesion area[^.\n]*",
            r"CAM[^。\n]*(定位|确认|显示病灶|显示病变)[^。\n]*",
            r"热力图[^。\n]*(定位|确认|显示病灶|显示病变)[^。\n]*",
        ]
        return self._match_patterns(
            text=text,
            patterns=patterns,
            claim_type="cam_or_heatmap_overclaim",
            reason=(
                "CAM or heatmap output is weak visual evidence only and must not be "
                "described as lesion localization or lesion confirmation."
            ),
        )

    def _check_unsupported_lesion_localization(self, text: str) -> RuleCheckResult:
        patterns = [
            r"\blocalizes retinal lesions\b[^.\n]*",
            r"\blesion localization\b[^.\n]*",
            r"\blesion area responsible\b[^.\n]*",
            r"\babnormal retinal regions\b[^.\n]*confirm[^.\n]*",
            r"定位[^。\n]*(病灶|病变)[^。\n]*",
            r"(病灶|病变)[^。\n]*定位[^。\n]*",
        ]
        return self._match_patterns(
            text=text,
            patterns=patterns,
            claim_type="unsupported_lesion_localization",
            reason=(
                "The current pipeline does not provide lesion-level annotation or "
                "validated lesion localization."
            ),
        )

    def _check_non_clinical_use_statement(self, text: str) -> RuleCheckResult:
        required_patterns = [
            r"\bnot for clinical use\b",
            r"不用于临床",
            r"非临床用途",
        ]

        if self._contains_any(text, required_patterns):
            return RuleCheckResult(passed=True)

        claim = FlaggedClaim(
            claim_type="missing_non_clinical_use_statement",
            text="Missing non-clinical-use statement.",
            reason="The report must explicitly state that it is not for clinical use.",
        )
        return RuleCheckResult(passed=False, matches=[claim])

    def _check_human_review_statement(self, text: str) -> RuleCheckResult:
        required_patterns = [
            r"\bhuman review is required\b",
            r"\brequires human review\b",
            r"需要人工审核",
            r"需要人工复核",
            r"人工审核",
        ]

        if self._contains_any(text, required_patterns):
            return RuleCheckResult(passed=True)

        claim = FlaggedClaim(
            claim_type="missing_human_review_statement",
            text="Missing human-review-required statement.",
            reason="The report must explicitly state that human review is required.",
        )
        return RuleCheckResult(passed=False, matches=[claim])

    def _check_image_quality_overclaim(self, text: str) -> RuleCheckResult:
        patterns = [
            r"\bimage quality is validated\b[^.\n]*",
            r"\bvalidated as sufficient for clinical decision-making\b[^.\n]*",
            r"\bsufficient for clinical decision-making\b[^.\n]*",
            r"图像质量[^。\n]*(已验证|足以|适合临床决策)[^。\n]*",
        ]
        return self._match_patterns(
            text=text,
            patterns=patterns,
            claim_type="image_quality_overclaim",
            reason=(
                "The report must not overclaim image quality or state that the image "
                "is sufficient for clinical decision-making unless explicitly supported."
            ),
        )

    def _check_clinical_use_overclaim(self, text: str) -> RuleCheckResult:
        patterns = [
            r"\bcan be used as a clinical reference\b[^.\n]*",
            r"\bcan be used for clinical\b[^.\n]*",
            r"\bfor clinical decision-making\b[^.\n]*",
            r"可用于临床[^。\n]*",
            r"临床参考[^。\n]*",
            r"临床决策[^。\n]*",
        ]
        return self._match_patterns(
            text=text,
            patterns=patterns,
            claim_type="clinical_use_overclaim",
            reason=(
                "The artifact is for non-clinical research/demo use and must not be "
                "described as clinical reference or clinical decision support."
            ),
        )

    def _match_patterns(
        self,
        text: str,
        patterns: list[str],
        claim_type: str,
        reason: str,
    ) -> RuleCheckResult:
        raw_matches: list[tuple[int, int, str]] = []

        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                matched_text = match.group(0).strip()
                if not matched_text:
                    continue

                if self._is_safe_negated_context(text, match.start(), match.end()):
                    continue

                raw_matches.append((match.start(), match.end(), matched_text))

        # 同一类规则内，保留更长的重叠命中，避免 safety_report.json 里出现重复片段。
        deduped_matches = self._dedupe_overlapping_matches(raw_matches)

        matches = [
            FlaggedClaim(
                claim_type=claim_type,
                text=matched_text,
                reason=reason,
            )
            for _, _, matched_text in deduped_matches
        ]

        return RuleCheckResult(passed=not matches, matches=matches)

    def _dedupe_overlapping_matches(
        self,
        raw_matches: list[tuple[int, int, str]],
    ) -> list[tuple[int, int, str]]:
        """对同一规则类型下的重叠命中进行去重。

        策略：
        - 先按文本长度从长到短排序；
        - 如果一个短命中被更长命中覆盖，则丢弃短命中；
        - 最后按原文出现顺序返回。
        """
        kept: list[tuple[int, int, str]] = []

        for start, end, matched_text in sorted(
            raw_matches,
            key=lambda item: (item[1] - item[0]),
            reverse=True,
        ):
            overlaps_existing = any(
                not (end <= kept_start or start >= kept_end)
                for kept_start, kept_end, _ in kept
            )

            if not overlaps_existing:
                kept.append((start, end, matched_text))

        return sorted(kept, key=lambda item: item[0])

    def _is_safe_negated_context(self, text: str, start: int, end: int) -> bool:
        """判断命中的风险短语是否处在安全的否定/边界声明上下文中。

        例如：
        - "should not be interpreted as lesion localization"
        - "does not establish a clinical diagnosis"

        这些句子虽然包含风险词，但实际是在声明边界，不能当作 unsafe claim。
        """
        window_start = max(0, start - 120)
        window_end = min(len(text), end + 120)
        context = text[window_start:window_end].lower()

        safe_patterns = [
            r"does not establish[^.\n]*(clinical diagnosis|diagnosis)",
            r"does not claim[^.\n]*(lesion|location|localization|diagnosis)",
            r"should not be interpreted as[^.\n]*(lesion|localization|diagnosis)",
            r"must not be described as[^.\n]*(lesion|localization|diagnosis)",
            r"not lesion segmentation[^.\n]*localization",
            r"not[^.\n]*(lesion segmentation|lesion localization|clinical diagnosis)",
            r"不能[^。\n]*(诊断|定位|病灶)",
            r"不应[^。\n]*(诊断|定位|病灶)",
            r"不构成[^。\n]*(诊断|定位)",
            r"不用于临床",
        ]

        return any(re.search(pattern, context, flags=re.IGNORECASE) for pattern in safe_patterns)

    def _contains_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
