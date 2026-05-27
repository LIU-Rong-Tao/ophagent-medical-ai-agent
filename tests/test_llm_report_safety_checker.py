"""v0.6.2 RuleBasedSafetyChecker 回归测试。

这些测试用于固定 v0.6.1 guarded generation 中最关键的安全检查行为。

注意：
- 这里测试的是 rule-based safety guard，不是真实 hallucination detector。
- 测试文本是 deterministic mock draft，用于覆盖高风险、可规则化识别的 unsafe claim。
- 后续接入真实 LLM 前，应先保证这些基础安全规则不会回退。
"""

import unittest

from reasoning.llm_report.safety_checker import RuleBasedSafetyChecker


SAFE_DRAFT = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The model prediction suggests a possible disease-related risk pattern in the fundus image. This statement is based only on the available structured findings and model output.

## Weak Visual Evidence

The CAM overlay is provided as weak visual evidence. It should not be interpreted as lesion localization or lesion segmentation.

## Safety Statement

This report is not for clinical use. Human review is required before any medical interpretation or downstream decision.
"""


UNSAFE_DIAGNOSIS_DRAFT = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The patient is diagnosed with diabetic retinopathy based on the fundus image and model prediction.

## Weak Visual Evidence

The CAM overlay is provided as supporting visual evidence.

## Safety Statement

This report is not for clinical use.
"""


UNSAFE_CAM_DRAFT = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The model prediction suggests disease-related risk.

## CAM Evidence

The CAM heatmap localizes retinal lesions and shows the lesion area responsible for the disease.

## Safety Statement

This report is not for clinical use. Human review is required before any medical interpretation or downstream decision.
"""


UNSAFE_IMAGE_QUALITY_DRAFT = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The image quality is validated as sufficient for clinical decision-making.

## Safety Statement

This report is not for clinical use. Human review is required before any medical interpretation or downstream decision.
"""


UNSAFE_CLINICAL_USE_DRAFT = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The model prediction suggests disease-related risk.

## Safety Statement

This report can be used as a clinical reference. Human review is required before any medical interpretation or downstream decision.
"""


class TestLLMReportSafetyChecker(unittest.TestCase):
    """RuleBasedSafetyChecker 基础回归测试。"""

    def setUp(self):
        """每个测试使用新的 checker 实例，避免状态污染。"""
        self.checker = RuleBasedSafetyChecker()

    def test_safe_draft_passes(self):
        """安全草稿应通过检查。"""
        result = self.checker.check(SAFE_DRAFT)

        self.assertTrue(result.overall_pass)
        self.assertFalse(result.fallback_triggered)
        self.assertEqual(result.flagged_claims, [])

    def test_unsafe_diagnosis_fails(self):
        """越权诊断草稿应触发回退。"""
        result = self.checker.check(UNSAFE_DIAGNOSIS_DRAFT)

        self.assertFalse(result.overall_pass)
        self.assertTrue(result.fallback_triggered)
        self.assertTrue(
            any(
                claim.claim_type == "clinical_diagnosis_overclaim"
                for claim in result.flagged_claims
            )
        )

    def test_unsafe_cam_overclaim_fails(self):
        """CAM / heatmap 夸大草稿应触发回退。"""
        result = self.checker.check(UNSAFE_CAM_DRAFT)

        self.assertFalse(result.overall_pass)
        self.assertTrue(result.fallback_triggered)
        self.assertTrue(
            any(
                claim.claim_type == "cam_or_heatmap_overclaim"
                for claim in result.flagged_claims
            )
        )
        self.assertTrue(
            any(
                claim.claim_type == "unsupported_lesion_localization"
                for claim in result.flagged_claims
            )
        )

    def test_image_quality_overclaim_fails(self):
        """图像质量临床用途夸大应触发回退。"""
        result = self.checker.check(UNSAFE_IMAGE_QUALITY_DRAFT)

        self.assertFalse(result.overall_pass)
        self.assertTrue(result.fallback_triggered)
        self.assertTrue(
            any(
                claim.claim_type == "image_quality_overclaim"
                for claim in result.flagged_claims
            )
        )

    def test_clinical_use_overclaim_fails(self):
        """临床用途夸大应触发回退。"""
        result = self.checker.check(UNSAFE_CLINICAL_USE_DRAFT)

        self.assertFalse(result.overall_pass)
        self.assertTrue(result.fallback_triggered)
        self.assertTrue(
            any(
                claim.claim_type == "clinical_use_overclaim"
                for claim in result.flagged_claims
            )
        )

    def test_missing_non_clinical_statement_fails(self):
        """缺少非临床用途声明应触发回退。"""
        draft = """
The model prediction suggests disease-related risk.

Human review is required before any medical interpretation or downstream decision.
"""
        result = self.checker.check(draft)

        self.assertFalse(result.overall_pass)
        self.assertTrue(result.fallback_triggered)
        self.assertTrue(
            any(
                claim.claim_type == "missing_non_clinical_use_statement"
                for claim in result.flagged_claims
            )
        )

    def test_missing_human_review_statement_fails(self):
        """缺少人工审核声明应触发回退。"""
        draft = """
The model prediction suggests disease-related risk.

This report is not for clinical use.
"""
        result = self.checker.check(draft)

        self.assertFalse(result.overall_pass)
        self.assertTrue(result.fallback_triggered)
        self.assertTrue(
            any(
                claim.claim_type == "missing_human_review_statement"
                for claim in result.flagged_claims
            )
        )


if __name__ == "__main__":
    unittest.main()
