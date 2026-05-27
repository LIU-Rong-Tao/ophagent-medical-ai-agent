"""v0.6.3 real LLM provider 与 renderer 的离线集成测试。

这些测试通过 mock `urllib.request.urlopen` 模拟 OpenAI-compatible response，
不调用真实网络、不需要 API key、不依赖真实 LLM 服务。
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reasoning.llm_report.renderer import render_guarded_report


class FakeOpenAICompatibleResponse:
    """模拟 OpenAI-compatible chat completions 响应。"""

    def __init__(self, content: str) -> None:
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": self.content,
                    }
                }
            ]
        }
        return json.dumps(payload).encode("utf-8")


class TestRealLLMRendererIntegration(unittest.TestCase):
    """real_llm provider 接入 renderer 后的离线集成测试。"""

    def setUp(self):
        """复制一个临时 case 目录，避免污染已提交的 example case。"""
        self.source_case_dir = Path("experiments/case_reports/d9bbdc33db83")
        self.temp_root = Path(tempfile.mkdtemp(prefix="ophagent_real_llm_test_"))
        self.case_dir = self.temp_root / "case"
        shutil.copytree(self.source_case_dir, self.case_dir)

    def tearDown(self):
        """清理临时目录。"""
        shutil.rmtree(self.temp_root)

    @patch.dict(
        os.environ,
        {
            "OPHAGENT_LLM_API_KEY": "dummy-secret-key",
            "OPHAGENT_LLM_BASE_URL": "https://example.test/v1",
            "OPHAGENT_LLM_MODEL": "dummy-model",
            "OPHAGENT_LLM_TEMPERATURE": "0",
            "OPHAGENT_LLM_TOP_P": "1",
            "OPHAGENT_LLM_TIMEOUT_SECONDS": "30",
        },
        clear=True,
    )
    @patch("urllib.request.urlopen")
    def test_real_llm_renderer_safe_response_writes_audit_report(self, mock_urlopen):
        """safe real_llm draft 应通过 renderer，并写出 audit metadata。"""
        safe_draft = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The model prediction suggests a possible disease-related risk pattern in the fundus image.

## Weak Visual Evidence

The CAM overlay is provided as weak visual evidence only. It should not be interpreted as lesion localization or lesion segmentation.

## Safety Statement

This report is not for clinical use. Human review is required before any medical interpretation or downstream decision.
"""
        mock_urlopen.return_value = FakeOpenAICompatibleResponse(safe_draft)

        result = render_guarded_report(
            case_dir=self.case_dir,
            provider_name="real_llm",
            mock_llm_mode="safe",
        )

        safety_report = json.loads(
            (self.case_dir / "safety_report.json").read_text(encoding="utf-8")
        )

        self.assertTrue(result.safety_passed)
        self.assertFalse(result.fallback_triggered)
        self.assertEqual(safety_report["provider"], "real_llm")
        self.assertTrue(safety_report["overall_pass"])
        self.assertFalse(safety_report["fallback_triggered"])

        audit_metadata = safety_report["audit_metadata"]
        self.assertEqual(audit_metadata["provider"], "real_llm")
        self.assertEqual(audit_metadata["provider_type"], "real_llm")
        self.assertEqual(
            audit_metadata["provider_version"],
            "v0.6.3-openai-compatible-provider",
        )
        self.assertTrue(audit_metadata["real_llm_used"])
        self.assertFalse(audit_metadata["deterministic_provider"])
        self.assertEqual(audit_metadata["checker_version"], "v0.6.2-rule-based-safety-checker")
        self.assertEqual(audit_metadata["safety_policy_version"], "v0.6.2-rule-based-safety-guard")
        self.assertEqual(len(audit_metadata["prompt_hash"]), 64)

        self.assertTrue((self.case_dir / "reports" / "llm_raw.md").exists())
        self.assertTrue((self.case_dir / "reports" / "llm_checked.md").exists())
        self.assertTrue((self.case_dir / "reports" / "llm_guarded.html").exists())

        metadata_text = json.dumps(safety_report, ensure_ascii=False)
        self.assertNotIn("dummy-secret-key", metadata_text)

    @patch.dict(
        os.environ,
        {
            "OPHAGENT_LLM_API_KEY": "dummy-secret-key",
            "OPHAGENT_LLM_BASE_URL": "https://example.test/v1",
            "OPHAGENT_LLM_MODEL": "dummy-model",
            "OPHAGENT_LLM_TEMPERATURE": "0",
            "OPHAGENT_LLM_TOP_P": "1",
            "OPHAGENT_LLM_TIMEOUT_SECONDS": "30",
        },
        clear=True,
    )
    @patch("urllib.request.urlopen")
    def test_real_llm_renderer_unsafe_response_triggers_fallback(self, mock_urlopen):
        """unsafe real_llm draft 应触发 fallback，并不生成 checked/guarded 产物。"""
        unsafe_draft = """
# Guarded LLM Case Report Draft

## Interpretation Summary

The patient is diagnosed with diabetic retinopathy based on the fundus image and model prediction.

## Safety Statement

This report is not for clinical use.
"""
        mock_urlopen.return_value = FakeOpenAICompatibleResponse(unsafe_draft)

        result = render_guarded_report(
            case_dir=self.case_dir,
            provider_name="real_llm",
            mock_llm_mode="safe",
        )

        safety_report = json.loads(
            (self.case_dir / "safety_report.json").read_text(encoding="utf-8")
        )

        self.assertFalse(result.safety_passed)
        self.assertTrue(result.fallback_triggered)
        self.assertFalse(safety_report["overall_pass"])
        self.assertTrue(safety_report["fallback_triggered"])
        self.assertEqual(safety_report["provider"], "real_llm")

        flagged_types = {
            claim["claim_type"]
            for claim in safety_report["safety_result"]["flagged_claims"]
        }
        self.assertIn("clinical_diagnosis_overclaim", flagged_types)
        self.assertIn("missing_human_review_statement", flagged_types)

        self.assertTrue((self.case_dir / "reports" / "llm_raw.md").exists())
        self.assertFalse((self.case_dir / "reports" / "llm_checked.md").exists())
        self.assertFalse((self.case_dir / "reports" / "llm_guarded.html").exists())

        metadata_text = json.dumps(safety_report, ensure_ascii=False)
        self.assertNotIn("dummy-secret-key", metadata_text)


if __name__ == "__main__":
    unittest.main()
