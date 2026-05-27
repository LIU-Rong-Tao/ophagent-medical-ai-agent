"""v0.6.3 RealLLMProvider 最小回归测试。

这些测试不调用真实 LLM API，只验证 provider 的配置失败路径是否清楚。
"""

import os
import unittest
from unittest.mock import patch

from reasoning.llm_report.provider import get_report_provider


class TestRealLLMProvider(unittest.TestCase):
    """OpenAI-compatible real LLM provider 的基础行为测试。"""

    @patch.dict(os.environ, {}, clear=True)
    def test_real_llm_provider_requires_api_key(self):
        """未配置 OPHAGENT_LLM_API_KEY 时应明确报错。"""
        provider = get_report_provider("real_llm")

        with self.assertRaisesRegex(RuntimeError, "OPHAGENT_LLM_API_KEY is not set"):
            provider.generate(prompt="test prompt", case_data={})

    @patch.dict(
        os.environ,
        {
            "OPHAGENT_LLM_API_KEY": "dummy-key-for-test",
        },
        clear=True,
    )
    def test_real_llm_provider_requires_model_name(self):
        """已配置 API key 但未配置 OPHAGENT_LLM_MODEL 时应明确报错。"""
        provider = get_report_provider("real_llm")

        with self.assertRaisesRegex(RuntimeError, "OPHAGENT_LLM_MODEL is not set"):
            provider.generate(prompt="test prompt", case_data={})


if __name__ == "__main__":
    unittest.main()
