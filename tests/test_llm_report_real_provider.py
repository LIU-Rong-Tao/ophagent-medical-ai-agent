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


class TestRealLLMProviderMockedResponse(unittest.TestCase):
    """使用 mock response 测试 real LLM provider 的成功路径，不发起真实网络请求。"""

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
    def test_real_llm_provider_parses_openai_compatible_response(self, mock_urlopen):
        """real LLM provider 应能解析 OpenAI-compatible chat completions 响应。"""

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return (
                    b'{"choices":[{"message":{"content":"Mocked real LLM draft."}}]}'
                )

        mock_urlopen.return_value = FakeResponse()

        provider = get_report_provider("real_llm")
        result = provider.generate(prompt="test prompt", case_data={})

        self.assertEqual(result.provider, "real_llm")
        self.assertEqual(result.text, "Mocked real LLM draft.")

        self.assertEqual(result.metadata["provider_type"], "real_llm")
        self.assertEqual(
            result.metadata["provider_version"],
            "v0.6.3-openai-compatible-provider",
        )
        self.assertTrue(result.metadata["real_llm_used"])
        self.assertFalse(result.metadata["deterministic"])
        self.assertEqual(result.metadata["model_name"], "dummy-model")
        self.assertEqual(result.metadata["temperature"], 0.0)
        self.assertEqual(result.metadata["top_p"], 1.0)
        self.assertEqual(result.metadata["timeout_seconds"], 30.0)

        metadata_text = str(result.metadata)
        self.assertNotIn("dummy-secret-key", metadata_text)
