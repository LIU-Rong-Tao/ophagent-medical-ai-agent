"""手动 smoke test：验证 real_llm provider 是否能调用 OpenAI-compatible endpoint。

使用方式：

    OPHAGENT_LLM_API_KEY=xxx \
    OPHAGENT_LLM_BASE_URL=https://api.openai.com/v1 \
    OPHAGENT_LLM_MODEL=your-model \
    python scripts/dev/smoke_real_llm_provider.py

注意：
- 本脚本不会在默认测试中运行。
- 不要提交任何 API key。
- 输出只用于手动检查 provider 是否可用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from reasoning.llm_report.provider import get_report_provider  # noqa: E402


def main() -> None:
    """运行最小真实 LLM provider smoke test。"""
    provider = get_report_provider("real_llm")

    prompt = (
        "Generate one cautious sentence for a non-clinical ophthalmology research/demo "
        "report. The sentence must include: This report is not for clinical use."
    )

    result = provider.generate(prompt=prompt, case_data={})

    output = {
        "provider": result.provider,
        "text_preview": result.text[:500],
        "metadata": result.metadata,
    }

    output_path = Path("/tmp/ophagent_real_llm_smoke_result.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("provider:", result.provider)
    print("text_preview:", result.text[:500])
    print("metadata:", result.metadata)
    print("saved_to:", output_path)


if __name__ == "__main__":
    main()
