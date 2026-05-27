"""v0.6.1 guarded report renderer。

renderer.py 是 guarded report pipeline 的编排层。

它只负责串联：

- prompt_builder
- provider
- safety_checker

并写出 v0.6.1 的报告产物：

- reports/template.md
- reports/template.html
- reports/llm_raw.md
- reports/llm_checked.md，仅 safety pass 时生成
- reports/llm_guarded.html，仅 safety pass 时生成
- safety_report.json
- report.md
- report.html

注意：
    renderer.py 不新增 prompt 规则，不新增 safety rule，不调用真实 LLM API。
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning.llm_report.prompt_builder import build_guarded_report_prompt, load_case_data
from reasoning.llm_report.provider import MockLLMMode, ReportProviderName, get_report_provider
from reasoning.llm_report.safety_checker import RuleBasedSafetyChecker


SAFETY_POLICY_VERSION = "v0.6.2-rule-based-safety-guard"
CHECKER_VERSION = "v0.6.2-rule-based-safety-checker"


@dataclass(frozen=True)
class ReportRenderResult:
    """guarded report 渲染结果摘要。"""

    case_dir: str
    provider: str
    mock_llm_mode: str | None
    safety_passed: bool
    fallback_triggered: bool
    final_report_md_path: str
    final_report_html_path: str
    safety_report_path: str
    generated_files: list[str] = field(default_factory=list)


def render_guarded_report(
    case_dir: Path | str,
    provider_name: ReportProviderName = "mock_llm",
    mock_llm_mode: MockLLMMode = "safe",
) -> ReportRenderResult:
    """渲染 v0.6.1 guarded report artifacts。"""
    case_path = Path(case_dir)
    reports_dir = case_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    case_data = load_case_data(case_path)
    prompt = build_guarded_report_prompt(case_data)

    template_provider = get_report_provider("template")
    template_result = template_provider.generate(prompt=prompt, case_data=case_data)

    provider = get_report_provider(provider_name, mock_llm_mode=mock_llm_mode)
    provider_result = provider.generate(prompt=prompt, case_data=case_data)

    checker = RuleBasedSafetyChecker()
    safety_result = checker.check(provider_result.text)

    template_md_path = reports_dir / "template.md"
    template_html_path = reports_dir / "template.html"
    llm_raw_md_path = reports_dir / "llm_raw.md"
    llm_checked_md_path = reports_dir / "llm_checked.md"
    llm_guarded_html_path = reports_dir / "llm_guarded.html"

    final_md_path = case_path / "report.md"
    final_html_path = case_path / "report.html"
    safety_report_path = case_path / "safety_report.json"

    generated_files: list[Path] = []

    _write_text(template_md_path, template_result.text)
    generated_files.append(template_md_path)

    _write_text(
        template_html_path,
        _render_html_report(
            markdown_text=template_result.text,
            title="Template Case Report",
            generation_status={
                "Report mode": "Template",
                "Provider": template_result.provider,
                "Safety check": "Not applicable",
                "Fallback used": "No",
            },
        ),
    )
    generated_files.append(template_html_path)

    _write_text(llm_raw_md_path, provider_result.text)
    generated_files.append(llm_raw_md_path)

    if safety_result.overall_pass:
        _write_text(llm_checked_md_path, provider_result.text)
        generated_files.append(llm_checked_md_path)

        _write_text(
            llm_guarded_html_path,
            _render_html_report(
                markdown_text=provider_result.text,
                title="Guarded LLM Case Report",
                generation_status={
                    "Report mode": "Guarded LLM Draft",
                    "Provider": provider_result.provider,
                    "Mock LLM mode": str(provider_result.metadata.get("mock_llm_mode", "")),
                    "Safety check": "Passed",
                    "Fallback used": "No",
                },
            ),
        )
        generated_files.append(llm_guarded_html_path)

        shutil.copyfile(llm_checked_md_path, final_md_path)
        shutil.copyfile(llm_guarded_html_path, final_html_path)
    else:
        _remove_if_exists(llm_checked_md_path)
        _remove_if_exists(llm_guarded_html_path)

        shutil.copyfile(template_md_path, final_md_path)
        shutil.copyfile(template_html_path, final_html_path)

    generated_files.extend([final_md_path, final_html_path])

    safety_report = _build_safety_report(
        case_data=case_data,
        prompt=prompt,
        provider_result=provider_result,
        safety_result=safety_result,
        template_md_path=template_md_path,
        llm_raw_md_path=llm_raw_md_path,
        llm_checked_md_path=llm_checked_md_path if safety_result.overall_pass else None,
        llm_guarded_html_path=llm_guarded_html_path if safety_result.overall_pass else None,
        final_md_path=final_md_path,
        final_html_path=final_html_path,
    )

    _write_json(safety_report_path, safety_report)
    generated_files.append(safety_report_path)

    return ReportRenderResult(
        case_dir=str(case_path),
        provider=provider_result.provider,
        mock_llm_mode=(
            str(provider_result.metadata.get("mock_llm_mode"))
            if provider_result.provider == "mock_llm"
            else None
        ),
        safety_passed=safety_result.overall_pass,
        fallback_triggered=safety_result.fallback_triggered,
        final_report_md_path=str(final_md_path),
        final_report_html_path=str(final_html_path),
        safety_report_path=str(safety_report_path),
        generated_files=[str(path) for path in generated_files],
    )


def _build_safety_report(
    case_data: dict[str, Any],
    prompt: str,
    provider_result: Any,
    safety_result: Any,
    template_md_path: Path,
    llm_raw_md_path: Path,
    llm_checked_md_path: Path | None,
    llm_guarded_html_path: Path | None,
    final_md_path: Path,
    final_html_path: Path,
) -> dict[str, Any]:
    """构建 safety_report.json。"""
    selected_output = llm_checked_md_path if safety_result.overall_pass else template_md_path

    decision_reason = (
        "LLM draft passed deterministic safety checks."
        if safety_result.overall_pass
        else "LLM draft failed deterministic safety checks; template fallback was selected."
    )

    prompt_hash = _sha256_text(prompt)
    provider_metadata = dict(provider_result.metadata or {})

    return {
        "case_id": case_data.get("case_id"),
        "provider": provider_result.provider,
        "provider_metadata": provider_metadata,
        "audit_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": prompt_hash,
            "prompt_hash_algorithm": "sha256",
            "prompt_length": len(prompt),
            "provider": provider_result.provider,
            "provider_type": provider_metadata.get("provider_type"),
            "provider_version": provider_metadata.get("provider_version", "not_specified"),
            "deterministic_provider": bool(provider_metadata.get("deterministic", False)),
            "real_llm_used": bool(provider_metadata.get("real_llm_used", False)),
            "checker_version": CHECKER_VERSION,
            "safety_policy_version": SAFETY_POLICY_VERSION,
        },
        "prompt_length": len(prompt),
        "input_report": str(llm_raw_md_path),
        "checked_report": str(llm_checked_md_path) if llm_checked_md_path else None,
        "guarded_html_report": str(llm_guarded_html_path) if llm_guarded_html_path else None,
        "fallback_report": str(template_md_path),
        "final_report_md": str(final_md_path),
        "final_report_html": str(final_html_path),
        "overall_pass": safety_result.overall_pass,
        "fallback_triggered": safety_result.fallback_triggered,
        "safety_result": safety_result.to_dict(),
        "decision": {
            "selected_output": str(selected_output),
            "reason": decision_reason,
        },
        "artifact_policy": {
            "full_fallback_on_any_unsafe_claim": True,
            "partial_repair_enabled": False,
            "raw_llm_output_retained_for_audit": True,
        },
    }


def _sha256_text(text: str) -> str:
    """计算文本的 SHA-256 哈希，用于审计 prompt 是否变化。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_html_report(
    markdown_text: str,
    title: str,
    generation_status: dict[str, str],
) -> str:
    """将 Markdown 草稿包装成简化 HTML 页面。

    v0.6.1 重点展示 guarded generation 状态和 safety trace，
    暂不追求替代 v0.6.0 的卡片式病例报告 UI。
    """
    status_items = "\n".join(
        f"<li><strong>{html.escape(key)}:</strong> {html.escape(value)}</li>"
        for key, value in generation_status.items()
    )
    escaped_markdown = html.escape(markdown_text)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 960px;
      margin: 32px auto;
      padding: 0 20px;
      line-height: 1.6;
      color: #222;
      background: #f7f8fa;
    }}
    .card {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 13px;
      font-weight: 600;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      padding: 16px;
      border-radius: 10px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="card">
    <span class="badge">v0.6.1 Guarded Generation</span>
    <h1>{html.escape(title)}</h1>
    <h2>Generation Metadata</h2>
    <ul>
      {status_items}
    </ul>
  </div>

  <div class="card">
    <h2>Report Draft</h2>
    <pre>{escaped_markdown}</pre>
  </div>
</body>
</html>
"""


def _write_text(path: Path, content: str) -> None:
    """写入 UTF-8 文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, content: dict[str, Any]) -> None:
    """写入格式化 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _remove_if_exists(path: Path) -> None:
    """如果文件存在则删除。"""
    if path.exists():
        path.unlink()
