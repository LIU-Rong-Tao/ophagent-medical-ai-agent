"""
Report provider 抽象基类。

v0.2.2 的设计目标：
- rule_based provider 默认可用
- openai provider 可选
- 后续可以扩展更多 provider
"""

from abc import ABC, abstractmethod

from findings.finding_schema import CaseFindings


class BaseReportProvider(ABC):
    """所有 report provider 都需要实现这个接口。"""

    @abstractmethod
    def generate_report(self, case_findings: CaseFindings) -> str:
        """
        根据结构化 findings 生成中文 summary。

        返回：
            str: 中文 clinical-style summary
        """
        raise NotImplementedError
