from __future__ import annotations

from abc import ABC, abstractmethod

from .records import ProviderHealth, UnifiedModelRecord


class BaseModelProvider(ABC):
    provider_id: str

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def list_models(self) -> tuple[UnifiedModelRecord, ...]:
        pass

    @abstractmethod
    def list_checkpoints(self, model_id: str) -> tuple[UnifiedModelRecord, ...]:
        pass

    @abstractmethod
    def get_model(self, unified_model_id: str) -> UnifiedModelRecord:
        pass

    @abstractmethod
    def health(self) -> ProviderHealth:
        pass
