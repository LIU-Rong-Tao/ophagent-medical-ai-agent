from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .base import BaseModelProvider
from .records import ProviderHealth, UnifiedModelRecord


@dataclass(frozen=True)
class ProviderCatalog:
    records: tuple[UnifiedModelRecord, ...]
    health: tuple[ProviderHealth, ...]


def build_provider_catalog(providers: Iterable[BaseModelProvider]) -> ProviderCatalog:
    records = []
    health = []
    for provider in providers:
        provider_health = provider.health()
        health.append(provider_health)
        if provider_health.available:
            records.extend(provider.list_models())
    records.sort(key=lambda record: record.unified_model_id)
    ids = [record.unified_model_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Unified provider catalog contains duplicate model IDs")
    return ProviderCatalog(tuple(records), tuple(health))
