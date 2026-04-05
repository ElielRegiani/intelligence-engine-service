"""Load versioned sklearn model + feature column order from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import joblib

from infrastructure.config.settings import Settings, get_settings


@dataclass
class LoadedModel:
    model: Any
    feature_columns: List[str]
    model_path: Path


class ModelLoader:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def _metadata_path(self) -> Path:
        return self.settings.models_dir / self.settings.metadata_filename

    def read_metadata(self) -> dict:
        path = self._metadata_path()
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_current(self) -> LoadedModel:
        meta = self.read_metadata()
        name = meta.get("current_model")
        if not name:
            raise FileNotFoundError("No current_model in metadata.json")
        path = self.settings.models_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}")
        bundle = joblib.load(path)
        model = bundle["model"]
        feature_columns = list(bundle["feature_columns"])
        return LoadedModel(model=model, feature_columns=feature_columns, model_path=path)
