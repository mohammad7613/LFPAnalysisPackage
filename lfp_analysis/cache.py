"""Persistent caching utilities for the LFP analysis pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Optional dependency; only used for type coercion when present.
    import numpy as _np
except Exception:  # pragma: no cover - numpy might be unavailable.
    _np = None


_DEFAULT_CACHE_FILENAME = ".lfp_cache.pkl"


class PipelineCache:
    """Stores feature results keyed by a fingerprint of the pipeline config."""

    def __init__(
        self,
        cache_path: Optional[os.PathLike[str] | str] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        default_dir = Path(__file__).resolve().parent
        raw_path = Path(cache_path).expanduser() if cache_path else default_dir
        if not raw_path.is_absolute():
            raw_path = Path.cwd() / raw_path

        if raw_path.suffix:  # path already points to a file; honor it.
            self.cache_dir = raw_path.parent
            self.cache_path = raw_path
        else:  # treat provided path as a directory and append the cache filename.
            self.cache_dir = raw_path
            self.cache_path = self.cache_dir / _DEFAULT_CACHE_FILENAME

        self._store: Dict[str, Dict[str, Any]] = {}
        if self.enabled:
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            with self.cache_path.open("rb") as stream:
                data = pickle.load(stream)
                if isinstance(data, dict):
                    self._store = data  # type: ignore[assignment]
        except FileNotFoundError:
            self._store = {}
        except Exception as exc:  # pragma: no cover - guard against corrupt caches.
            warnings.warn(
                f"Could not load pipeline cache from {self.cache_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._store = {}

    def _dump(self) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as stream:
            pickle.dump(self._store, stream, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.cache_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_fingerprint(
        self,
        datasets_cfg: Any,
        preprocessors_cfg: Any,
        features_cfg: Any,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        payload = {
            "datasets": self._normalize(datasets_cfg),
            "preprocessors": self._normalize(preprocessors_cfg),
            "features": self._normalize(features_cfg),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_pipeline_results(self, fingerprint: Optional[str]) -> Dict[str, Any]:
        if not self.enabled or not fingerprint:
            return {}
        cached = self._store.get(fingerprint, {})
        return dict(cached)

    def set_pipeline_results(self, fingerprint: Optional[str], results: Dict[str, Any]) -> None:
        if not self.enabled or not fingerprint:
            return
        self._store[fingerprint] = dict(results)
        self._dump()

    def clear(self) -> None:
        if not self.enabled:
            return
        self._store.clear()
        if self.cache_path.exists():
            self.cache_path.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(value: Any) -> Any:
        from collections.abc import Mapping, Sequence

        if isinstance(value, Mapping):
            return {str(k): PipelineCache._normalize(v) for k, v in value.items()}
        if isinstance(value, set):
            return sorted(PipelineCache._normalize(v) for v in value)
        if isinstance(value, (list, tuple)):
            return [PipelineCache._normalize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if _np is not None:
            if isinstance(value, _np.generic):  # numpy scalar
                return value.item()
            if isinstance(value, _np.ndarray):
                return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)


__all__ = ["PipelineCache"]
