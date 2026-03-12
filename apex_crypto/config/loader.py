"""Configuration loader for the APEX Crypto Trading System.

Loads YAML configuration files and environment variables, resolves
``${ENV_VAR}`` placeholders, and exposes a thread-safe singleton
:class:`Config` instance for use throughout the application.

Typical usage::

    from apex_crypto.config.loader import Config

    cfg = Config()
    max_lev = cfg.get("risk.max_leverage", default=1.0)
    tier1 = cfg.get_assets()["tier1"]
    timeframes = cfg.get_timeframes()
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
import yaml
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_YAML = _CONFIG_DIR / "config.yaml"
_ASSETS_YAML = _CONFIG_DIR / "assets.yaml"
_PROJECT_ROOT = _CONFIG_DIR.parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# Keys that *must* be present in config.yaml after loading.
REQUIRED_CONFIG_KEYS: List[str] = [
    "system.name",
    "system.mode",
    "exchange.name",
    "risk.risk_per_trade_pct",
    "risk.max_leverage",
    "risk.max_drawdown_pct",
    "signals.full_position_score",
    "signals.min_agreeing_strategies",
    "data.timescaledb_url",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ``${ENV_VAR}`` references in configuration values.

    Args:
        value: A configuration value — string, dict, list, or scalar.

    Returns:
        The value with all ``${…}`` placeholders replaced by the
        corresponding environment variable.  If the environment variable
        is not set the placeholder is left intact and a warning is logged.
    """
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                logger.warning(
                    "env_var_not_set",
                    variable=var_name,
                    hint="Set it in .env or export it before starting the system.",
                )
                return match.group(0)  # leave placeholder as-is
            return env_val

        return _ENV_VAR_PATTERN.sub(_replacer, value)

    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]

    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Args:
        base: The base dictionary.
        override: The dictionary whose values take precedence.

    Returns:
        A new dictionary containing the merged result.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _traverse(data: Dict[str, Any], key_path: str) -> Any:
    """Walk a nested dict using dot-separated *key_path*.

    Args:
        data: The nested dictionary to traverse.
        key_path: Dot-separated path, e.g. ``"risk.max_leverage"``.

    Returns:
        The value found at the given path.

    Raises:
        KeyError: If any segment of the path is missing.
    """
    current: Any = data
    for segment in key_path.split("."):
        if isinstance(current, dict):
            current = current[segment]
        else:
            raise KeyError(segment)
    return current


# ---------------------------------------------------------------------------
# Config singleton
# ---------------------------------------------------------------------------


class Config:
    """Thread-safe singleton that provides access to the merged configuration.

    The first instantiation loads the YAML files, the ``.env`` file, and
    resolves environment-variable placeholders.  Subsequent calls to
    ``Config()`` return the same instance.

    Examples:
        >>> cfg = Config()
        >>> cfg.get("exchange.name")
        'mexc'
        >>> cfg.get("risk.max_leverage", default=1.0)
        3.0
        >>> cfg.get_assets()["tier1"]
        [{'symbol': 'BTC/USDT', ...}, ...]
    """

    _instance: Optional[Config] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # ---- singleton machinery ------------------------------------------------

    def __new__(cls, *args: Any, **kwargs: Any) -> Config:
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking.
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        assets_path: Optional[Union[str, Path]] = None,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialise the configuration (only runs once).

        Args:
            config_path: Override path for ``config.yaml``.
            assets_path: Override path for ``assets.yaml``.
            dotenv_path: Override path for the ``.env`` file.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._config_path = Path(config_path) if config_path else _CONFIG_YAML
            self._assets_path = Path(assets_path) if assets_path else _ASSETS_YAML
            self._dotenv_path = Path(dotenv_path) if dotenv_path else _DOTENV_PATH

            self._data: Dict[str, Any] = {}
            self._assets_data: Dict[str, Any] = {}

            self._load()
            self._initialized = True

    # ---- loading ------------------------------------------------------------

    def _load(self) -> None:
        """Load .env, YAML files, resolve env vars, and validate."""
        # 1. Environment variables from .env
        self._load_dotenv()

        # 2. Main configuration
        self._data = self._load_yaml(self._config_path, label="config")

        # 3. Assets configuration
        self._assets_data = self._load_yaml(self._assets_path, label="assets")

        # 4. Resolve ${ENV_VAR} placeholders
        self._data = _resolve_env_vars(self._data)
        self._assets_data = _resolve_env_vars(self._assets_data)

        # 5. Validate required keys
        self._validate_required_keys()

        logger.info(
            "config_loaded",
            config_path=str(self._config_path),
            assets_path=str(self._assets_path),
            system_mode=self._data.get("system", {}).get("mode"),
            num_required_keys_validated=len(REQUIRED_CONFIG_KEYS),
        )

    def _load_dotenv(self) -> None:
        """Load the ``.env`` file if it exists.

        Logs a debug message if the file is missing — this is not an error
        because environment variables may be provided by the OS or a
        container orchestrator.
        """
        if self._dotenv_path.is_file():
            load_dotenv(dotenv_path=self._dotenv_path, override=False)
            logger.debug("dotenv_loaded", path=str(self._dotenv_path))
        else:
            logger.debug(
                "dotenv_not_found",
                path=str(self._dotenv_path),
                hint="Environment variables must be set externally.",
            )

    @staticmethod
    def _load_yaml(path: Path, label: str) -> Dict[str, Any]:
        """Read and parse a YAML file.

        Args:
            path: Filesystem path to the YAML file.
            label: Human-readable label used in log messages.

        Returns:
            Parsed YAML contents as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        if not path.is_file():
            logger.error("yaml_file_missing", path=str(path), label=label)
            raise FileNotFoundError(f"{label} file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                logger.error("yaml_parse_error", path=str(path), error=str(exc))
                raise

        if not isinstance(data, dict):
            logger.error("yaml_not_dict", path=str(path), label=label)
            raise ValueError(f"{label} YAML root must be a mapping, got {type(data).__name__}")

        logger.debug("yaml_loaded", path=str(path), label=label, top_keys=list(data.keys()))
        return data

    # ---- validation ---------------------------------------------------------

    def _validate_required_keys(self) -> None:
        """Ensure every key listed in :data:`REQUIRED_CONFIG_KEYS` exists.

        Raises:
            KeyError: With a message listing all missing keys.
        """
        missing: List[str] = []
        for key_path in REQUIRED_CONFIG_KEYS:
            try:
                _traverse(self._data, key_path)
            except (KeyError, TypeError):
                missing.append(key_path)

        if missing:
            logger.error("config_missing_keys", missing=missing)
            raise KeyError(
                f"Required configuration keys are missing: {', '.join(missing)}"
            )

        logger.debug("config_validation_passed", keys_checked=len(REQUIRED_CONFIG_KEYS))

    # ---- public API ---------------------------------------------------------

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-notation.

        Args:
            key_path: Dot-separated path into the config tree,
                e.g. ``"risk.max_leverage"`` or ``"exchange.name"``.
            default: Value returned when the key is not found.
                Defaults to ``None``.

        Returns:
            The configuration value, or *default* if the path does not exist.

        Examples:
            >>> Config().get("system.mode")
            'paper'
            >>> Config().get("nonexistent.key", default="fallback")
            'fallback'
        """
        try:
            value = _traverse(self._data, key_path)
            return value
        except (KeyError, TypeError):
            logger.debug("config_key_miss", key_path=key_path, default=default)
            return default

    def get_assets(self) -> Dict[str, Any]:
        """Return asset tier information from ``assets.yaml``.

        Returns:
            A dictionary with the following keys:

            - **tier1** (``list[dict]``): List of tier-1 symbol dicts.
            - **tier2** (``list[dict]``): List of tier-2 symbol dicts.
            - **all** (``list[dict]``): Combined tier-1 + tier-2 list.
            - **tier1_symbols** (``list[str]``): Flat list of tier-1 symbol strings.
            - **tier2_symbols** (``list[str]``): Flat list of tier-2 symbol strings.
            - **all_symbols** (``list[str]``): Flat list of all symbol strings.
            - **dynamic_watchlist** (``dict``): Dynamic watchlist configuration.

        Examples:
            >>> assets = Config().get_assets()
            >>> assets["tier1_symbols"]
            ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        """
        tier1_entries: List[Dict[str, Any]] = (
            self._assets_data.get("tier1", {}).get("symbols", [])
        )
        tier2_entries: List[Dict[str, Any]] = (
            self._assets_data.get("tier2", {}).get("symbols", [])
        )

        tier1_symbols = [entry["symbol"] for entry in tier1_entries]
        tier2_symbols = [entry["symbol"] for entry in tier2_entries]

        return {
            "tier1": tier1_entries,
            "tier2": tier2_entries,
            "all": tier1_entries + tier2_entries,
            "tier1_symbols": tier1_symbols,
            "tier2_symbols": tier2_symbols,
            "all_symbols": tier1_symbols + tier2_symbols,
            "dynamic_watchlist": self._assets_data.get("dynamic_watchlist", {}),
        }

    def get_timeframes(self) -> List[str]:
        """Return a deduplicated, flat list of all configured timeframes.

        The timeframes section in ``config.yaml`` is organised by category
        (scalping, intraday, swing, macro, regime_classification).  This
        method flattens them into a single list with duplicates removed,
        preserving insertion order.

        Returns:
            Ordered list of unique timeframe strings.

        Examples:
            >>> Config().get_timeframes()
            ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w']
        """
        tf_section: Dict[str, List[str]] = self._data.get("timeframes", {})
        seen: set[str] = set()
        result: List[str] = []
        for _category, frames in tf_section.items():
            if not isinstance(frames, list):
                continue
            for tf in frames:
                if tf not in seen:
                    seen.add(tf)
                    result.append(tf)
        return result

    @property
    def raw(self) -> Dict[str, Any]:
        """Return the full raw configuration dictionary.

        Returns:
            The complete, resolved configuration from ``config.yaml``.
        """
        return self._data

    @property
    def raw_assets(self) -> Dict[str, Any]:
        """Return the full raw assets dictionary.

        Returns:
            The complete, resolved configuration from ``assets.yaml``.
        """
        return self._assets_data

    def reload(self) -> None:
        """Re-read configuration files from disk.

        Useful during development or after a live config change.
        Thread-safe: acquires the singleton lock while reloading.
        """
        with self._lock:
            logger.info("config_reloading")
            self._load()
            logger.info("config_reloaded")

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton instance so the next ``Config()`` call
        creates a fresh one.

        Intended **only** for testing.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.debug("config_singleton_reset")

    # ---- dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<Config mode={self._data.get('system', {}).get('mode', '?')!r} "
            f"config={self._config_path} assets={self._assets_path}>"
        )
