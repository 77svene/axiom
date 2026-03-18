from typing import TYPE_CHECKING, Any
from axiom.engines.toolbelt import ProxyRotator

if TYPE_CHECKING:
    from axiom.fetchers.requests import Fetcher, AsyncFetcher, FetcherSession
    from axiom.fetchers.chrome import DynamicFetcher, DynamicSession, AsyncDynamicSession
    from axiom.fetchers.stealth_chrome import StealthyFetcher, StealthySession, AsyncStealthySession


# Lazy import mapping
_LAZY_IMPORTS = {
    "Fetcher": ("axiom.fetchers.requests", "Fetcher"),
    "AsyncFetcher": ("axiom.fetchers.requests", "AsyncFetcher"),
    "FetcherSession": ("axiom.fetchers.requests", "FetcherSession"),
    "DynamicFetcher": ("axiom.fetchers.chrome", "DynamicFetcher"),
    "DynamicSession": ("axiom.fetchers.chrome", "DynamicSession"),
    "AsyncDynamicSession": ("axiom.fetchers.chrome", "AsyncDynamicSession"),
    "StealthyFetcher": ("axiom.fetchers.stealth_chrome", "StealthyFetcher"),
    "StealthySession": ("axiom.fetchers.stealth_chrome", "StealthySession"),
    "AsyncStealthySession": ("axiom.fetchers.stealth_chrome", "AsyncStealthySession"),
}

__all__ = [
    "Fetcher",
    "AsyncFetcher",
    "ProxyRotator",
    "FetcherSession",
    "DynamicFetcher",
    "DynamicSession",
    "AsyncDynamicSession",
    "StealthyFetcher",
    "StealthySession",
    "AsyncStealthySession",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Support for dir() and autocomplete."""
    return sorted(list(_LAZY_IMPORTS.keys()))
