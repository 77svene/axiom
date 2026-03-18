__author__ = "Karim Shoair (karim.shoair@pm.me)"
__version__ = "0.4.2"
__copyright__ = "Copyright (c) 2024 Karim Shoair"

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from axiom.parser import Selector, Selectors
    from axiom.core.custom_types import AttributesHandler, TextHandler
    from axiom.fetchers import Fetcher, AsyncFetcher, StealthyFetcher, DynamicFetcher


# Lazy import mapping
_LAZY_IMPORTS = {
    "Fetcher": ("axiom.fetchers", "Fetcher"),
    "Selector": ("axiom.parser", "Selector"),
    "Selectors": ("axiom.parser", "Selectors"),
    "AttributesHandler": ("axiom.core.custom_types", "AttributesHandler"),
    "TextHandler": ("axiom.core.custom_types", "TextHandler"),
    "AsyncFetcher": ("axiom.fetchers", "AsyncFetcher"),
    "StealthyFetcher": ("axiom.fetchers", "StealthyFetcher"),
    "DynamicFetcher": ("axiom.fetchers", "DynamicFetcher"),
}
__all__ = ["Selector", "Fetcher", "AsyncFetcher", "StealthyFetcher", "DynamicFetcher"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Support for dir() and autocomplete."""
    return sorted(__all__ + ["fetchers", "parser", "cli", "core", "__author__", "__version__", "__copyright__"])
