# mypy: allow-untyped-defs
import contextlib
from typing import Any, Callable, NoReturn, TYPE_CHECKING, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

"""
This was semi-automatically generated by running

    stubgen torch.utils._config_module.py

And then manually extracting the methods of ConfigModule and converting them into top-level functions.

This file should be imported into any file that uses install_config_module like so:

    if TYPE_CHECKING:
        from torch.utils._config_typing import *  # noqa: F401, F403

    from torch.utils._config_module import install_config_module

    # adds patch, save_config, etc
    install_config_module(sys.modules[__name__])

Note that the import should happen before the call to install_config_module(), otherwise runtime errors may occur.
"""

assert TYPE_CHECKING, "Do not use at runtime"

def save_config() -> bytes: ...
def save_config_portable() -> dict[str, Any]: ...
def codegen_config() -> str: ...
def get_hash() -> bytes: ...
def to_dict() -> dict[str, Any]: ...
def shallow_copy_dict() -> dict[str, Any]: ...
def load_config(config: bytes | dict[str, Any]) -> None: ...
def get_config_copy() -> dict[str, Any]: ...
def patch(
    arg1: str | dict[str, Any] | None = None, arg2: Any = None, **kwargs
) -> ContextDecorator: ...

class ContextDecorator(contextlib.ContextDecorator):
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> NoReturn: ...
    def __call__(self, func: _F) -> _F: ...
