from typing import Union

from . import compiled_autograd, eval_frame, guards  # noqa: F401

def strip_function_call(name: str) -> str: ...
def is_valid_var_name(name: str) -> Union[bool, int]: ...
