from typing import Dict, List


from typing import Any, Optional

def parse_param(param_dict: Dict, key: str, enforce_values: Optional[List] = None, default: Any = None, optional: bool = False):
    if not optional:
        if key not in param_dict:
            raise ValueError(f'No value exists for {key} but is not optional')
    param = param_dict.get(key, default)
    if enforce_values is not None and param not in enforce_values:
        raise ValueError(f'Invalid value {param} for param {key}')

    return param