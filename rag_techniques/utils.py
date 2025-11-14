import json
from collections import OrderedDict
from typing import Any, List, Dict, Tuple

def _is_primitive(x):
    return x is None or isinstance(x, (str, int, float, bool))

def _all_dicts_with_same_keys(lst: List[Any]) -> Tuple[bool, List[str]]:
    if not lst:
        return False, []
    if not all(isinstance(el, dict) for el in lst):
        return False, []
    # Use insertion order of keys from first element
    keys = list(lst[0].keys())
    for el in lst:
        if list(el.keys()) != keys:
            return False, []
    return True, keys

def _needs_quote(s: str) -> bool:
    # Minimal quoting rules (inspired by TOON description):
    # quote when the string contains the delimiter ',' or newline,
    # when it contains a double quote, or when it has leading/trailing whitespace.
    if not isinstance(s, str):
        return False
    if s == "":
        return True
    if s[0].isspace() or s[-1].isspace():
        return True
    if any(ch in s for ch in [',', '\n', '\r', '"']):
        return True
    return False

def _quote_str(s: str) -> str:
    # escape double quotes and wrap in double quotes
    s2 = s.replace('"', '\\"')
    return f'"{s2}"'

def _format_value(v: Any) -> str:
    # primitives: numbers, booleans, null stay unquoted (JSON style)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        # preserve numeric representation
        return str(v)
    if isinstance(v, str):
        if _needs_quote(v):
            return _quote_str(v)
        else:
            return v
    # for objects/arrays where a primitive expected, serialize to compact JSON as fallback
    return _quote_str(json.dumps(v, separators=(',', ':')))

def json_to_toon(data: Any, name: str = None, indent: int = 0) -> str:
    """
    Convert a Python object (from json.loads) into TOON format.
    - name: optional name for the current node (used for top-level object keys)
    - indent: internal use, number of spaces to indent nested blocks
    Returns a TOON-format string.
    """
    pad = " " * indent
    out_lines: List[str] = []

    # Helper to write a named primitive or inline list
    def write_named_line(key: str, line: str):
        if indent:
            out_lines.append(f"{pad}{key}{line}")
        else:
            out_lines.append(f"{key}{line}")

    # If top-level is a dict: iterate keys in insertion order
    if isinstance(data, dict):
        for k, v in data.items():
            # Objects -> key: newline then nested block
            if isinstance(v, dict):
                out_lines.append(f"{pad}{k}:")
                out_lines.append(json_to_toon(v, name=None, indent=indent + 2))
            # Arrays -> try tabular optimization for uniform dict arrays or primitives
            elif isinstance(v, list):
                lst = v
                # empty list: emit length 0
                if len(lst) == 0:
                    out_lines.append(f"{pad}{k}[0]:")
                    continue

                all_dicts, cols = _all_dicts_with_same_keys(lst)
                if all_dicts:
                    # tabular array with header
                    header = "{" + ",".join(cols) + "}"
                    out_lines.append(f"{pad}{k}[{len(lst)}]{header}:")
                    # each row
                    for item in lst:
                        row_vals = [_format_value(item[c]) for c in cols]
                        out_lines.append(f"{pad}{','.join(row_vals)}")
                elif all(_is_primitive(el) for el in lst):
                    # array of primitives -> inline comma-separated
                    vals = [_format_value(el) for el in lst]
                    out_lines.append(f"{pad}{k}[{len(lst)}]: {','.join(vals)}")
                else:
                    # mixed or non-uniform -> list block with each element printed as nested block
                    out_lines.append(f"{pad}{k}[{len(lst)}]:")
                    for el in lst:
                        # For primitives, write a single-line value
                        if _is_primitive(el):
                            out_lines.append(f'{" " * (indent + 2)}{_format_value(el)}')
                        else:
                            # complex element: recurse with increased indent
                            out_lines.append(json_to_toon(el, name=None, indent=indent + 2))
            else:
                # primitive value
                out_lines.append(f"{pad}{k}: {_format_value(v)}")
        return "\n".join(out_lines)

    # If top-level is a list (no key provided)
    if isinstance(data, list):
        lst = data
        if len(lst) == 0:
            name_part = (f"{name}" if name else "")
            return f"{pad}{name_part}[0]:"
        all_dicts, cols = _all_dicts_with_same_keys(lst)
        if all_dicts:
            header = "{" + ",".join(cols) + "}"
            name_part = (f"{name}" if name else "")
            # If name provided, include it; otherwise just do header with length
            if name_part:
                out_lines.append(f"{pad}{name_part}[{len(lst)}]{header}:")
            else:
                out_lines.append(f"{pad}[{len(lst)}]{header}:")
            for item in lst:
                row_vals = [_format_value(item[c]) for c in cols]
                out_lines.append(f"{pad}{','.join(row_vals)}")
            return "\n".join(out_lines)
        elif all(_is_primitive(el) for el in lst):
            vals = [_format_value(el) for el in lst]
            name_part = (f"{name}" if name else "")
            if name_part:
                out_lines.append(f"{pad}{name_part}[{len(lst)}]: {','.join(vals)}")
            else:
                out_lines.append(f"{pad}[{len(lst)}]: {','.join(vals)}")
            return "\n".join(out_lines)
        else:
            # mixed elements
            name_part = (f"{name}" if name else "")
            out_lines.append(f"{pad}{name_part}[{len(lst)}]:")
            for el in lst:
                if _is_primitive(el):
                    out_lines.append(f'{" " * (indent + 2)}{_format_value(el)}')
                else:
                    out_lines.append(json_to_toon(el, name=None, indent=indent + 2))
            return "\n".join(out_lines)

    # Primitive scalar at top-level
    if _is_primitive(data):
        if name:
            return f"{pad}{name}: {_format_value(data)}"
        else:
            return f"{pad}{_format_value(data)}"

    # Fallback: dump JSON in quotes
    txt = json.dumps(data, separators=(',', ':'))
    if name:
        return f'{pad}{name}: "{txt}"'
    else:
        return f'{pad}"{txt}"'

# Convenience wrapper that accepts a JSON string or a Python object
def convert_json_to_toon(obj_or_json: Any) -> str:
    """
    If input is a string, it will be parsed as JSON. Otherwise it must be a Python object
    (dict/list/primitive) like the result of json.loads(...).
    Returns TOON-format string.
    """
    if isinstance(obj_or_json, str):
        parsed = json.loads(obj_or_json)
    else:
        parsed = obj_or_json
    # If top-level is a dict, we convert directly
    return json_to_toon(parsed)

# -------------------------
# Example usage:
if __name__ == "__main__":
    data = {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin", "salary": 75000},
            {"id": 2, "name": "Bob", "role": "user", "salary": 65000},
            {"id": 3, "name": "Charlie", "role": "user", "salary": 70000}
        ],
        "tags": ["admin", "ops", "dev"],
        "meta": {"count": 3, "source": "hr db"}
    }

    print(convert_json_to_toon(data))
