"""Small wrappers around pygnmi operations."""

from __future__ import annotations

import json
from typing import Any


def json_pretty(obj: Any) -> str:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return obj
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def run_get(client: Any, paths: list[str], datatype: str = "config") -> str:
    return json_pretty(client.get(path=paths, datatype=datatype, encoding="json_ietf"))


def run_set_update(client: Any, path: str, value: dict[str, Any]) -> str:
    return json_pretty(client.set(update=[(path, value)], encoding="json_ietf"))


def run_set_replace(client: Any, path: str, value: dict[str, Any]) -> str:
    return json_pretty(client.set(replace=[(path, value)], encoding="json_ietf"))


def run_set_delete(client: Any, paths: list[str]) -> str:
    return json_pretty(client.set(delete=paths, encoding="json_ietf"))


def capabilities_summary(result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if "supported_encodings" in result:
        output["supported_encodings"] = result["supported_encodings"]
    if "supported_models" in result:
        models = result["supported_models"]
        nokia_models = [m for m in models if "nokia" in m.get("name", "").lower()]
        ietf_models = [m for m in models if "ietf" in m.get("name", "").lower()]
        output["model_count"] = len(models)
        output["nokia_models"] = len(nokia_models)
        output["ietf_models"] = len(ietf_models)
        output["nokia_sample"] = [
            f"{m.get('name', '?')} ({m.get('version', '?')})"
            for m in nokia_models[:10]
        ]
    if "gnmi_version" in result:
        output["gnmi_version"] = result["gnmi_version"]
    return output


def run_capabilities(client: Any) -> str:
    return json_pretty(capabilities_summary(client.capabilities()))


def run_cli_command(client: Any, command: str) -> str:
    try:
        from pygnmi.spec.v080.gnmi_ext_pb2 import Extension, RegisteredExtension

        ext = Extension(registered_ext=RegisteredExtension(id=1001, msg=command.encode()))
        return json_pretty(client.get(path=["/"], encoding="ascii", extension=[ext]))
    except Exception as exc:
        return f"CLI via gNMI extension failed: {exc}. Use gNMI native paths instead."
