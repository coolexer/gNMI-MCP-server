from __future__ import annotations

from nokia_gnmi_mcp.gnmi_ops import (
    capabilities_summary,
    json_pretty,
    run_get,
    run_set_delete,
    run_set_replace,
    run_set_update,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return {"ok": True, "kwargs": kwargs}

    def set(self, **kwargs):
        self.calls.append(("set", kwargs))
        return {"set": kwargs}


def test_json_pretty_formats_objects_and_json_strings():
    assert json_pretty({"a": 1}) == '{\n  "a": 1\n}'
    assert json_pretty('{"b": 2}') == '{\n  "b": 2\n}'


def test_gnmi_operation_wrappers_call_pygnmi_shape():
    client = FakeClient()

    run_get(client, ["/configure"], datatype="config")
    run_set_update(client, "/configure/system", {"name": "x"})
    run_set_replace(client, "/configure/system", {"name": "x"})
    run_set_delete(client, ["/configure/system"])

    assert client.calls[0] == (
        "get",
        {"path": ["/configure"], "datatype": "config", "encoding": "json_ietf"},
    )
    assert client.calls[1] == (
        "set",
        {"update": [("/configure/system", {"name": "x"})], "encoding": "json_ietf"},
    )
    assert client.calls[2][1]["replace"] == [("/configure/system", {"name": "x"})]
    assert client.calls[3][1]["delete"] == ["/configure/system"]


def test_capabilities_summary_counts_model_families():
    summary = capabilities_summary(
        {
            "supported_encodings": ["json_ietf"],
            "gnmi_version": "0.8.0",
            "supported_models": [
                {"name": "nokia-conf", "version": "1"},
                {"name": "ietf-interfaces", "version": "2"},
            ],
        }
    )

    assert summary["model_count"] == 2
    assert summary["nokia_models"] == 1
    assert summary["ietf_models"] == 1
