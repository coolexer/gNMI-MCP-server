from __future__ import annotations

import asyncio

from nokia_gnmi_mcp import server
from nokia_gnmi_mcp.models import DeviceSession


def test_fastmcp_server_exports_expected_tool_functions():
    assert server.mcp.name == "nokia-gnmi-mcp"
    assert callable(server.gnmi_connect)
    assert callable(server.sros_connect)
    assert callable(server.yang_search)


def test_fastmcp_server_registers_generic_tools_and_legacy_aliases():
    tools = asyncio.run(server.mcp.list_tools())
    tool_names = {tool.name for tool in tools}

    assert {
        "gnmi_connect",
        "gnmi_disconnect",
        "gnmi_get_config",
        "gnmi_get_state",
        "gnmi_set_update",
        "gnmi_set_replace",
        "gnmi_set_delete",
        "gnmi_cli_command",
        "gnmi_capabilities",
        "gnmi_list_sessions",
        "yang_search",
    }.issubset(tool_names)
    assert {
        "sros_connect",
        "sros_get_config",
        "sros_set_update",
        "sros_list_sessions",
    }.issubset(tool_names)


def test_server_list_sessions_uses_shared_manager():
    server.session_manager.sessions.clear()
    server.session_manager.connections.clear()
    server.session_manager.sessions["pe1"] = DeviceSession(
        name="pe1",
        host="192.0.2.1",
        username="admin",
        password="admin",
    )

    assert "pe1" in server.gnmi_list_sessions()
    assert server.sros_list_sessions() == server.gnmi_list_sessions()
    server.session_manager.sessions.clear()
