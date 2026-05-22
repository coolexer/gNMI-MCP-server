from __future__ import annotations

from nokia_gnmi_mcp import server
from nokia_gnmi_mcp.models import DeviceSession


def test_fastmcp_server_exports_expected_tool_functions():
    assert server.mcp.name == "nokia-gnmi-mcp"
    assert callable(server.sros_connect)
    assert callable(server.yang_search)


def test_server_list_sessions_uses_shared_manager():
    server.session_manager.sessions.clear()
    server.session_manager.connections.clear()
    server.session_manager.sessions["pe1"] = DeviceSession(
        name="pe1",
        host="192.0.2.1",
        username="admin",
        password="admin",
    )

    assert "pe1" in server.sros_list_sessions()
    server.session_manager.sessions.clear()
