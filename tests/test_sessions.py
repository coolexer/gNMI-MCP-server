from __future__ import annotations

from nokia_gnmi_mcp.models import DeviceSession
from nokia_gnmi_mcp.sessions import SessionManager


class FakeGNMIClient:
    attempts = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeGNMIClient.attempts.append(kwargs)

    def connect(self):
        if not self.kwargs.get("insecure"):
            raise RuntimeError("SSL handshake failed")

    def capabilities(self):
        return {
            "gnmi_version": "0.8.0",
            "supported_encodings": ["json_ietf"],
            "supported_models": [{"name": "nokia-conf", "version": "1"}],
        }

    def close(self):
        pass


def test_session_manager_falls_back_to_insecure_on_tls_failure():
    FakeGNMIClient.attempts = []
    manager = SessionManager(client_factory=FakeGNMIClient)

    result = manager.connect(
        DeviceSession(
            name="pe1",
            host="192.0.2.1",
            username="admin",
            password="admin",
            insecure=False,
            skip_verify=True,
        )
    )

    assert "connected to 'pe1'" in result
    assert FakeGNMIClient.attempts[0]["insecure"] is False
    assert FakeGNMIClient.attempts[1]["insecure"] is True
    assert manager.sessions["pe1"].insecure is True


def test_session_manager_lists_and_disconnects_sessions():
    manager = SessionManager(client_factory=FakeGNMIClient)
    manager.sessions["pe1"] = DeviceSession(
        name="pe1",
        host="192.0.2.1",
        username="admin",
        password="admin",
    )

    assert "pe1" in manager.list_sessions()
    assert manager.disconnect("pe1") == "Session 'pe1' closed."
    assert manager.list_sessions() == "No active sessions."
