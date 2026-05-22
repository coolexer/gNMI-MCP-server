"""Device session lifecycle for Nokia gNMI MCP."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .models import DeviceSession

logger = logging.getLogger("nokia-gnmi-mcp.sessions")


def default_client_factory(**kwargs: Any) -> Any:
    from pygnmi.client import gNMIclient

    return gNMIclient(**kwargs)


class SessionManager:
    def __init__(self, client_factory: Callable[..., Any] = default_client_factory):
        self.client_factory = client_factory
        self.sessions: dict[str, DeviceSession] = {}
        self.connections: dict[str, Any] = {}

    def connect(self, session: DeviceSession) -> str:
        if session.name in self.sessions:
            self.disconnect(session.name)
        self.sessions[session.name] = session
        try:
            client = self._connect(session.name)
            caps = client.capabilities()
        except Exception as exc:
            err = str(exc).lower()
            if not session.insecure and any(token in err for token in ("ssl", "certificate", "tls")):
                logger.info("TLS failed for '%s', retrying with insecure=True", session.name)
                self.disconnect(session.name)
                session.insecure = True
                session.skip_verify = False
                self.sessions[session.name] = session
                client = self._connect(session.name)
                caps = client.capabilities()
            else:
                self.disconnect(session.name)
                raise

        model_count = len(caps.get("supported_models", []))
        encodings = caps.get("supported_encodings", [])
        gnmi_version = caps.get("gnmi_version", "unknown")
        tls_mode = "insecure (plain gRPC)" if session.insecure else (
            "skip_verify" if session.skip_verify else "TLS"
        )
        return (
            f"gNMI connected to '{session.name}' ({session.host}:{session.port}) [{tls_mode}]\n"
            f"  gNMI version: {gnmi_version}\n"
            f"  Supported encodings: {', '.join(encodings)}\n"
            f"  YANG models: {model_count}"
        )

    def get_client(self, name: str) -> Any:
        if name not in self.sessions:
            raise ValueError(f"Device '{name}' not registered. Use 'sros_connect' tool first.")
        if name not in self.connections:
            return self._connect(name)
        return self.connections[name]

    def disconnect(self, name: str) -> str:
        if name in self.connections:
            try:
                self.connections[name].close()
            except Exception:
                pass
            del self.connections[name]
        if name in self.sessions:
            del self.sessions[name]
            return f"Session '{name}' closed."
        return f"Session '{name}' not found."

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No active sessions."
        lines = []
        for name, session in self.sessions.items():
            connected = name in self.connections
            status = "connected" if connected else "registered"
            tls = "insecure" if session.insecure else (
                "skip_verify" if session.skip_verify else "TLS"
            )
            lines.append(
                f"  {name}: {session.host}:{session.port} "
                f"user={session.username} [{status}] ({tls})"
            )
        return "Sessions:\n" + "\n".join(lines)

    def _connect(self, name: str) -> Any:
        session = self.sessions[name]
        logger.info("gNMI connecting to %s:%s as %s", session.host, session.port, session.username)
        client = self.client_factory(
            target=(session.host, session.port),
            username=session.username,
            password=session.password,
            skip_verify=session.skip_verify,
            insecure=session.insecure,
            timeout=session.timeout,
        )
        client.connect()
        self.connections[name] = client
        return client
