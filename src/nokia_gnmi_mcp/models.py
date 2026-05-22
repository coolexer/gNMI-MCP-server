"""Shared data models for Nokia gNMI MCP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceSession:
    name: str
    host: str
    port: int = 57400
    username: str = "admin"
    password: str = "admin"
    skip_verify: bool = True
    insecure: bool = False
    timeout: int = 10
