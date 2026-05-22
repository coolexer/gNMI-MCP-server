"""FastMCP server for Nokia SR OS gNMI operations."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .gnmi_ops import (
    run_capabilities,
    run_cli_command,
    run_get,
    run_set_delete,
    run_set_replace,
    run_set_update,
)
from .logging_setup import configure_logging
from .models import DeviceSession
from .sessions import SessionManager
from .yang_cache import YangSearch

configure_logging()

session_manager = SessionManager()
yang_searcher = YangSearch()

mcp = FastMCP(
    name="nokia-gnmi-mcp",
    instructions=(
        "Nokia SR OS gNMI operations. For best path selection, use the separate "
        "nokia-yang-mcp server first, then pass validated paths to these gNMI tools."
    ),
)


def sros_connect(
    name: str,
    host: str,
    username: str,
    password: str,
    port: int = 57400,
    skip_verify: bool = True,
    insecure: bool = False,
    timeout: int = 10,
) -> str:
    """Connect to a Nokia SR OS device via gNMI.

    Args:
        name: Local session name such as `pe1`.
        host: Device hostname or IP address.
        username: gNMI username.
        password: gNMI password.
        port: gNMI port, usually 57400.
        skip_verify: Skip TLS certificate verification for lab/self-signed certs.
        insecure: Use plain gRPC with no TLS, common for srsim/containerlab.
        timeout: Connection timeout in seconds.
    """
    return session_manager.connect(
        DeviceSession(
            name=name,
            host=host,
            port=port,
            username=username,
            password=password,
            skip_verify=skip_verify,
            insecure=insecure,
            timeout=timeout,
        )
    )


def sros_disconnect(name: str) -> str:
    """Close a gNMI session."""
    return session_manager.disconnect(name)


def sros_get_config(name: str, paths: list[str]) -> str:
    """Get configuration via gNMI Get using datatype `config`.

    Use `nokia-yang-mcp` to find and validate paths before calling this tool.
    """
    return run_get(session_manager.get_client(name), paths, datatype="config")


def sros_get_state(name: str, paths: list[str]) -> str:
    """Get operational state via gNMI Get using datatype `state`.

    Use `nokia-yang-mcp` to find and validate paths before calling this tool.
    """
    return run_get(session_manager.get_client(name), paths, datatype="state")


def sros_set_update(name: str, path: str, value: dict[str, Any]) -> str:
    """Merge configuration with gNMI Set update.

    Validate the target path with `nokia-yang-mcp.yang_check_path_support` before
    writing to a router.
    """
    return run_set_update(session_manager.get_client(name), path, value)


def sros_set_replace(name: str, path: str, value: dict[str, Any]) -> str:
    """Replace the complete configuration subtree at a path with gNMI Set replace."""
    return run_set_replace(session_manager.get_client(name), path, value)


def sros_set_delete(name: str, paths: list[str]) -> str:
    """Delete configuration paths with gNMI Set delete."""
    return run_set_delete(session_manager.get_client(name), paths)


def sros_cli_command(name: str, command: str) -> str:
    """Run an MD-CLI show command through the Nokia gNMI CLI extension.

    Prefer native gNMI paths for automation; use this mainly for exploratory show
    commands when the device supports the extension.
    """
    return run_cli_command(session_manager.get_client(name), command)


def sros_capabilities(name: str) -> str:
    """Return supported gNMI encodings, models, and version for a session."""
    return run_capabilities(session_manager.get_client(name))


def sros_list_sessions() -> str:
    """List registered and connected gNMI sessions."""
    return session_manager.list_sessions()


def yang_search(keyword: str, tree: str = "configure", max_results: int = 50) -> str:
    """Search local fallback YANG path cache.

    Prefer the separate `nokia-yang-mcp` server for authoritative bundled DB
    search and platform support. This fallback only searches local `yang/`
    files or `yang/cache/*.txt` when present in this repository.
    """
    return yang_searcher.search(keyword, tree=tree, max_results=max_results)


mcp.tool()(sros_connect)
mcp.tool()(sros_disconnect)
mcp.tool()(sros_get_config)
mcp.tool()(sros_get_state)
mcp.tool()(sros_set_update)
mcp.tool()(sros_set_replace)
mcp.tool()(sros_set_delete)
mcp.tool()(sros_cli_command)
mcp.tool()(sros_capabilities)
mcp.tool()(sros_list_sessions)
mcp.tool()(yang_search)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
