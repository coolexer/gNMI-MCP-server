"""
Nokia SR OS gNMI MCP Server
============================
MCP server for managing Nokia SR OS devices via gNMI (gRPC).

Features:
  - gNMI Get for configuration and operational state
  - gNMI Set (update/replace/delete) for config changes
  - MD-CLI show commands via Nokia gNMI CLI extension
  - gNMI Capabilities discovery
  - Runtime credential management
  - Multiple simultaneous device sessions
  - Claude Desktop compatible (stdio transport, Windows)

Usage with Claude Desktop:
  Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "nokia-gnmi": {
        "command": "uv",
        "args": ["--directory", "C:\\\\path\\\\to\\\\nokia-gnmi-mcp", "run", "nokia-gnmi-mcp"]
      }
    }
  }
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from pygnmi.client import gNMIclient

import sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("nokia-gnmi-mcp")

# Suppress pygnmi INFO/DEBUG logs to avoid polluting MCP stdout stream
logging.getLogger("pygnmi").setLevel(logging.WARNING)
logging.getLogger("pygnmi.client").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════
# Device session management
# ═══════════════════════════════════════════════

@dataclass
class DeviceSession:
    host: str
    port: int = 57400
    username: str = "admin"
    password: str = "admin"
    skip_verify: bool = True
    insecure: bool = False
    timeout: int = 10


_sessions: dict[str, DeviceSession] = {}
_connections: dict[str, gNMIclient] = {}


def _connect(name: str) -> gNMIclient:
    """Get existing gNMI connection or create new one."""
    if name not in _sessions:
        raise ValueError(
            f"Device '{name}' not registered. Use 'sros_connect' tool first."
        )

    if name in _connections:
        return _connections[name]

    s = _sessions[name]
    logger.info(f"gNMI connecting to {s.host}:{s.port} as {s.username}")
    gc = gNMIclient(
        target=(s.host, s.port),
        username=s.username,
        password=s.password,
        skip_verify=s.skip_verify,
        insecure=s.insecure,
        timeout=s.timeout,
    )
    gc.connect()
    _connections[name] = gc
    logger.info(f"gNMI connected to '{name}'")
    return gc


def _close(name: str) -> str:
    """Close gNMI connection and remove session."""
    if name in _connections:
        try:
            _connections[name].close()
        except Exception:
            pass
        del _connections[name]
    if name in _sessions:
        del _sessions[name]
        return f"Session '{name}' closed."
    return f"Session '{name}' not found."


def _json_pretty(obj) -> str:
    """Pretty-print any object as JSON."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return obj
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════
# gNMI operations
# ═══════════════════════════════════════════════

def _do_get(gc: gNMIclient, path: list[str], datatype: str = "config") -> str:
    """gNMI Get — retrieve config or state."""
    result = gc.get(path=path, datatype=datatype, encoding="json_ietf")
    return _json_pretty(result)


def _do_set_update(gc: gNMIclient, path: str, value: dict) -> str:
    """gNMI Set with update (merge)."""
    update = [(path, value)]
    result = gc.set(update=update, encoding="json_ietf")
    return _json_pretty(result)


def _do_set_replace(gc: gNMIclient, path: str, value: dict) -> str:
    """gNMI Set with replace."""
    replace = [(path, value)]
    result = gc.set(replace=replace, encoding="json_ietf")
    return _json_pretty(result)


def _do_set_delete(gc: gNMIclient, paths: list[str]) -> str:
    """gNMI Set with delete."""
    result = gc.set(delete=paths, encoding="json_ietf")
    return _json_pretty(result)


def _do_cli_command(gc: gNMIclient, command: str) -> str:
    """Execute MD-CLI command via Nokia gNMI CLI extension.

    Nokia supports CLI commands through the
    'urn:nokia.com:srlinux:gnmi:cli' or standard ASCII CLI extension.
    For SROS, we use the cli_command approach via pygnmi or raw RPC.
    """
    # pygnmi supports Nokia CLI via get with origin='cli'
    # Nokia SROS gNMI also supports the '/cli' path
    try:
        result = gc.get(
            path=["/"],
            datatype="config",
            encoding="ascii",
        )
        # Fallback: try the Nokia-specific approach
        return _json_pretty(result)
    except Exception:
        pass

    # Alternative: use Nokia's md-cli through gNMI extension
    # This uses the gnmi_ext for CLI commands
    try:
        from pygnmi.spec.v080.gnmi_ext_pb2 import Extension, RegisteredExtension
        ext = Extension(
            registered_ext=RegisteredExtension(
                id=1001,  # Nokia CLI extension ID
                msg=command.encode(),
            )
        )
        result = gc.get(path=["/"], encoding="ascii", extension=[ext])
        return _json_pretty(result)
    except Exception as e:
        return f"CLI via gNMI extension failed: {e}. Use gNMI native paths instead."


def _do_capabilities(gc: gNMIclient) -> str:
    """gNMI Capabilities — list supported models and encodings."""
    result = gc.capabilities()
    # Format capabilities nicely
    output = {}
    if "supported_encodings" in result:
        output["supported_encodings"] = result["supported_encodings"]
    if "supported_models" in result:
        models = result["supported_models"]
        output["model_count"] = len(models)
        # Group Nokia models
        nokia_models = [m for m in models if "nokia" in m.get("name", "").lower()]
        ietf_models = [m for m in models if "ietf" in m.get("name", "").lower()]
        other_models = [
            m for m in models
            if "nokia" not in m.get("name", "").lower()
            and "ietf" not in m.get("name", "").lower()
        ]
        output["nokia_models"] = len(nokia_models)
        output["ietf_models"] = len(ietf_models)
        output["other_models"] = len(other_models)
        # Show first 10 Nokia models as sample
        output["nokia_sample"] = [
            f"{m.get('name', '?')} ({m.get('version', '?')})"
            for m in nokia_models[:10]
        ]
    if "gnmi_version" in result:
        output["gnmi_version"] = result["gnmi_version"]
    return _json_pretty(output)


# ═══════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════

TOOLS = [
    Tool(
        name="sros_connect",
        description=(
            "Connect to a Nokia SR OS device via gNMI (gRPC). "
            "Registers a named session with credentials for subsequent operations. "
            "Default port is 57400. Uses skip_verify=true for lab environments."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Session name (e.g. 'pe1', 'core-rtr')",
                },
                "host": {
                    "type": "string",
                    "description": "IP address or hostname of the SR OS device",
                },
                "port": {
                    "type": "integer",
                    "description": "gNMI port (default: 57400)",
                    "default": 57400,
                },
                "username": {
                    "type": "string",
                    "description": "gRPC/gNMI username",
                },
                "password": {
                    "type": "string",
                    "description": "gRPC/gNMI password",
                },
                "skip_verify": {
                    "type": "boolean",
                    "description": "Skip TLS certificate verification (default: true)",
                    "default": True,
                },
                "insecure": {
                    "type": "boolean",
                    "description": "Use insecure (non-TLS) connection (default: false)",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Connection timeout in seconds (default: 10)",
                    "default": 10,
                },
            },
            "required": ["name", "host", "username", "password"],
        },
    ),
    Tool(
        name="sros_disconnect",
        description="Close a gNMI session to a Nokia SR OS device.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Session name to disconnect"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="sros_get_config",
        description=(
            "Get configuration from Nokia SR OS via gNMI Get (datatype=config). "
            "Provide one or more YANG paths. Uses json_ietf encoding.\n\n"
            "Nokia SROS YANG path examples:\n"
            "  /configure/router[router-name=Base]/interface\n"
            "  /configure/port[port-id=1/1/c2/1]\n"
            "  /configure/service/vprn[service-name=CUST-1]\n"
            "  /configure/router[router-name=Base]/bgp\n"
            "  /configure/card[slot-number=1]\n"
            "  /configure (entire config)"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of YANG paths to retrieve. "
                        "Example: ['/configure/router[router-name=Base]/interface']"
                    ),
                },
            },
            "required": ["name", "paths"],
        },
    ),
    Tool(
        name="sros_get_state",
        description=(
            "Get operational state from Nokia SR OS via gNMI Get (datatype=state). "
            "Provide one or more YANG paths. Uses json_ietf encoding.\n\n"
            "Nokia SROS state path examples:\n"
            "  /state/router[router-name=Base]/interface[interface-name=to-pe2]\n"
            "  /state/port[port-id=1/1/c2/1]\n"
            "  /state/router[router-name=Base]/bgp/neighbor\n"
            "  /state/card[slot-number=1]\n"
            "  /state/system/information"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of YANG state paths to retrieve. "
                        "Example: ['/state/system/information']"
                    ),
                },
            },
            "required": ["name", "paths"],
        },
    ),
    Tool(
        name="sros_set_update",
        description=(
            "Update (merge) configuration on Nokia SR OS via gNMI Set. "
            "Provide a YANG path and JSON value to merge.\n\n"
            "Example — create interface:\n"
            "  path: /configure/router[router-name=Base]/interface[interface-name=lo0]\n"
            "  value: {\"ipv4\": {\"primary\": {\"address\": \"10.0.0.1\", \"prefix-length\": 32}}}\n\n"
            "Example — enable port:\n"
            "  path: /configure/port[port-id=1/1/c2/1]\n"
            '  value: {"admin-state": "enable"}'
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
                "path": {
                    "type": "string",
                    "description": "YANG path to update",
                },
                "value": {
                    "type": "object",
                    "description": "JSON object with config values to merge",
                },
            },
            "required": ["name", "path", "value"],
        },
    ),
    Tool(
        name="sros_set_replace",
        description=(
            "Replace configuration on Nokia SR OS via gNMI Set. "
            "Replaces the entire subtree at the given path with the provided value. "
            "WARNING: This removes any existing config under the path that is not in the new value."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
                "path": {
                    "type": "string",
                    "description": "YANG path to replace",
                },
                "value": {
                    "type": "object",
                    "description": "JSON object — complete replacement config",
                },
            },
            "required": ["name", "path", "value"],
        },
    ),
    Tool(
        name="sros_set_delete",
        description=(
            "Delete configuration on Nokia SR OS via gNMI Set. "
            "Removes the config subtree at the given path(s).\n\n"
            "Example:\n"
            "  paths: ['/configure/router[router-name=Base]/interface[interface-name=test]']"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of YANG paths to delete",
                },
            },
            "required": ["name", "paths"],
        },
    ),
    Tool(
        name="sros_cli_command",
        description=(
            "Execute an MD-CLI show command on Nokia SR OS via gNMI CLI extension. "
            "Note: CLI via gNMI may not be supported on all platforms/versions. "
            "Prefer gNMI native paths (sros_get_config/sros_get_state) when possible.\n\n"
            "Examples: 'show router interface', 'show port', 'show card state'"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
                "command": {
                    "type": "string",
                    "description": "MD-CLI command to execute",
                },
            },
            "required": ["name", "command"],
        },
    ),
    Tool(
        name="sros_capabilities",
        description=(
            "Get gNMI capabilities from Nokia SR OS device. "
            "Returns supported encodings, YANG models, and gNMI version."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Device session name"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="sros_list_sessions",
        description="List all active Nokia SR OS gNMI sessions.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ═══════════════════════════════════════════════
# Tool dispatch
# ═══════════════════════════════════════════════

def handle_tool(tool_name: str, args: dict) -> str:
    """Route tool call to the appropriate handler."""

    if tool_name == "sros_connect":
        name = args["name"]
        if name in _sessions:
            _close(name)
        _sessions[name] = DeviceSession(
            host=args["host"],
            port=args.get("port", 57400),
            username=args["username"],
            password=args["password"],
            skip_verify=args.get("skip_verify", True),
            insecure=args.get("insecure", False),
            timeout=args.get("timeout", 10),
        )
        gc = _connect(name)
        # Quick capabilities check
        try:
            caps = gc.capabilities()
            model_count = len(caps.get("supported_models", []))
            encodings = caps.get("supported_encodings", [])
            gnmi_ver = caps.get("gnmi_version", "unknown")
            return (
                f"✓ gNMI connected to '{name}' ({args['host']}:{args.get('port', 57400)})\n"
                f"  gNMI version: {gnmi_ver}\n"
                f"  Supported encodings: {', '.join(encodings)}\n"
                f"  YANG models: {model_count}"
            )
        except Exception:
            return f"✓ gNMI connected to '{name}' ({args['host']}:{args.get('port', 57400)})"

    elif tool_name == "sros_disconnect":
        return _close(args["name"])

    elif tool_name == "sros_get_config":
        gc = _connect(args["name"])
        return _do_get(gc, args["paths"], datatype="config")

    elif tool_name == "sros_get_state":
        gc = _connect(args["name"])
        return _do_get(gc, args["paths"], datatype="state")

    elif tool_name == "sros_set_update":
        gc = _connect(args["name"])
        return _do_set_update(gc, args["path"], args["value"])

    elif tool_name == "sros_set_replace":
        gc = _connect(args["name"])
        return _do_set_replace(gc, args["path"], args["value"])

    elif tool_name == "sros_set_delete":
        gc = _connect(args["name"])
        return _do_set_delete(gc, args["paths"])

    elif tool_name == "sros_cli_command":
        gc = _connect(args["name"])
        return _do_cli_command(gc, args["command"])

    elif tool_name == "sros_capabilities":
        gc = _connect(args["name"])
        return _do_capabilities(gc)

    elif tool_name == "sros_list_sessions":
        if not _sessions:
            return "No active sessions."
        lines = []
        for sname, sess in _sessions.items():
            connected = sname in _connections
            status = "✓ connected" if connected else "○ registered"
            tls = "insecure" if sess.insecure else ("skip_verify" if sess.skip_verify else "TLS")
            lines.append(
                f"  {sname}: {sess.host}:{sess.port} user={sess.username} [{status}] ({tls})"
            )
        return "Sessions:\n" + "\n".join(lines)

    else:
        return f"Unknown tool: {tool_name}"


# ═══════════════════════════════════════════════
# MCP Server entrypoint
# ═══════════════════════════════════════════════

app = Server("nokia-gnmi-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = handle_tool(name, arguments)
    except Exception as e:
        logger.exception(f"Tool '{name}' failed")
        result = f"✗ Error: {type(e).__name__}: {e}"
    return [TextContent(type="text", text=result)]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main():
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
