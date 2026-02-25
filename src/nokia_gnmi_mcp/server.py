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
  - YANG path search (no external tools needed)
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
import re
from dataclasses import dataclass
from pathlib import Path
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
    """Execute MD-CLI command via Nokia gNMI CLI extension."""
    try:
        result = gc.get(
            path=["/"],
            datatype="config",
            encoding="ascii",
        )
        return _json_pretty(result)
    except Exception:
        pass

    try:
        from pygnmi.spec.v080.gnmi_ext_pb2 import Extension, RegisteredExtension
        ext = Extension(
            registered_ext=RegisteredExtension(
                id=1001,
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
    output = {}
    if "supported_encodings" in result:
        output["supported_encodings"] = result["supported_encodings"]
    if "supported_models" in result:
        models = result["supported_models"]
        output["model_count"] = len(models)
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
        output["nokia_sample"] = [
            f"{m.get('name', '?')} ({m.get('version', '?')})"
            for m in nokia_models[:10]
        ]
    if "gnmi_version" in result:
        output["gnmi_version"] = result["gnmi_version"]
    return _json_pretty(output)


# ═══════════════════════════════════════════════
# YANG path search
# ═══════════════════════════════════════════════

# YANG files location — place Nokia YANG models here:
# <project_root>/yang/sros-25.10/YANG/nokia-combined/nokia-conf.yang
# <project_root>/yang/sros-25.10/YANG/nokia-combined/nokia-state.yang
# <project_root>/yang/sros-25.10/YANG/nokia-submodule/*.yang
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_YANG_BASE = _PROJECT_ROOT / "yang" / "sros-25.10" / "YANG"
_YANG_CACHE = _PROJECT_ROOT / "yang" / "cache"

# In-memory cache — loaded once per process
_path_cache: dict[str, list[str]] = {}


def _get_yang_paths(tree: str) -> list[str]:
    """
    Return cached YANG paths for 'configure' or 'state' tree.
    Builds cache from .yang files on first call.
    """
    if tree in _path_cache:
        return _path_cache[tree]

    _YANG_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _YANG_CACHE / f"{tree}-paths.txt"

    yang_sources = {
        "configure": _YANG_BASE / "nokia-combined" / "nokia-conf.yang",
        "state": _YANG_BASE / "nokia-combined" / "nokia-state.yang",
    }
    yang_file = yang_sources.get(tree)

    # Use disk cache if fresh
    if cache_file.exists() and yang_file and yang_file.exists():
        if cache_file.stat().st_mtime >= yang_file.stat().st_mtime:
            paths = cache_file.read_text(encoding="utf-8").splitlines()
            _path_cache[tree] = paths
            return paths

    # Build from YANG files
    if not _YANG_BASE.exists():
        return []

    paths = _build_paths_from_yang(_YANG_BASE, tree)
    cache_file.write_text("\n".join(paths), encoding="utf-8")
    _path_cache[tree] = paths
    return paths


def _build_paths_from_yang(yang_dir: Path, tree: str) -> list[str]:
    """
    Parse Nokia YANG submodule files and build a flat list of gNMI paths.
    Uses simple regex — no pyang dependency required.
    """
    prefix = "/configure" if tree == "configure" else "/state"
    submod_dir = yang_dir / "nokia-submodule"

    # Determine which submodule files belong to this tree
    if tree == "configure":
        pattern = re.compile(r"nokia-conf-.*\.yang$")
    else:
        pattern = re.compile(r"nokia-state-.*\.yang$")

    yang_files = []
    if submod_dir.exists():
        yang_files = [f for f in submod_dir.iterdir() if pattern.match(f.name)]

    # Also include the combined file
    combined = yang_dir / "nokia-combined" / f"nokia-{'conf' if tree == 'configure' else 'state'}.yang"
    if combined.exists():
        yang_files.append(combined)

    all_paths: set[str] = set()

    for yfile in yang_files:
        try:
            content = yfile.read_text(encoding="utf-8", errors="ignore")
            _extract_paths_from_content(content, prefix, all_paths)
        except Exception:
            continue

    result = sorted(all_paths)
    return result


def _extract_paths_from_content(content: str, prefix: str, paths: set[str]) -> None:
    """
    Extract YANG container/list/leaf paths using a simple stack-based parser.
    Handles nested structures and builds full paths.
    """
    # Match container, list, leaf, leaf-list declarations
    node_re = re.compile(
        r'^\s*(container|list|leaf|leaf-list|choice|case|augment|uses)\s+"?([a-zA-Z0-9_\-./]+)"?\s*\{?',
        re.MULTILINE,
    )

    stack: list[str] = []
    brace_depth = 0
    stack_depths: list[int] = []

    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()

        # Track brace depth
        open_braces = line.count("{")
        close_braces = line.count("}")
        brace_depth += open_braces - close_braces

        # Pop stack entries that are now out of scope
        while stack_depths and stack_depths[-1] >= brace_depth:
            stack.pop()
            stack_depths.pop()

        # Match node declarations
        m = node_re.match(line)
        if m:
            node_type = m.group(1)
            node_name = m.group(2)

            # Build current path
            current_path = prefix + "/" + "/".join(stack + [node_name]) if stack else prefix + "/" + node_name

            # Add key placeholder for lists
            if node_type == "list":
                current_path_with_key = current_path + "[<key>]"
                paths.add(current_path_with_key)
            elif node_type in ("container", "leaf", "leaf-list"):
                paths.add(current_path)

            # Push onto stack if opens a block
            if "{" in line:
                stack.append(node_name)
                stack_depths.append(brace_depth - 1)


def _do_yang_search(keyword: str, tree: str = "configure", max_results: int = 50) -> str:
    """
    Search YANG path cache for keyword.
    Returns matching paths sorted by relevance.
    """
    if not _YANG_BASE.exists():
        return (
            f"YANG files not found at: {_YANG_BASE}\n\n"
            f"To enable YANG search, place Nokia SR OS 25.10 YANG models at:\n"
            f"  {_YANG_BASE / 'nokia-combined' / 'nokia-conf.yang'}\n\n"
            f"Clone with:\n"
            f"  git clone --depth 1 --branch sros_25.10.r1 \\\n"
            f"    https://github.com/nokia/7x50_YangModels \\\n"
            f"    {_PROJECT_ROOT / 'yang' / 'sros-25.10'}"
        )

    paths = _get_yang_paths(tree)

    if not paths:
        return f"No paths found for tree='{tree}'. Check YANG files at {_YANG_BASE}"

    kw_lower = keyword.lower()

    # Score: exact segment match > substring match
    exact: list[str] = []
    partial: list[str] = []

    for p in paths:
        if kw_lower in p.lower():
            # Check if keyword matches a full path segment
            segments = re.split(r"[/\[\]]", p)
            if any(s.lower() == kw_lower for s in segments if s):
                exact.append(p)
            else:
                partial.append(p)

    results = exact + partial
    total = len(results)
    results = results[:max_results]

    if not results:
        return f"No paths found matching '{keyword}' in {tree} tree."

    lines = [
        f"Found {total} paths matching '{keyword}' in /{tree} tree",
        f"(showing {len(results)}):",
        "",
    ]
    lines.extend(results)

    if total > max_results:
        lines.append(f"\n... and {total - max_results} more. Use a more specific keyword.")

    return "\n".join(lines)


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
    Tool(
        name="yang_search",
        description=(
            "Search Nokia SR OS YANG model paths by keyword. "
            "Use this to find the correct gNMI path before calling sros_get_config/sros_set_update.\n\n"
            "Searches locally stored YANG files — no external tools needed.\n\n"
            "Examples:\n"
            "  keyword='vpls' tree='configure'  → find VPLS service paths\n"
            "  keyword='fpe'  tree='configure'  → find FPE/PXC paths\n"
            "  keyword='bgp-evpn'               → find EVPN BGP paths\n"
            "  keyword='end-dt2'                → find SRv6 EVPN function paths\n"
            "  keyword='isis' tree='state'      → find IS-IS state paths\n\n"
            "Requires YANG files at: <project>/yang/sros-25.10/YANG/\n"
            "Clone: git clone --depth 1 --branch sros_25.10.r1 "
            "https://github.com/nokia/7x50_YangModels <project>/yang/sros-25.10"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search keyword (case-insensitive). E.g. 'vpls', 'fpe', 'segment-routing-v6'",
                },
                "tree": {
                    "type": "string",
                    "enum": ["configure", "state"],
                    "description": "Which YANG tree to search: 'configure' (default) or 'state'",
                    "default": "configure",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["keyword"],
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

    elif tool_name == "yang_search":
        keyword = args["keyword"]
        tree = args.get("tree", "configure")
        max_results = args.get("max_results", 50)
        return _do_yang_search(keyword, tree, max_results)

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
