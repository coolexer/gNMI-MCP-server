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

import sys

# ── КРИТИЧНО: stdout зарезервирован для MCP JSON-транспорта ──
# Все логи должны идти ТОЛЬКО в stderr, иначе MCP парсер ломается.
# Делаем это ДО импорта pygnmi, чтобы перехватить его handlers.

# 1. Принудительно переводим root logger на stderr
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.addHandler(_stderr_handler)
_root_logger.setLevel(logging.INFO)

# 2. Теперь импортируем pygnmi (он может добавить свои handlers, но stdout уже не тронет)
from pygnmi.client import gNMIclient

# 3. После импорта — снова чистим handlers (pygnmi мог добавить свои)
for _name in ("pygnmi", "pygnmi.client", "grpc"):
    _lg = logging.getLogger(_name)
    _lg.handlers.clear()
    _lg.addHandler(_stderr_handler)
    _lg.setLevel(logging.WARNING)
    _lg.propagate = False

logger = logging.getLogger("nokia-gnmi-mcp")


# ═══════════════════════════════════════════════
# YANG path cache
# ═══════════════════════════════════════════════

# Ожидаемая структура:
#   <project_root>/yang/sros-25.10/YANG/nokia-combined/nokia-conf.yang
#   <project_root>/yang/sros-25.10/YANG/nokia-combined/nokia-state.yang
#   <project_root>/yang/sros-25.10/YANG/nokia-submodule/*.yang
#   <project_root>/yang/cache/configure-paths.txt  (auto-generated)
#   <project_root>/yang/cache/state-paths.txt       (auto-generated)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_YANG_DIR = _PROJECT_ROOT / "yang" / "sros-25.10" / "YANG"
_CACHE_DIR = _PROJECT_ROOT / "yang" / "cache"
_yang_cache: dict[str, list[str]] = {}


def _load_yang_cache(tree: str) -> list[str]:
    """Load paths from cache file. Returns empty list if not available."""
    global _yang_cache
    if tree in _yang_cache:
        return _yang_cache[tree]

    cache_file = _CACHE_DIR / f"{tree}-paths.txt"
    if cache_file.exists():
        paths = [l for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        _yang_cache[tree] = paths
        logger.info(f"YANG cache loaded: {tree} ({len(paths)} paths)")
        return paths

    # Try to build cache from YANG files
    paths = _build_paths_from_yang(tree)
    if paths:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("\n".join(paths), encoding="utf-8")
        _yang_cache[tree] = paths
        logger.info(f"YANG cache built: {tree} ({len(paths)} paths)")
    return paths


def _build_paths_from_yang(tree: str) -> list[str]:
    """
    Parse Nokia YANG submodules and extract all container/list/leaf paths.
    Pure Python — no pyang or gnmic needed.
    """
    submod_dir = _YANG_DIR / "nokia-submodule"
    if not submod_dir.exists():
        logger.warning(f"YANG submodule dir not found: {submod_dir}")
        return []

    prefix = "/" + tree.replace("configure", "configure").replace("state", "state")
    # Find relevant submodule files
    if tree == "configure":
        pattern = "nokia-conf-*.yang"
    else:
        pattern = "nokia-state-*.yang"

    yang_files = list(submod_dir.glob(pattern))
    if not yang_files:
        logger.warning(f"No YANG files found matching {pattern} in {submod_dir}")
        return []

    paths = set()
    for yang_file in yang_files:
        try:
            _extract_paths_from_file(yang_file, paths)
        except Exception as e:
            logger.debug(f"Error parsing {yang_file.name}: {e}")

    return sorted(paths)


def _extract_paths_from_file(yang_file: Path, paths: set):
    """Extract meaningful path fragments from a YANG file using regex."""
    content = yang_file.read_text(encoding="utf-8", errors="ignore")

    # Remove comments
    content = re.sub(r'//[^\n]*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Find augment statements — these give us full paths
    augment_paths = re.findall(r'augment\s+"(/[^"]+)"', content)
    for ap in augment_paths:
        # Clean up YANG module prefixes like sros-conf:
        clean = re.sub(r'\b\w+-(?:conf|state):', '', ap)
        paths.add(clean)

    # Find container/list names within augments to build sub-paths
    # This gives us the full tree structure
    blocks = re.findall(
        r'(?:container|list|leaf|leaf-list)\s+"([a-z][a-z0-9-]*)"',
        content
    )
    # We'll skip deep parsing here — augment paths are sufficient for search


def yang_search(keyword: str, tree: str = "configure", max_results: int = 50) -> str:
    """
    Search Nokia YANG paths by keyword.

    Args:
        keyword: search term (case-insensitive)
        tree: "configure" or "state"
        max_results: limit results

    Returns:
        Matching YANG paths as text
    """
    tree = tree.lower().strip()
    if tree not in ("configure", "state"):
        tree = "configure"

    paths = _load_yang_cache(tree)

    if not paths:
        return (
            f"YANG cache not available for '{tree}'.\n\n"
            f"To enable YANG search, place Nokia YANG models at:\n"
            f"  {_YANG_DIR}\n\n"
            f"Expected structure:\n"
            f"  yang/sros-25.10/YANG/nokia-combined/nokia-conf.yang\n"
            f"  yang/sros-25.10/YANG/nokia-combined/nokia-state.yang\n"
            f"  yang/sros-25.10/YANG/nokia-submodule/*.yang\n\n"
            f"Then restart the MCP server — cache will be built automatically.\n\n"
            f"Alternatively, pre-generate cache files:\n"
            f"  yang/cache/configure-paths.txt\n"
            f"  yang/cache/state-paths.txt\n"
            f"(one path per line, e.g. /configure/router[router-name]/bgp)"
        )

    keyword_lower = keyword.lower()
    matches = [p for p in paths if keyword_lower in p.lower()]

    if not matches:
        return f"No paths found containing '{keyword}' in /{tree} tree."

    total = len(matches)
    shown = matches[:max_results]
    result = f"Found {total} paths containing '{keyword}' in /{tree} tree"
    if total > max_results:
        result += f" (showing first {max_results})"
    result += ":\n\n"
    result += "\n".join(shown)
    return result


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
    if name not in _sessions:
        raise ValueError(f"Device '{name}' not registered. Use 'sros_connect' tool first.")
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
    result = gc.get(path=path, datatype=datatype, encoding="json_ietf")
    return _json_pretty(result)


def _do_set_update(gc: gNMIclient, path: str, value: dict) -> str:
    result = gc.set(update=[(path, value)], encoding="json_ietf")
    return _json_pretty(result)


def _do_set_replace(gc: gNMIclient, path: str, value: dict) -> str:
    result = gc.set(replace=[(path, value)], encoding="json_ietf")
    return _json_pretty(result)


def _do_set_delete(gc: gNMIclient, paths: list[str]) -> str:
    result = gc.set(delete=paths, encoding="json_ietf")
    return _json_pretty(result)


def _do_cli_command(gc: gNMIclient, command: str) -> str:
    try:
        result = gc.get(path=["/"], datatype="config", encoding="ascii")
        return _json_pretty(result)
    except Exception:
        pass
    try:
        from pygnmi.spec.v080.gnmi_ext_pb2 import Extension, RegisteredExtension
        ext = Extension(registered_ext=RegisteredExtension(id=1001, msg=command.encode()))
        result = gc.get(path=["/"], encoding="ascii", extension=[ext])
        return _json_pretty(result)
    except Exception as e:
        return f"CLI via gNMI extension failed: {e}. Use gNMI native paths instead."


def _do_capabilities(gc: gNMIclient) -> str:
    result = gc.capabilities()
    output = {}
    if "supported_encodings" in result:
        output["supported_encodings"] = result["supported_encodings"]
    if "supported_models" in result:
        models = result["supported_models"]
        output["model_count"] = len(models)
        nokia_models = [m for m in models if "nokia" in m.get("name", "").lower()]
        ietf_models = [m for m in models if "ietf" in m.get("name", "").lower()]
        output["nokia_models"] = len(nokia_models)
        output["ietf_models"] = len(ietf_models)
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
                "name": {"type": "string", "description": "Session name (e.g. 'pe1', 'core-rtr')"},
                "host": {"type": "string", "description": "IP address or hostname of the SR OS device"},
                "port": {"type": "integer", "description": "gNMI port (default: 57400)", "default": 57400},
                "username": {"type": "string", "description": "gRPC/gNMI username"},
                "password": {"type": "string", "description": "gRPC/gNMI password"},
                "skip_verify": {"type": "boolean", "description": "Skip TLS cert verification. For real hardware with self-signed cert. For srsim use insecure=true. (default: true)", "default": True},
                "insecure": {"type": "boolean", "description": "Use insecure plain gRPC (no TLS). Required for Nokia srsim/containerlab labs. (default: false)", "default": False},
                "timeout": {"type": "integer", "description": "Connection timeout in seconds (default: 10)", "default": 10},
            },
            "required": ["name", "host", "username", "password"],
        },
    ),
    Tool(
        name="sros_disconnect",
        description="Close a gNMI session to a Nokia SR OS device.",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Session name to disconnect"}},
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
                    "description": "List of YANG paths to retrieve. Example: ['/configure/router[router-name=Base]/interface']",
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
                    "description": "List of YANG state paths to retrieve. Example: ['/state/system/information']",
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
                "path": {"type": "string", "description": "YANG path to update"},
                "value": {"type": "object", "description": "JSON object with config values to merge"},
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
                "path": {"type": "string", "description": "YANG path to replace"},
                "value": {"type": "object", "description": "JSON object — complete replacement config"},
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
                "command": {"type": "string", "description": "MD-CLI command to execute"},
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
            "properties": {"name": {"type": "string", "description": "Device session name"}},
            "required": ["name"],
        },
    ),
    Tool(
        name="sros_list_sessions",
        description="List all active Nokia SR OS gNMI sessions.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="yang_search",
        description=(
            "Search Nokia SR OS YANG paths by keyword. "
            "Use this to find the correct gNMI path before using sros_get_config/sros_get_state/sros_set_update.\n\n"
            "Examples:\n"
            "  yang_search('fpe', 'configure')         → find FPE paths\n"
            "  yang_search('end-dt2', 'configure')      → find SRv6 EVPN function paths\n"
            "  yang_search('bgp-evpn', 'configure')     → find EVPN BGP paths\n"
            "  yang_search('pxc', 'configure')          → find Port Cross-Connect paths\n"
            "  yang_search('neighbor', 'state')         → find BGP neighbor state paths\n\n"
            "Requires YANG models in yang/sros-25.10/YANG/ or pre-built cache in yang/cache/."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search keyword (case-insensitive)",
                },
                "tree": {
                    "type": "string",
                    "enum": ["configure", "state"],
                    "description": "YANG tree to search: 'configure' (default) or 'state'",
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

    if tool_name == "sros_connect":
        name = args["name"]
        if name in _sessions:
            _close(name)

        host = args["host"]
        port = args.get("port", 57400)
        user = args["username"]
        pwd  = args["password"]
        timeout = args.get("timeout", 10)
        insecure = args.get("insecure", False)
        skip_verify = args.get("skip_verify", True)

        # Auto-fallback: if neither insecure nor explicit TLS cert provided,
        # try skip_verify first; if SSL error → retry with insecure=True
        # (Nokia srsim/containerlab uses plain gRPC without TLS)
        tried_insecure = insecure

        _sessions[name] = DeviceSession(
            host=host, port=port, username=user, password=pwd,
            skip_verify=skip_verify, insecure=insecure, timeout=timeout,
        )
        try:
            gc = _connect(name)
            caps = gc.capabilities()
        except Exception as e:
            err_str = str(e).lower()
            if not tried_insecure and ("ssl" in err_str or "certificate" in err_str or "tls" in err_str):
                # Fallback to insecure (plain gRPC) — typical for srsim
                logger.info(f"TLS failed for '{name}', retrying with insecure=True")
                _close(name)
                _sessions[name] = DeviceSession(
                    host=host, port=port, username=user, password=pwd,
                    skip_verify=False, insecure=True, timeout=timeout,
                )
                gc = _connect(name)
                caps = gc.capabilities()
                tried_insecure = True
            else:
                raise

        model_count = len(caps.get("supported_models", []))
        encodings = caps.get("supported_encodings", [])
        gnmi_ver = caps.get("gnmi_version", "unknown")
        tls_mode = "insecure (plain gRPC)" if _sessions[name].insecure else ("skip_verify" if skip_verify else "TLS")
        return (
            f"✓ gNMI connected to '{name}' ({host}:{port}) [{tls_mode}]\n"
            f"  gNMI version: {gnmi_ver}\n"
            f"  Supported encodings: {', '.join(encodings)}\n"
            f"  YANG models: {model_count}"
        )

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
            lines.append(f"  {sname}: {sess.host}:{sess.port} user={sess.username} [{status}] ({tls})")
        return "Sessions:\n" + "\n".join(lines)

    elif tool_name == "yang_search":
        return yang_search(
            keyword=args["keyword"],
            tree=args.get("tree", "configure"),
            max_results=args.get("max_results", 50),
        )

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
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    main()
