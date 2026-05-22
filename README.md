# nokia-gnmi-mcp

FastMCP server for operating gNMI targets with Nokia-friendly defaults.

This server is intentionally focused on live device operations. It uses generic
`gnmi_*` tool names so it can be paired with Nokia labs today and other gNMI
targets later without confusing agents about the target vendor. For Nokia path
work, use it together with [`nokia-yang-mcp`](https://github.com/coolexer/nokia-yang-mcp):
first find and validate YANG/gNMI paths with the YANG MCP, then pass those paths
to this gNMI MCP for `get`, `set`, and capabilities operations.

## Features

- gNMI Get for configuration and operational state.
- gNMI Set update, replace, and delete.
- gNMI Capabilities discovery.
- Optional MD-CLI show command access through the Nokia gNMI CLI extension.
- Runtime credentials; no passwords in MCP client config.
- Multiple named device sessions in one MCP server process.
- Local fallback YANG path search when `yang/cache/*.txt` or local Nokia YANG
  files are present.

## Recommended Agent Flow

For configuration work, use both MCP servers:

1. Ask `nokia-yang-mcp` for candidate paths:
   - `yang_suggest_gnmi_candidates`
   - `yang_check_path_support`
   - `yang_stats`
2. Connect with this server:
   - `gnmi_connect`
3. Read current state/config:
   - `gnmi_get_config`
   - `gnmi_get_state`
4. Apply only validated paths:
   - `gnmi_set_update`
   - `gnmi_set_replace`
   - `gnmi_set_delete`

Example:

```text
1. nokia-yang-mcp.yang_suggest_gnmi_candidates(
     product="sros",
     feature_or_query="cpipe buffer jitter",
     platform="7750 SR-1",
     kind="config"
   )
2. nokia-gnmi-mcp.gnmi_get_config(
     name="pe1",
     paths=["/configure/service/cpipe[service-name=100]/sap[sap-id=1/1/1:100]/cem/packet/jitter-buffer"]
   )
```

## Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## MCP Client Configuration

Example config with both YANG and gNMI MCP servers:

```json
{
  "mcpServers": {
    "nokia-yang-mcp": {
      "command": "nokia-yang",
      "args": ["serve"]
    },
    "nokia-gnmi-mcp": {
      "command": "nokia-gnmi-mcp"
    }
  }
}
```

When a FastMCP client aggregates both servers into one tool namespace, tool
names may be prefixed with the server name, for example
`nokia-yang-mcp_yang_suggest_gnmi_candidates` and
`nokia-gnmi-mcp_gnmi_get_config`. Direct MCP clients usually show the shorter
tool names from the table below.

## Tools

| Tool | Purpose |
|---|---|
| `gnmi_connect` | Register and connect to a gNMI target. |
| `gnmi_disconnect` | Close a named session. |
| `gnmi_list_sessions` | Show registered/connected sessions. |
| `gnmi_get_config` | Read config paths with gNMI Get `datatype=config`. |
| `gnmi_get_state` | Read state paths with gNMI Get `datatype=state`. |
| `gnmi_set_update` | Merge config using gNMI Set update. |
| `gnmi_set_replace` | Replace config subtree using gNMI Set replace. |
| `gnmi_set_delete` | Delete config paths using gNMI Set delete. |
| `gnmi_capabilities` | Summarize gNMI capabilities. |
| `gnmi_cli_command` | Run an MD-CLI show command via Nokia gNMI extension when supported. |
| `yang_search` | Local fallback path search; prefer `nokia-yang-mcp` for authoritative search. |

Legacy `sros_*` aliases are still registered for existing prompts and MCP
clients, but new workflows should use the generic `gnmi_*` names.

## SR OS gRPC Setup

For labs, enable gRPC/gNMI on SR OS:

```text
configure system grpc admin-state enable
configure system grpc allow-unsecure-connection
configure system grpc gnmi admin-state enable
configure system grpc gnmi auto-config-save true
```

Use `insecure=true` for plain gRPC lab targets such as srsim/containerlab. Use
`skip_verify=true` for TLS with self-signed certificates.

## Development

```bash
python -m pytest
python -m ruff check src tests
python -m compileall src tests
```

The server entrypoint is `src/nokia_gnmi_mcp/server.py`; runtime logic is split
across focused modules under `src/nokia_gnmi_mcp/`.
