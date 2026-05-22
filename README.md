# nokia-gnmi-mcp

FastMCP server for operating Nokia SR OS devices over gNMI.

This server is intentionally focused on live device operations. Use it together
with [`nokia-yang-mcp`](https://github.com/coolexer/nokia-yang-mcp): first find
and validate YANG/gNMI paths with the YANG MCP, then pass those paths to this
gNMI MCP for `get`, `set`, and capabilities operations.

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
   - `sros_connect`
3. Read current state/config:
   - `sros_get_config`
   - `sros_get_state`
4. Apply only validated paths:
   - `sros_set_update`
   - `sros_set_replace`
   - `sros_set_delete`

Example:

```text
1. nokia-yang-mcp.yang_suggest_gnmi_candidates(
     product="sros",
     feature_or_query="cpipe buffer jitter",
     platform="7750 SR-1",
     kind="config"
   )
2. nokia-gnmi-mcp.sros_get_config(
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

## Tools

| Tool | Purpose |
|---|---|
| `sros_connect` | Register and connect to a Nokia SR OS gNMI target. |
| `sros_disconnect` | Close a named session. |
| `sros_list_sessions` | Show registered/connected sessions. |
| `sros_get_config` | Read config paths with gNMI Get `datatype=config`. |
| `sros_get_state` | Read state paths with gNMI Get `datatype=state`. |
| `sros_set_update` | Merge config using gNMI Set update. |
| `sros_set_replace` | Replace config subtree using gNMI Set replace. |
| `sros_set_delete` | Delete config paths using gNMI Set delete. |
| `sros_capabilities` | Summarize gNMI capabilities. |
| `sros_cli_command` | Run an MD-CLI show command via Nokia gNMI extension when supported. |
| `yang_search` | Local fallback path search; prefer `nokia-yang-mcp` for authoritative search. |

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
