# Nokia SR OS gNMI MCP server

A local stdio MCP server for Nokia SR OS gNMI. It exposes configuration and state reads, configuration update/replace/delete, capabilities, device sessions, and optional YANG path search.

## Requirements

Python 3.11 or newer and a gNMI-enabled Nokia SR OS device. The server uses the official MCP Python SDK 2.x and pyGNMI. MCP 2.x serves both current and older MCP clients over stdio.

## Install

```powershell
uv sync --python 3.13
uv run nokia-gnmi-mcp
```

For Codex, configure `~/.codex/config.toml` with an absolute path to the environment executable:

```toml
[mcp_servers.nokia-gnmi-mcp]
command = 'C:\path\to\gNMI-MCP-server\.venv\Scripts\nokia-gnmi-mcp.exe'
args = []
```

Restart Codex after changing the configuration. A typical first call is `sros_connect` with a session name, host, username, and password. Credentials are kept in process memory for that session; they are never written to the configuration file.

TLS certificate verification is enabled by default. For a lab with a self-signed certificate, explicitly set `skip_verify=true`; for a plaintext lab endpoint, explicitly set `insecure=true`. The server never retries a failed TLS connection over plaintext.

## Tools

| Tool | Purpose |
| --- | --- |
| `sros_connect` / `sros_disconnect` | Open or close a named gNMI session |
| `sros_get_config` / `sros_get_state` | Read YANG paths with JSON IETF encoding |
| `sros_set_update` / `sros_set_replace` / `sros_set_delete` | Change configuration; replace removes unspecified nodes in its subtree |
| `sros_capabilities` | Read gNMI version, encodings, and model summary |
| `sros_list_sessions` | Show in-memory sessions without passwords |
| `yang_search` | Search local Nokia YANG paths |

`yang_search` requires Nokia YANG submodule files under a `yang` directory. Set `NOKIA_GNMI_YANG_DIR` to the absolute path of that directory. The server creates `cache/configure-paths.txt` and `cache/state-paths.txt` within it. You can also place prebuilt files there, one path per line. The directory must be writable to build the cache.

The previous `sros_cli_command` tool was removed: pyGNMI's `get()` does not accept the vendor extension argument used by that tool. Use gNMI paths for reads or a separate CLI transport for MD-CLI commands.

## Check

```powershell
uv run python -m unittest discover -s tests -v
```

The test suite checks MCP tool discovery, argument validation, and that a TLS failure never triggers a plaintext retry. Device operations require a live Nokia endpoint and are not exercised by the local tests.
