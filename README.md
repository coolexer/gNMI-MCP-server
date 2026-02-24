# Nokia SR OS gNMI MCP Server

MCP server for managing Nokia SR OS devices via gNMI (gRPC) from Claude Desktop.

## Features

- **gNMI Get** — retrieve configuration and operational state with YANG paths
- **gNMI Set** — update, replace, or delete configuration
- **MD-CLI commands** — via Nokia gNMI CLI extension
- **gNMI Capabilities** — discover supported models and encodings
- **Runtime credentials** — no passwords in config files
- **Multi-device** — manage multiple SR OS devices simultaneously
- **JSON output** — clean json_ietf encoding (easier to read than XML)

## Installation

```bash
cd nokia-gnmi-mcp

# Option A: uv (recommended)
uv sync

# Option B: pip
pip install -e .
```

## Claude Desktop Configuration

Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "nokia-gnmi": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\YourUser\\nokia-gnmi-mcp",
        "run",
        "nokia-gnmi-mcp"
      ]
    }
  }
}
```

## Usage

### 1. Connect to a device

> "Connect to my SR OS router at 192.168.1.1 via gNMI, username admin, password Nokia123, call it pe1"

### 2. Get configuration

> "Show me the router interfaces config on pe1"

Claude uses path: `/configure/router[router-name=Base]/interface`

### 3. Get operational state

> "What's the BGP neighbor state on pe1?"

Claude uses path: `/state/router[router-name=Base]/bgp/neighbor`

### 4. Update configuration

> "Create a loopback lo5 with IP 10.10.10.5/32 on pe1"

Claude uses gNMI Set update.

### 5. Delete configuration

> "Remove interface test from pe1"

Claude uses gNMI Set delete.

## Tools Reference

| Tool | gNMI Op | Description |
|------|---------|-------------|
| `sros_connect` | — | Connect to device (host, port, credentials, TLS options) |
| `sros_disconnect` | — | Close gNMI session |
| `sros_get_config` | Get (CONFIG) | Retrieve configuration by YANG path |
| `sros_get_state` | Get (STATE) | Retrieve operational state by YANG path |
| `sros_set_update` | Set (update) | Merge configuration changes |
| `sros_set_replace` | Set (replace) | Replace configuration subtree |
| `sros_set_delete` | Set (delete) | Delete configuration elements |
| `sros_cli_command` | CLI ext | MD-CLI show commands via gNMI |
| `sros_capabilities` | Capabilities | List supported models and encodings |
| `sros_list_sessions` | — | List active sessions |

## Nokia YANG Path Reference

### Configuration paths (`/configure/...`)

```
/configure/router[router-name=Base]/interface
/configure/router[router-name=Base]/bgp
/configure/router[router-name=Base]/isis[isis-instance=0]
/configure/port[port-id=1/1/c2/1]
/configure/card[slot-number=1]
/configure/service/vprn[service-name=CUST-1]
/configure/service/vpls[service-name=L2-1]
/configure/system
```

### State paths (`/state/...`)

```
/state/router[router-name=Base]/interface[interface-name=to-pe2]
/state/router[router-name=Base]/bgp/neighbor
/state/port[port-id=1/1/c2/1]
/state/card[slot-number=1]
/state/system/information
```

## Differences from NETCONF MCP

| Feature | NETCONF MCP | gNMI MCP |
|---------|-------------|----------|
| Protocol | NETCONF/SSH | gRPC/HTTP2 |
| Port | 830 | 57400 |
| Encoding | XML | JSON (json_ietf) |
| Candidate datastore | Yes (commit/rollback) | No (direct apply) |
| Output readability | XML verbose | JSON clean |
| Streaming telemetry | No | Possible (future) |

## Troubleshooting

**Connection refused**: Ensure gRPC is enabled on SR OS:
```
configure system grpc admin-state enable
configure system grpc allow-unsecure-connection
configure system grpc gnmi admin-state enable
configure system grpc gnmi auto-config-save true
```

**TLS errors**: Use `skip_verify=true` for lab or `insecure=true` for non-TLS connections.

**Timeout**: Increase timeout in `sros_connect`. Large state queries may need more time.
