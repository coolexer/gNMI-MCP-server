# gNMI FastMCP Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Nokia gNMI MCP server into a tested FastMCP package focused on smooth operation beside `nokia-yang-mcp`.

**Architecture:** Split the current monolithic `server.py` into focused modules for logging, YANG cache/search, session management, gNMI operations, and FastMCP tools. Preserve existing tool names and behavior where possible, but make the YANG MCP integration workflow explicit in tool descriptions and README.

**Tech Stack:** Python 3.11+, FastMCP, pygnmi, pytest, ruff, hatchling.

---

### Task 1: Regression Tests And Package Metadata

- [x] Add pytest/ruff dev dependencies and CI.
- [x] Add tests for YANG search when no local YANG tree exists.
- [x] Add tests for session registration, TLS fallback, JSON formatting, and FastMCP tool functions.

### Task 2: Split Runtime Modules

- [x] Move logging setup into `logging_setup.py`.
- [x] Move dataclasses into `models.py`.
- [x] Move YANG cache/search into `yang_cache.py`.
- [x] Move gNMI client operations into `gnmi_ops.py`.
- [x] Move session lifecycle into `sessions.py`.

### Task 3: FastMCP Server

- [x] Replace manual `mcp.server.Server` schema/dispatch with decorated FastMCP tool functions.
- [x] Keep initial tool names: `sros_connect`, `sros_disconnect`, `sros_get_config`, `sros_get_state`, `sros_set_update`, `sros_set_replace`, `sros_set_delete`, `sros_cli_command`, `sros_capabilities`, `sros_list_sessions`, `yang_search`.
- [x] Add vendor-neutral `gnmi_*` tool names as the primary API and retain `sros_*` as legacy aliases.
- [x] Ensure all logs go to stderr and stdout remains MCP-safe.

### Task 4: Docs And Validation

- [x] Rewrite README around gNMI + `nokia-yang-mcp` workflow.
- [x] Remove legacy protocol comparison.
- [x] Add MIT license and CI.
- [x] Run pytest, ruff, compileall, and a FastMCP smoke test.
