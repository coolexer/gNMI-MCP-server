import asyncio
import sys
import unittest
from unittest.mock import patch

from mcp import Client, StdioServerParameters
from nokia_gnmi_mcp import server


class ServerTest(unittest.TestCase):
    def test_stdio_legacy_client(self):
        async def check():
            params = StdioServerParameters(command=sys.executable, args=["-m", "nokia_gnmi_mcp.server"])
            async with Client(params, mode="legacy") as client:
                tools = await client.list_tools()
                self.assertEqual(len(tools.tools), 10)
        asyncio.run(check())

    def test_mcp_tools_and_validation(self):
        async def check():
            async with Client(server.app) as client:
                tools = await client.list_tools()
                self.assertEqual(len(tools.tools), 10)
                self.assertTrue(all(t.input_schema for t in tools.tools))
                invalid = await client.call_tool("sros_connect", {"name": "r1"})
                self.assertTrue(invalid.is_error)
                self.assertIn("Invalid arguments", invalid.content[0].text)
                sessions = await client.call_tool("sros_list_sessions")
                self.assertFalse(sessions.is_error)
                self.assertIn("No active sessions", sessions.content[0].text)
        asyncio.run(check())

    def test_tls_failure_does_not_fallback_to_plaintext(self):
        class FailedClient:
            calls = []
            def __init__(self, **kwargs):
                self.calls.append(kwargs)
            def connect(self):
                raise RuntimeError("TLS certificate failure")
            def close(self):
                pass

        with patch.object(server, "gNMIclient", FailedClient):
            with self.assertRaisesRegex(RuntimeError, "TLS"):
                server.handle_tool("sros_connect", {
                    "name": "r1", "host": "localhost", "username": "u", "password": "p"
                })
        self.assertEqual(len(FailedClient.calls), 1)
        self.assertFalse(FailedClient.calls[0]["insecure"])
        self.assertFalse(FailedClient.calls[0]["skip_verify"])
        self.assertNotIn("r1", server._sessions)


if __name__ == "__main__":
    unittest.main()

