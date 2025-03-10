import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_servers(self, server_configs: Dict[str, Any]):
        stdio_transports = []
        for server_config in server_configs.values():
            print(server_config)
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env=None
            )
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio_transports.append(stdio_transport)
        for stdio_transport in stdio_transports:
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            self.sessions.append(session)

        for session in self.sessions:
            await session.initialize()
            response = await session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        # load tools
        tool2session = {}
        available_tools = []
        for session in self.sessions:
            response = await session.list_tools()
            tool_names = [tool.name for tool in response.tools]
            for name in tool_names:
                tool2session[name] = session
            available_tools += [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in response.tools]

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await tool2session[tool_name].call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                        "role": "assistant",
                        "content": content.text
                    })
                messages.append({
                    "role": "user",
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    server_config = json.load(open(sys.argv[1]))
    client = MCPClient()
    try:
        await client.connect_to_servers(server_config["mcpServers"])
        await client.chat_loop()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":

    asyncio.run(main())