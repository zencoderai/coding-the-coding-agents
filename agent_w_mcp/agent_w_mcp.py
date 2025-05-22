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


ANTHROPIC_WEB_SEARCH = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5
    }
]
MAX_TOKENS = 8192
MODEL = "claude-sonnet-4-20250514"
SYSTEM_PROMPT = """
For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.
Put DONE to the message when you are done with the task
"""


class MCPClient:
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_servers(self, server_configs: Dict[str, Any]):
        stdio_transports = []
        for server_config in server_configs.values():
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
            print("\033[92mConnected to server with tools\033[0m:", ", ".join([tool.name for tool in tools]))

    async def process_query(self, user_query: str, max_iter: int = 30) -> str:
        messages = [
            {
                "role": "user",
                "content": user_query
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
        available_tools += ANTHROPIC_WEB_SEARCH

        response = self.anthropic.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            tools=available_tools,
            stop_sequences=["DONE"],
            system=SYSTEM_PROMPT
        )

        counter = 1
        while counter < max_iter:
            print(f"\n\033[95mLoop\033[0m: {counter}")

            tool_results = []
            assistant_message_contents = []
            # for anthropic's web search tool the agent outputs multipls text blocks, so combining them into one
            is_previous_content_text = False
            all_texts = []
            for content in response.content:
                if content.type == "text" and content.text:
                    message = content.text.strip("\n").rstrip()
                    if message:
                        assistant_message_contents.append({
                            "type": "text",
                            "text": message
                        })
                        all_texts.append(message)
                    if not is_previous_content_text:
                        is_previous_content_text = True
                elif content.type == "tool_use":
                    # reset text compilation
                    is_previous_content_text = False
                    if all_texts:
                        print("\033[92mAgent\033[0m: ", "\n".join(all_texts))
                        all_texts = []
                    tool_name = content.name
                    tool_args = content.input
                    print(f"\033[96mCalling tool\033[0m \033[93m{tool_name}\033[0m \033[96mwith the following input\033[0m: \033[93m{tool_args}\033[0m")
                    result = await tool2session[tool_name].call_tool(tool_name, tool_args)
                    tool_results.append({"type": "tool_result", "tool_use_id": content.id, "content": result.content[0].text})
                    print(f"\033[94mTool call result\033[0m: {result.content[0].text}")
                    assistant_message_contents.append({
                        "type": "tool_use",
                        "id": content.id,
                        "name": tool_name,
                        "input": tool_args
                    })
                elif content.type == "server_tool_use":
                    # reset text compilation
                    is_previous_content_text = False
                    if all_texts:
                        print("\033[92mAgent\033[0m: ", "\n".join(all_texts))
                        all_texts = []
                    tool_name = content.name
                    tool_args = content.input
                    print(f"\033[96mCalling remote tool\033[0m \033[93m{tool_name}\033[0m \033[96mwith the following input\033[0m: \033[93m{tool_args}\033[0m")
                    if hasattr(content, 'text') and content.text:
                        assistant_message_contents.append({
                            "type": "text",
                            "text": content.text.rstrip()
                        })
                elif content.type == "web_search_tool_result":
                    # reset text compilation
                    is_previous_content_text = False
                    if all_texts:
                        print("\033[92mAgent\033[0m: ", "\n".join(all_texts))
                        all_texts = []
                    for ws_result in content.content:
                        print(f"\033[94mGot web-search result\033[0m, search request - \033[93m{ws_result['title']}\033[0m, URL - \033[93m{ws_result['url']}\033[0m")
                    if hasattr(content, 'text') and content.text:
                        assistant_message_contents.append({
                            "type": "text",
                            "text": content.text.rstrip()
                        })

            messages.append({
                "role": "assistant",
                "content": assistant_message_contents
            })
            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=available_tools,
                stop_sequences=["DONE"],
                system=SYSTEM_PROMPT
            )

            counter += 1
            if response.stop_reason == "stop_sequence":
                if response.content and response.content[0].text:
                    all_texts.append(response.content[0].text.strip("\n"))
                if all_texts:
                    print("\033[92mAgent\033[0m: ", "\n".join(all_texts))
                break
            else:
                if all_texts:
                    print("\033[92mAgent\033[0m: ", "\n".join(all_texts))

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    server_config = json.load(open("server.json"))
    client = MCPClient()
    try:
        await client.connect_to_servers(server_config["mcpServers"])
        await client.process_query("Execute file fail.py and see if there are any errors. If there are, fix them")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":

    asyncio.run(main())