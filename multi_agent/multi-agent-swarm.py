import asyncio
import os
import warnings
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from langchain_community.tools import ShellTool
from langchain_community.tools import BraveSearch
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph_swarm import create_swarm, create_handoff_tool

load_dotenv()
warnings.filterwarnings('ignore')


sonnet = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_retries=5, max_tokens=16384, temperature=0)
haiku = ChatAnthropic(model="claude-haiku-4-5-20251001", max_retries=5, max_tokens=16384, temperature=0)

bash_tool = ShellTool()
search_tool = BraveSearch.from_api_key(api_key=os.getenv("BRAVE_API_KEY"), search_kwargs={"count": 3})
fs_tool = FileManagementToolkit(root_dir="./").get_tools()

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
pw_tools = toolkit.get_tools()

human = load_tools(["human"])

transfer_to_backend = create_handoff_tool(agent_name="backend", description="Transfer control to the backend agent.")
transfer_to_frontend = create_handoff_tool(agent_name="frontend", description="Transfer control to the frontend agent.")
transfer_to_devops = create_handoff_tool(agent_name="devops", description="Transfer control to the devops agent.")
transfer_to_qa = create_handoff_tool(agent_name="qa", description="Transfer control to the QA agent.")
transfer_to_product_manager = create_handoff_tool(agent_name="product_manager", description="Return control to the product manager.")


team_members = ["backend", "frontend", "devops", "qa"]

available_tools = [bash_tool, search_tool] + fs_tool + pw_tools + human

product_manager_prompt = (
    "You are a product manager and project planner. A user will give you a high‑level "
    "software development request. Break the request into small, atomic steps. "
    f"Assign each step to one of the following specialists: {', '.join(team_members)}. "
    "Use exactly the agent name as the label for the step (e.g., 'Step 1 (backend): …'). "
    "After outlining a step, immediately call the corresponding transfer tool (e.g., transfer_to_backend) "
    "to hand control to that specialist. If you need clarification from the user, ask using the human tool."
)

# Prompts for specialists: execute only their labelled steps, hand back control, and ask human if needed
backend_prompt = (
    "You are a senior backend developer. Implement task assigned to you by the product manager. "
    "For each step labelled 'backend': "
    "1. Produce a detailed sub‑plan describing how you will accomplish the step. "
    "2. Execute your sub‑plan using the available tools. "
    "3. Immediately call the transfer_to_product_manager tool to return control to the product manager. "
    "Do not move on to the next step until the product manager reassigns you. "
    "Only execute steps labelled for backend; do not perform tasks labelled for frontend, devops, or QA. "
    "Use the human tool if you need to ask the user a question."
)
frontend_prompt = (
    "You are a senior frontend developer. Implement task assigned to you by the product manager. "
    "For each step labelled 'frontend': "
    "1. Produce a detailed sub‑plan describing how you will accomplish the step. "
    "2. Execute your sub‑plan using the available tools. "
    "3. Immediately call the transfer_to_product_manager tool to return control to the product manager. "
    "Do not move on to the next step until the product manager reassigns you. "
    "Only execute steps labelled for frontend; do not perform tasks labelled for backend, devops, or QA. "
    "Use the human tool if you need to ask the user a question."
)
devops_prompt = (
    "You are a senior devops engineer. Implement task assigned to you by the product manager. "
    "For each step labelled 'devops': "
    "1. Produce a detailed sub‑plan describing how you will accomplish the step. "
    "2. Execute your sub‑plan using the available tools. "
    "3. Immediately call the transfer_to_product_manager tool to return control to the product manager. "
    "Do not move on to the next step until the product manager reassigns you. "
    "Only execute steps labelled for devops; do not perform tasks labelled for backend, frontend, or QA. "
    "Use the human tool if you need to ask the user a question."
)
qa_prompt = (
    "You are a senior QA engineer. Implement task assigned to you by the product manager. "
    "For each step labelled 'qa': "
    "1. Produce a detailed sub‑plan describing how you will accomplish the step. "
    "2. Execute your sub‑plan using the available tools. "
    "3. Immediately call the transfer_to_product_manager tool to return control to the product manager. "
    "Do not move on to the next step until the product manager reassigns you. "
    "Only execute steps labelled for QA; do not perform tasks labelled for backend, frontend, or devops. "
    "Use the human tool if you need to ask the user a question."
)

class MultiAgentSwarm:
    def __init__(self):
        self.sonnet = sonnet
        self.haiku = haiku

    async def create_swarm(self):
        backend_agent = create_react_agent(
            self.haiku,
            tools=available_tools + [transfer_to_product_manager],
            prompt=backend_prompt,
            name="backend",
        )
        frontend_agent = create_react_agent(
            self.haiku,
            tools=available_tools + [transfer_to_product_manager],
            prompt=frontend_prompt,
            name="frontend",
        )
        devops_agent = create_react_agent(
            self.haiku,
            tools=available_tools + [transfer_to_product_manager],
            prompt=devops_prompt,
            name="devops",
        )
        qa_agent = create_react_agent(
            self.haiku,
            tools=available_tools + [transfer_to_product_manager],
            prompt=qa_prompt,
            name="qa",
        )
        product_manager_agent = create_react_agent(
            self.sonnet,
            tools=available_tools + [transfer_to_backend, transfer_to_frontend, transfer_to_devops, transfer_to_qa],
            prompt=product_manager_prompt,
            name="product_manager",
        )
        self.swarm = create_swarm(
            agents=[product_manager_agent, backend_agent, frontend_agent, devops_agent, qa_agent],
            default_active_agent="product_manager",
        ).compile()


def formatting(s):
    node, event = s
    if len(node) == 0:
        print("Entering graph")
        print(event)
        return
    agent_type = node[0].split(':')[0]
    print(f"\n\033[92mCurrent agent\033[0m - \033[91m{agent_type}\033[0m")
    event_type = list(event.keys())[0]
    if event_type == "tools":
        if event[event_type]['messages'][0].content:
            print(f"\033[94mTool call result\033[0m: {event[event_type]['messages'][0].content}")
    elif event_type == "agent":
        content = event[event_type]['messages'][0].content
        if isinstance(content, str):
            print(f"\033[92m{agent_type}\033[0m: {content}")
            return
        agent_messages = list(filter(lambda x: x["type"] == "text", content))
        if agent_messages:
            print(f"\033[92m{agent_type}\033[0m: {agent_messages[0]['text']}")
        tools = list(filter(lambda x: x["type"] == "tool_use", content))
        if tools:
            for tool in tools:
                if tool["input"]:
                    print(f"\033[92m{agent_type}\033[0m: calling tool \033[93m{tool['name']} \033[0mwith the following input:")
                    for key, value in tool["input"].items():
                        print(f"\033[96m{key}\033[0m: \033[97m{value}\033[0m")
                else:
                    print(f"\033[92m{agent_type}\033[0m: using tool \033[93m{tool['name']}\033[0m")
    else:
        print("event", event)


async def main():
    user_prompt = (
        "I want to build a website for a conference, it should have several pages, "
        "namely: 1. Intro page about conference, 2. Page for people to submit their talks, "
        "3. Page with submitted talks. Frontend part needs to be written in react, backend - in fastapi. "
        "I want to store the submissions in postgresql database. "
        "In the end run the project in docker and docker compose and give me the local url to test. "
    )
    client = MultiAgentSwarm()
    try:
        await client.create_swarm()
        async for s in client.swarm.astream(
                {"messages": [("user", user_prompt)]},
                subgraphs=True,
                stream_mode="values",
        ):
            formatting(s)
            print("-" * 30)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())