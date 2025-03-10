import asyncio
import os
import warnings
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import ShellTool
from langchain_community.tools import BraveSearch
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

load_dotenv()
warnings.filterwarnings('ignore')


team_members = ["frontend developer", "backend developer", "devops engineer"]

options = team_members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

llm = ChatAnthropic(model="claude-3-5-sonnet-latest", max_retries=5)

bash_tool = ShellTool()
search_tool = BraveSearch.from_api_key(api_key=os.getenv("BRAVE_API_KEY"), search_kwargs={"count": 3})
fs_tool = FileManagementToolkit(root_dir="./").get_tools()
human = load_tools(
    ["human"],
    llm=llm,
)

available_tools = [bash_tool, search_tool] + fs_tool + human


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


class State(MessagesState):
    next: str


class MultiAgent:
    def __init__(self):
        self.llm = llm

    async def create_agent_graph(self):
        def supervisor_node(state: State) -> Command[Literal[*team_members, "__end__"]]:
            messages = [
                           {"role": "system", "content": system_prompt},
                       ] + state["messages"]
            response = self.llm.with_structured_output(Router).invoke(messages)
            goto = response["next"]
            if goto == "FINISH":
                goto = END

            return Command(goto=goto, update={"next": goto})

        frontend_agent = create_react_agent(
            self.llm,
            tools=available_tools,
            prompt=(
                "You are a frontend developer. Create frontend applications in framework "
                "requested by user. Frontend should work together with backend created by"
                " backend developer. You can ask backend or devops engineer for help"
            )
        )

        async def frontend_node(state: State) -> Command[Literal["supervisor"]]:
            result = await frontend_agent.ainvoke(state)
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="frontend developer")
                    ]
                },
                goto="supervisor",
            )

        backend_agent = create_react_agent(
            self.llm,
            tools=available_tools,
            prompt=(
                "You are a backend developer. Create backend part of the applications in framework "
                "requested by user. Backend should work together with frontend created by frontend "
                "developer. You can ask frontend or devops engineer for help"
            )
        )

        async def backend_node(state: State) -> Command[Literal["supervisor"]]:
            result = await backend_agent.ainvoke(state)
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="backend developer")
                    ]
                },
                goto="supervisor",
            )

        devops_agent = create_react_agent(
            self.llm,
            tools=available_tools,
            prompt=(
                "You are a devops engineer. Help backend and frontend engineers to deploy their "
                "applications as instructed in the user request. You can ask frontend or backend "
                "engineer any clarification questions about their code"
            )
        )

        async def devops_node(state: State) -> Command[Literal["supervisor"]]:
            result = await devops_agent.ainvoke(state)
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=result["messages"][-1].content, name="devops engineer")
                    ]
                },
                goto="supervisor",
            )

        builder = StateGraph(State)
        builder.add_edge(START, "supervisor")
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("frontend developer", frontend_node)
        builder.add_node("backend developer", backend_node)
        builder.add_node("devops engineer", devops_node)
        self.graph = builder.compile()


async def main():
    user_prompt = (
        "I want to build a website for a conference, it should have several pages, "
        "namely: 1. Intro page about conference, 2. Page for people to submit their talks, "
        "3. Page with submitted talks. Frontend part needs to be written in react, backend - in fastapi. "
        "I want to store the submissions in postgresql database. "
        "In the end run the project in docker and docker compose and give me the local url to test. "
        "You can ask human client for any clarifications"
    )
    client = MultiAgent()
    try:
        await client.create_agent_graph()
        async for s in client.graph.astream(
                {"messages": [("user", user_prompt)]},
                subgraphs=True,
                stream_mode="values",
        ):
            print(s)
            print("----")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())