import json
import os
import threading
import subprocess
from typing import Dict, List, Tuple, Optional

import logging

import openai

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

# Configure root logger to include timestamp, log level and agent name (logger name)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

from python_a2a import (
    A2AServer,
    run_server,
    AgentNetwork,
    TaskStatus,
    TaskState,
    AgentCard,
    AgentSkill,
    A2AClient,
)
from python_a2a.mcp import FastMCP, MCPClient


# -----------------------------------------------------------------------------
# MCP server factory functions
# -----------------------------------------------------------------------------

def create_backend_tools(port: int) -> Tuple[FastMCP, int]:
    """Create an MCP server exposing backend file manipulation tools.

    The backend developer often needs to read and write source code files.  The
    tools are limited to the directory ``/workspace/backend`` within the
    container for safety.  Additional tools could be added here as needed.

    Returns the MCP server instance along with the port it should run on.
    """
    mcp = FastMCP(name="Backend Tools", description="Filesystem utilities for backend tasks")
    allowed_dir = "/workspace/backend"

    @mcp.tool(name="read_file", description="Read the contents of a file")
    def read_file(path: str) -> str:
        full_path = os.path.abspath(path)
        if not full_path.startswith(os.path.abspath(allowed_dir)):
            raise ValueError("Access to this path is not allowed")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    @mcp.tool(name="write_file", description="Write content to a file, overwriting if it exists")
    def write_file(path: str, content: str) -> str:
        full_path = os.path.abspath(path)
        if not full_path.startswith(os.path.abspath(allowed_dir)):
            raise ValueError("Access to this path is not allowed")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {full_path}"

    return mcp, port


def create_frontend_tools(port: int) -> Tuple[FastMCP, int]:
    """Create an MCP server exposing frontend file manipulation tools.

    Frontend tasks typically involve editing HTML, CSS and JavaScript.  The
    allowed directory is ``/workspace/frontend``.
    """
    mcp = FastMCP(name="Frontend Tools", description="Filesystem utilities for frontend tasks")
    allowed_dir = "/workspace/frontend"

    @mcp.tool(name="read_file", description="Read the contents of a file")
    def read_file(path: str) -> str:
        full_path = os.path.abspath(path)
        if not full_path.startswith(os.path.abspath(allowed_dir)):
            raise ValueError("Access to this path is not allowed")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    @mcp.tool(name="write_file", description="Write content to a file, overwriting if it exists")
    def write_file(path: str, content: str) -> str:
        full_path = os.path.abspath(path)
        if not full_path.startswith(os.path.abspath(allowed_dir)):
            raise ValueError("Access to this path is not allowed")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {full_path}"

    return mcp, port


def create_devops_tools(port: int) -> Tuple[FastMCP, int]:
    """Create an MCP server exposing devops utilities.

    DevOps tasks may require running shell commands (e.g. building Docker
    images or deploying services).  For security reasons commands are
    restricted to a whitelist of safe commands.  The whitelist can be
    extended as needed.
    """
    mcp = FastMCP(name="DevOps Tools", description="Utilities for deployment and system management")
    allowed_commands = ["ls", "echo", "pwd", "whoami"]

    @mcp.tool(name="run_command", description="Run a shell command and return its output")
    def run_command(command: str) -> str:
        parts = command.split()
        if not parts or parts[0] not in allowed_commands:
            raise ValueError(f"Command '{parts[0]}' is not allowed")
        try:
            result = subprocess.check_output(parts, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {e}")
        return result.strip()

    return mcp, port


def create_qa_tools(port: int) -> Tuple[FastMCP, int]:
    """Create an MCP server exposing QA utilities.

    The QA engineer needs to run tests and inspect results.  A simple
    ``run_tests`` tool is provided which executes pytest in a dedicated
    ``/workspace/tests`` folder.  Additional QA tools could be added for
    linting or static analysis.
    """
    mcp = FastMCP(name="QA Tools", description="Utilities for running tests and quality checks")
    tests_dir = "/workspace/tests"

    @mcp.tool(name="run_tests", description="Execute pytest on the tests directory")
    def run_tests() -> str:
        if not os.path.exists(tests_dir):
            return f"No tests directory found at {tests_dir}"
        try:
            result = subprocess.check_output([
                "pytest", tests_dir, "-q", "--disable-warnings"
            ], text=True)
        except subprocess.CalledProcessError as e:
            # Even when tests fail pytest returns non-zero; capture output
            result = e.output
        return result.strip()

    return mcp, port


# -----------------------------------------------------------------------------
# Helper functions for interacting with OpenAI
# -----------------------------------------------------------------------------

def call_openai_llm(
    system_prompt: str,
    conversation: List[Dict[str, str]],
    model: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> str:
    """Call the OpenAI ChatCompletion API with a given system prompt and conversation.

    Args:
        system_prompt: The system instructions describing the agent's role.
        conversation: A list of message dicts with 'role' and 'content' keys.
        model: Optional override for the OpenAI model; defaults to 'gpt-4o' or
            the environment variable ``OPENAI_MODEL``.

    Returns:
        The assistant's response content.

    This helper centralizes error handling and model selection.  It assumes
    ``OPENAI_API_KEY`` is set in the environment.  See the OpenAI‑powered
    agent example for reference【136238366048500†L508-L526】.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set")
    selected_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    # Build messages array for the chat API
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})
    for msg in conversation:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Retrieve a logger for this agent if specified
    logger: Optional[logging.Logger] = None
    if agent_name:
        logger = logging.getLogger(agent_name)
        # Log the input conversation and system prompt
        try:
            logger.info("LLM input: system_prompt=%s messages=%s", system_prompt, messages)
        except Exception:
            pass

    try:
        response = openai.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    # Log the output
    if logger:
        try:
            logger.info("LLM output: %s", content)
        except Exception:
            pass

    return content.strip()


def parse_tool_call(response: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse a tool call directive from the assistant's response.

    The convention used here is that if a response begins with ``TOOL_CALL``
    followed by a JSON object, the JSON will contain the name of the tool
    under ``"tool"`` and the arguments in ``"args"``.  Example:

    ``TOOL_CALL {"tool": "read_file", "args": {"path": "/workspace/backend/app.py"}}``

    Returns:
        A tuple of (tool_name, arguments dict) if a tool call is detected,
        otherwise None.
    """
    prefix = "TOOL_CALL"
    if response.startswith(prefix):
        try:
            json_str = response[len(prefix):].strip()
            payload = json.loads(json_str)
            tool_name = payload.get("tool")
            args = payload.get("args", {}) or {}
            if tool_name:
                return tool_name, args
        except json.JSONDecodeError:
            # If JSON fails to parse treat as a normal response
            return None
    return None


def parse_agent_query(response: str) -> Optional[Tuple[str, str]]:
    """Parse an agent query directive from the assistant's response.

    If the assistant wants to consult another agent it should emit a line
    starting with ``ASK_AGENT`` followed by the agent name and the query.
    Example:

        ``ASK_AGENT backend What's the database schema?``

    Returns:
        A tuple of (agent_name, query) if such a directive is found, else None.
    """
    prefix = "ASK_AGENT"
    if response.startswith(prefix):
        remainder = response[len(prefix):].strip()
        parts = remainder.split(None, 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


# -----------------------------------------------------------------------------
# Base class for subordinate agents
# -----------------------------------------------------------------------------

class DeveloperAgent(A2AServer):
    """A generic subordinate agent representing a software engineer.

    Each DeveloperAgent uses an OpenAI LLM to decide how to handle its
    tasks.  It has access to its own MCP server via ``mcp_client`` and
    can communicate with other agents through ``network_client``.  The
    system prompt sets the agent's persona and instructs it on how to
    interact with tools and other agents.  The handle_task method
    extracts the incoming user query, consults the LLM, performs any
    requested tool invocations, handles cross‑agent queries, and
    assembles the final response.
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        mcp_url: str,
        peer_endpoints: Dict[str, str],
        port: int,
    ) -> None:
        # Define an agent card describing capabilities
        agent_card = AgentCard(
            name=name,
            description=description,
            url=f"http://localhost:{port}",
            version="1.0.0",
            skills=[
                AgentSkill(
                    name="Respond to development tasks",
                    description="Discuss and perform tasks related to the agent's role using tools when necessary."
                )
            ],
        )
        super().__init__(agent_card=agent_card)
        self.system_prompt = system_prompt
        self.mcp_client = MCPClient(mcp_url)
        # Map agent names to their HTTP endpoints for cross‑agent queries
        self.peer_endpoints = peer_endpoints
        self.port = port

    def handle_task(self, task):  # type: ignore[override]
        """Process an incoming task using the agent's LLM and tools.

        The task's message content is treated as a user query.  The agent
        consults its LLM with the system prompt and the query to produce
        a response.  If the response indicates a tool call (via the
        ``TOOL_CALL`` convention) the tool is invoked through the MCP
        client and the result is appended to the conversation.  If the
        response requests another agent's input (via ``ASK_AGENT``) the
        agent uses its network client to forward the question and awaits
        a reply.  After satisfying tool calls and agent queries, the
        agent asks its LLM to craft a final answer.  The completed
        answer is stored in the task's artifacts and the task is
        marked as completed.
        """
        # Extract the raw text from the incoming A2A task
        message_data = task.message or {}
        content = message_data.get("content", {})
        user_text = ""
        # Support both TextContent dictionaries and plain strings
        if isinstance(content, dict):
            user_text = content.get("text", "")
        elif isinstance(content, str):
            user_text = content
        user_text = user_text.strip()

        # Initialise conversation with user message
        conversation: List[Dict[str, str]] = [
            {"role": "user", "content": user_text}
        ]

        # First, let the LLM reason about the task and decide next steps
        response = call_openai_llm(self.system_prompt, conversation, agent_name=self.agent_card.name)

        # Check for tool call directive
        tool_call = parse_tool_call(response)
        if tool_call:
            tool_name, args = tool_call
            try:
                tool_result = self.mcp_client.call_tool(tool_name, args)
            except Exception as e:
                tool_result = f"Error calling tool {tool_name}: {e}"
            # Append the tool invocation and result to the conversation
            conversation.append({"role": "assistant", "content": response})
            conversation.append({"role": "tool", "content": json.dumps(tool_result)})
            # Ask the LLM to produce a final answer incorporating tool output
            final_reply = call_openai_llm(self.system_prompt, conversation, agent_name=self.agent_card.name)
            answer_text = final_reply
        else:
            # Check for agent query directive
            agent_query = parse_agent_query(response)
            if agent_query:
                agent_name, query = agent_query
                # Resolve the endpoint for the target agent
                endpoint = self.peer_endpoints.get(agent_name)
                if endpoint:
                    try:
                        client = A2AClient(endpoint)
                        other_response = client.ask(query)
                        other_content = getattr(other_response, "content", other_response)
                        if isinstance(other_content, dict):
                            other_text = other_content.get("text", str(other_content))
                        else:
                            other_text = str(other_content)
                    except Exception as e:
                        other_text = f"Error contacting agent {agent_name}: {e}"
                else:
                    other_text = f"Unknown agent '{agent_name}'"
                # Append the assistant directive and the other agent's response
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "assistant", "content": f"Response from {agent_name}: {other_text}"})
                final_reply = call_openai_llm(self.system_prompt, conversation, agent_name=self.agent_card.name)
                answer_text = final_reply
            else:
                # No tool call or agent query; use the LLM's first answer directly
                answer_text = response

        # Populate task artifacts
        task.artifacts = [{
            "parts": [
                {"type": "text", "text": answer_text}
            ]
        }]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task


# -----------------------------------------------------------------------------
# Controller (Main) Agent
# -----------------------------------------------------------------------------

class MainAgent(A2AServer):
    """The primary orchestrator that plans and coordinates development tasks.

    The main agent receives a high‑level user query and uses its LLM to
    break the work down into subtasks for the backend, frontend, devops and
    QA agents.  It dispatches these tasks via an internal network and
    continues to refine its plan based on the responses.  Once all
    subtasks are complete the main agent synthesizes a final answer for
    the user.
    """

    def __init__(self, network: AgentNetwork, subordinate_ports: Dict[str, int], port: int) -> None:
        """Initialize the main planner agent.

        Args:
            network: An ``AgentNetwork`` used for registering subordinate agents.  The
                network is not used for messaging in this implementation but is kept
                for compatibility with the overall A2A ecosystem.
            subordinate_ports: A mapping from subordinate agent names (backend,
                frontend, devops, qa) to the port on which their A2A server is
                listening.  This mapping is used to construct ``A2AClient``
                instances when delegating tasks.
            port: The port for the main agent's A2A server.
        """
        # Build an agent card for the main agent
        agent_card = AgentCard(
            name="Main Planner",
            description="Plans and coordinates tasks among backend, frontend, devops and QA agents",
            url=f"http://localhost:{port}",
            version="1.0.0",
            skills=[
                AgentSkill(
                    name="Project Planning",
                    description="Break down user requests into developer tasks and assemble results."
                )
            ],
        )
        super().__init__(agent_card=agent_card)
        # Save the network for registration purposes
        self.network = network
        # Store subordinate ports for message routing via A2AClient
        self.subordinate_ports = subordinate_ports
        self.port = port
        # System prompt for planning
        self.system_prompt = (
            "You are a project planner and task router.  A user will give you a high‑level "
            "software development request.  Respond with a JSON array of tasks, "
            "each containing two fields: 'agent' (one of backend, frontend, devops, qa) "
            "and 'task' (a short description of what that agent should do).  Only "
            "include agents that are relevant.  Do not perform any work yourself."
        )

    def handle_task(self, task):  # type: ignore[override]
        """Plan the work, delegate to subordinate agents and return the final answer."""
        # Extract the user message
        message_data = task.message or {}
        content = message_data.get("content", {})
        user_text = ""
        if isinstance(content, dict):
            user_text = content.get("text", "")
        elif isinstance(content, str):
            user_text = content
        user_text = user_text.strip()

        # Ask the planner LLM to propose a breakdown of tasks
        planning_conv = [{"role": "user", "content": user_text}]
        plan_response = call_openai_llm(self.system_prompt, planning_conv, agent_name=self.agent_card.name)
        # Try to parse the response as JSON.  Remove markdown code fences if present.
        tasks_plan: List[Dict[str, str]] = []
        parsed = False
        # First attempt: extract JSON array between brackets to handle code fences
        import re
        match = re.search(r"\[[\s\S]*\]", plan_response)
        if match:
            json_str = match.group(0)
            try:
                tasks_plan = json.loads(json_str)
                parsed = True
            except Exception:
                parsed = False
        if not parsed:
            # Fallback: direct JSON parse without extraction
            try:
                tasks_plan = json.loads(plan_response)
                parsed = True
            except Exception:
                parsed = False
        if not parsed:
            # If JSON parsing fails, fall back to a simple heuristic: split lines with a colon
            tasks_plan = []
            for line in plan_response.split("\n"):
                if ":" in line:
                    agent, description = line.split(":", 1)
                    tasks_plan.append({"agent": agent.strip().strip('"').lower(), "task": description.strip().strip('",')})

        # Dispatch tasks to subordinate agents and collect responses
        results: Dict[str, str] = {}
        for item in tasks_plan:
            agent_name = item.get("agent", "").lower()
            description = item.get("task", "")
            if agent_name not in ["backend", "frontend", "devops", "qa"]:
                continue
            # Resolve the subordinate's port and construct an A2AClient
            port = self.subordinate_ports.get(agent_name)
            if port is None:
                results[agent_name] = f"No endpoint configured for agent '{agent_name}'"
                continue
            endpoint = f"http://localhost:{port}"
            try:
                client = A2AClient(endpoint)
                response_message = client.ask(description)
                # Extract text from response_message.content; handle both dict and string
                resp_content = getattr(response_message, "content", response_message)
                if isinstance(resp_content, dict):
                    results[agent_name] = resp_content.get("text", str(resp_content))
                else:
                    results[agent_name] = str(resp_content)
            except Exception as e:
                results[agent_name] = f"Error communicating with {agent_name} agent: {e}"

        # Synthesize final answer using LLM
        synthesis_prompt = (
            "You are an assistant summarizing the results of a multi‑agent software development project. "
            "Provide a concise, cohesive summary of what each agent accomplished based on the following JSON:\n"
            f"{json.dumps(results, indent=2)}"
        )
        synthesis_conv = [{"role": "user", "content": synthesis_prompt}]
        final_answer = call_openai_llm("You are a helpful assistant.", synthesis_conv, agent_name=self.agent_card.name)

        # Populate task artifacts with the final answer and mark as complete
        task.artifacts = [{
            "parts": [
                {"type": "text", "text": final_answer}
            ]
        }]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task


# -----------------------------------------------------------------------------
# Entry point and server startup
# -----------------------------------------------------------------------------

def run_mcp_servers() -> Dict[str, Tuple[FastMCP, int]]:
    """Instantiate and return MCP servers for each subordinate agent.

    Each server is returned along with its port, so the caller can
    launch the servers in threads.  Ports are assigned starting from 6101.
    """
    servers: Dict[str, Tuple[FastMCP, int]] = {}
    # Assign distinct ports for each tool server
    servers["backend"] = create_backend_tools(6101)
    servers["frontend"] = create_frontend_tools(6102)
    servers["devops"] = create_devops_tools(6103)
    servers["qa"] = create_qa_tools(6104)
    return servers


def run_subordinate_agents(mcp_servers: Dict[str, Tuple[FastMCP, int]]) -> Dict[str, Tuple[DeveloperAgent, int]]:
    """Instantiate subordinate DeveloperAgents and return them with their ports.

    This function starts the MCP servers and then creates the subordinate agents.
    Each agent receives a dictionary of peer endpoints so it can issue queries
    directly via ``A2AClient``.  The agents are not started here; they are
    returned for the caller to launch.
    """
    agents: Dict[str, Tuple[DeveloperAgent, int]] = {}
    # Map subordinate names to system prompts
    prompts = {
        "backend": (
            "You are a senior backend developer.  When given a task, decide whether you "
            "need to read or write code files using the provided tools.  You can call a tool "
            "by responding with 'TOOL_CALL' followed by a JSON object containing the "
            "tool name and arguments.  If you need information from another agent, respond with "
            "'ASK_AGENT <agent> <question>'.  Otherwise explain what you did or will do."
        ),
        "frontend": (
            "You are a creative frontend developer.  Decide when to edit or read HTML/CSS/JS files using "
            "the tools.  Use the 'TOOL_CALL' convention for tool invocations and 'ASK_AGENT' to consult "
            "other agents.  Provide clear updates about your progress."
        ),
        "devops": (
            "You are a devops engineer responsible for deployment and infrastructure.  Determine when "
            "to run shell commands via the devops tools and when to ask other agents for information.  "
            "Use the 'TOOL_CALL' convention to run commands.  Provide concise status updates."
        ),
        "qa": (
            "You are a QA engineer ensuring quality through testing.  Decide when to run tests using "
            "the QA tools.  Use 'TOOL_CALL' to run tests and 'ASK_AGENT' to request clarifications.  "
            "Describe test outcomes clearly."
        ),
    }
    # Assign ports for subordinate A2A servers starting at 6001
    base_port = 6001
    names = ["backend", "frontend", "devops", "qa"]
    # Precompute endpoints for each agent
    endpoints = {name: f"http://localhost:{base_port + idx}" for idx, name in enumerate(names)}
    for name in names:
        mcp, mcp_port = mcp_servers[name]
        # Start the MCP server in a background thread
        threading.Thread(target=run_server, args=(mcp,), kwargs={"port": mcp_port}, daemon=True).start()
        # Create the developer agent with peer endpoints
        agent_port = int(endpoints[name].split(":")[-1])
        agent = DeveloperAgent(
            name=f"{name.capitalize()} Agent",
            description=f"Handles {name} development tasks",
            system_prompt=prompts[name],
            mcp_url=f"http://localhost:{mcp_port}",
            peer_endpoints=endpoints,
            port=agent_port,
        )
        agents[name] = (agent, agent_port)
    return agents


def run_all_servers() -> None:
    """Launch MCP servers, subordinate agents and the main planner agent."""
    # Create an agent network; this will be shared by the main and subordinate agents
    network = AgentNetwork(name="Development Network")

    # Create and start MCP servers
    mcp_servers = run_mcp_servers()

    # Instantiate subordinate agents and start their MCP servers in threads
    subordinate_agents = run_subordinate_agents(mcp_servers)

    # Add subordinate agents to the network with their URLs
    for name, (_, port) in subordinate_agents.items():
        network.add(name, f"http://localhost:{port}")

    # Create the main agent and add to the network
    main_port = 6000
    main_agent = MainAgent(network=network, subordinate_ports={k: p for k, (_, p) in subordinate_agents.items()}, port=main_port)

    # Start subordinate A2A servers in background threads
    for name, (agent, port) in subordinate_agents.items():
        threading.Thread(target=run_server, args=(agent,), kwargs={"port": port}, daemon=True).start()

    # Finally run the main agent (blocking)
    print(f"Main agent listening on port {main_port}")
    run_server(main_agent, port=main_port)


if __name__ == "__main__":
    # To avoid blocking the interactive environment when imported, the script
    # only starts the servers when executed as a program.
    run_all_servers()
