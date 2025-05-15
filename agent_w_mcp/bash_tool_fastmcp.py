import subprocess as sp

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("bash")

@mcp.tool()
async def bash(command: str) -> str:
    """Run a command in the Bash shell."""
    try:
        out = sp.check_output(command, shell=True, stderr=sp.STDOUT)
        return out.decode("utf-8").rstrip()
    except sp.CalledProcessError as error:
        return error.output


if __name__ == "__main__":
    mcp.run(transport='stdio')
