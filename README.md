# AI Coding agents and how to code them

Code for the talk on coding the AI coding agents

## MCP Servers
### Time
```json
        "time": {
            "command": "npx",
            "args": ["-y", "time-mcp"]
        }
 ```

### Git
```json
        "git": {
            "command": "uvx",
            "args": ["mcp-server-git", "--repository", "/ABSOLUTE/PATH/TO/GIT/REPO"]
        }
```

### Docker
```json
        "mcp-server-docker": {
            "command": "uvx",
            "args": [
                "mcp-server-docker"
            ]
        }
```

### Grafana
Download and unpack from https://github.com/grafana/mcp-grafana/releases, provide absolute path to mcp-grafana binary. Update GRAFANA_URL and GRAFANA_API_KEY as needed.
```json
        "grafana": {
            "command": "/PATH/TO/mcp-grafana",
            "args": [],
            "env": {
                "GRAFANA_URL": "http://localhost:3001",
                "GRAFANA_API_KEY": "SERVICE_ACCOUNT_TOKEN"
            }
        }
```

### Slack
`.env.slack` file contains bot token and team id. Create a new app at https://api.slack.com/apps and add it to your workspace. Then create a bot user and get its token. Also find team ID by going to `Manage Workspace -> About -> Team ID`.
More instructions here: https://github.com/modelcontextprotocol/servers/tree/main/src/slack#setup
```
SLACK_BOT_TOKEN=YOUR_TOKEN
SLACK_TEAM_ID=YOUR_TEAM_ID
```
MCP config:
```json
        "slack": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "--env-file",
                "/PATH/TO/.env.slack",
                "mcp/slack"
            ]
        }
```

### Github
Create Github PAT https://github.com/github/github-mcp-server?tab=readme-ov-file#prerequisites and put it into `.env.github` file
```
GITHUB_PERSONAL_ACCESS_TOKEN=YOUR_GITHUB_PAT
```

MCP config:
```json
        "github": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "--env-file",
                "/PATH/TO/.env.github",
                "ghcr.io/github/github-mcp-server"
            ]
        }
```
