This is WIP for multi agent system based on Google's [A2A](https://a2a-protocol.org/latest/) and Python wrapper [python-a2a](https://github.com/themanojdesai/python-a2a).

To start the server, run `python multi-agent-a2a.py`. Then send request:
```bash
curl -X POST http://localhost:6000/tasks/send \
-H "Content-Type: application/json" \
-d '{"message": {"content": {"type": "text", "text": "Build me a simple web app"}, "role": "user"}}'
```
