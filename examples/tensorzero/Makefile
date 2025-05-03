.PHONY: all

resources:
	docker compose up -d

tensorzero-logs:
	docker compose logs -f tensorzero

mcp-logs:
	docker compose logs -f mcp-server

stop:
	docker compose stop

agent:
	uv run agent.py --model=tensorzero.test_chat
