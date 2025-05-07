.PHONY: all

build:
	docker compose build

up:
	docker compose up -d

logs:
	docker compose logs -f

tensorzero-logs:
	docker compose logs -f gateway

mcp-logs:
	docker compose logs -f mcp-server

minio-logs:
	docker compose logs -f minio

stop:
	docker compose stop

agent:
	uv run agent.py --model=tensorzero.test_chat

simple-agent:
	uv run simple_agent.py --model=tensorzero.simple_chat

image-test:
	uv run image_demo.py 
