.PHONY: help build run stop clean logs test install deps up down restart ps

# Variables
IMAGE_NAME = mcp-vector-store
CONTAINER_NAME = mcp-vector-store
COMPOSE_FILE = docker-compose.yml

help:
	@echo "MCP Vector Store Server - Available Commands:"
	@echo ""
	@echo "  make install       - Install Python dependencies"
	@echo "  make deps          - Show dependency tree"
	@echo "  make test          - Run test script"
	@echo "  make run           - Run MCP server locally (no Docker)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build         - Build Docker image"
	@echo "  make up            - Start all services with docker-compose"
	@echo "  make down          - Stop all services"
	@echo "  make restart       - Restart all services"
	@echo "  make logs          - Show logs from all services"
	@echo "  make logs-mcp      - Show logs from MCP server only"
	@echo "  make ps            - Show running containers"
	@echo "  make clean         - Remove containers and images"
	@echo ""
	@echo "Development:"
	@echo "  make pull-models   - Pull required Ollama models"
	@echo "  make check-weaviate - Check Weaviate health"
	@echo "  make check-ollama  - Check Ollama health"

install:
	pip install -r requirements.txt

deps:
	pip list

test:
	cd ../../../.. && python -m linus.mcp.vector_store.test_server

run:
	cd ../../../.. && python -m linus.mcp.vector_store

# Docker commands
build:
	cd ../../../.. && docker build -f src/linus/mcp/vector_store/Dockerfile -t $(IMAGE_NAME):latest .

up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 5
	@make check-weaviate
	@make check-ollama

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

logs-mcp:
	docker-compose logs -f mcp-vector-store

ps:
	docker-compose ps

clean:
	docker-compose down -v
	docker rmi $(IMAGE_NAME):latest 2>/dev/null || true

# Health checks
check-weaviate:
	@echo "Checking Weaviate health..."
	@curl -s http://localhost:18080/v1/.well-known/ready && echo "✓ Weaviate is ready" || echo "✗ Weaviate is not ready"

check-ollama:
	@echo "Checking Ollama health..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is ready" || echo "✗ Ollama is not ready"

pull-models:
	@echo "Pulling Ollama models..."
	docker exec ollama ollama pull nomic-embed-text:latest
	docker exec ollama ollama pull gemma3:27b
	@echo "✓ Models pulled successfully"

# Development helpers
shell:
	docker-compose exec mcp-vector-store /bin/bash

shell-weaviate:
	docker-compose exec weaviate /bin/bash

shell-ollama:
	docker exec -it ollama /bin/bash
