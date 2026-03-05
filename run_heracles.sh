#!/usr/bin/env bash
set -e

docker compose -f docker/docker-compose.yaml up -d ollama
#docker compose -f docker/docker-compose.yaml exec -it ollama ollama pull llama3.1
docker compose -f docker/docker-compose.yaml exec -it ollama ollama pull gemma3:27b

#docker compose -f docker/docker-compose.yaml up -d neo4j hydra_visualization

#docker compose -f docker/docker-compose.yaml run --rm chatdsg
docker compose -f docker/docker-compose.yaml run --rm cli
docker compose -f docker/docker-compose.yaml run --rm heracles_dev

#docker compose -f docker/docker-compose.yaml down hydra_visualization