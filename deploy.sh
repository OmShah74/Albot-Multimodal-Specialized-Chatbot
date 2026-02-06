#!/bin/bash

# Deployment script for Multimodal RAG System

echo "================================"
echo "Multimodal RAG System Deployment"
echo "================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not installed"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose not installed"
    exit 1
fi

echo "✓ Docker and Docker Compose found"
echo ""

# Create data directories
echo "Creating data directories..."
mkdir -p data/uploads data/database data/cache data/models
echo "✓ Data directories created"
echo ""

# Build and start services
echo "Building Docker images..."
docker-compose build

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "================================"
    echo "✓ Deployment successful!"
    echo "================================"
    echo ""
    echo "Access the application:"
    echo "  Gradio UI:    http://localhost:7860"
    echo "  FastAPI:      http://localhost:8000"
    echo "  API Docs:     http://localhost:8000/docs"
    echo "  ArangoDB UI:  http://localhost:8529"
    echo ""
    echo "Credentials:"
    echo "  ArangoDB: root / rootpassword"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop:      docker-compose down"
    echo ""
else
    echo "Error: Services failed to start"
    echo "Check logs: docker-compose logs"
    exit 1
