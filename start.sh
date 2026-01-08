#!/bin/bash

# Start script for voice-chat services

echo "Starting LiveKit server and agent..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start services
docker-compose up --build

echo "Services started successfully!"
