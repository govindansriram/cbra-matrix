#!/bin/bash

# Name of the container to check
CONTAINER_NAME="cbf-dev"

# Check if the container is running
if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q $CONTAINER_NAME; then
    echo "Container $CONTAINER_NAME is running."
else
    echo "Container $CONTAINER_NAME is not running."
    echo "Starting Container $CONTAINER_NAME"
    docker compose up -d
fi

docker exec $CONTAINER_NAME /bin/bash -c "cd /home && ./build.sh $@"
