#!/bin/bash

# Name of the container to check
CONTAINER_NAME="cbf-dev"

# Check if the container is running
if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q $CONTAINER_NAME; then
    echo "Container $CONTAINER_NAME is running."
    # Run the command in the container
    docker exec $CONTAINER_NAME /bin/bash -c "cd /home && ./build.sh"
else
    echo "Container $CONTAINER_NAME is not running."
fi
