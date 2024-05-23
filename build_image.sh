#!/bin/bash
set -e
# Set the image name
IMAGE_NAME="javers"

# Generate the tag based on the current date and time
TAG=$(date +"%Y%m%d%H%M")

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Optionally, you can also tag this build as the latest
# docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest
echo '------------------------------------------------------------------------------------------------'
docker images | head -n 2
echo '------------------------------------------------------------------------------------------------'
 