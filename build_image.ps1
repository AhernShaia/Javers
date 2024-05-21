# Set the image name
$IMAGE_NAME="javers"

# Generate the tag based on the current date and time
$TAG=Get-Date -Format "yyyyMMddHHmm"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Optionally, you can also tag this build as the latest
# docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest

docker images | Select-String -Pattern ${IMAGE_NAME} | Select-Object -First 1