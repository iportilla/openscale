# Default port for the Streamlit UI
PORT ?= 8501
IMAGE_NAME = loan-approval-monitor

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the container with the specified port
run:
	docker run -p $(PORT):8501 $(IMAGE_NAME)

# Stop and remove all containers of this image
stop:
	@CONTAINERS=$$(docker ps -aq --filter ancestor=$(IMAGE_NAME)); \
	if [ -n "$$CONTAINERS" ]; then \
		echo "Stopping and removing containers..."; \
		docker rm -f $$CONTAINERS; \
	else \
		echo "No containers found to stop."; \
	fi

# Clean up local Docker images
clean: stop
	@if [ $$(docker images -q $(IMAGE_NAME) | wc -l) -gt 0 ]; then \
		echo "Removing image $(IMAGE_NAME)..."; \
		docker rmi $(IMAGE_NAME); \
	else \
		echo "No image found to remove."; \
	fi

help:
	@echo "Available commands:"
	@echo "  make build          - Build the Docker image"
	@echo "  make run            - Run the container (default port 8501)"
	@echo "  make run PORT=9000  - Run the container on a custom port"
	@echo "  make stop           - Stop the running container"
	@echo "  make clean          - Remove the Docker image"
