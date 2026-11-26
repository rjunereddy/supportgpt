.PHONY: build up down test logs shell deploy-local

# Build the Docker image
build:
	docker-compose build

# Start the application in detached mode
up:
	docker-compose up -d

# Stop the application
down:
	docker-compose down

# View logs from the API container
logs:
	docker-compose logs -f api

# Run the verification script inside the container
test:
	docker-compose exec api python scripts/verify.py

# Drop into a shell inside the container
shell:
	docker-compose exec api /bin/bash

# A helper to build and run everything from scratch
deploy-local: down build up
	@echo "================================================="
	@echo "SupportGPT is running locally on http://localhost:8000"
	@echo "Check API documentation at http://localhost:8000/docs"
	@echo "================================================="
