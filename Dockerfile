FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set PYTHONPATH so absolute imports work across the app
ENV PYTHONPATH=/app

# Generate initial FAISS index data on build
RUN python scripts/generate_data.py
RUN python retrieval/vector_store.py

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
