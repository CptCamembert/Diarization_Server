FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Create directories for mounted volumes
# These will be mount points for external volumes
RUN mkdir -p /app/model /app/embeddings

# Make model and embeddings directories writable
RUN chmod 777 /app/model /app/embeddings

# Set environment variables for model and embeddings paths
ENV MODEL_DIR=/app/model
ENV EMBEDDINGS_DIR=/app/embeddings

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]