# Dockerfile for PDF Text RAG System
# Python 3.13 environment with all dependencies

FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for models and data if they don't exist
RUN mkdir -p models pdf CMs templates

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose ports (for Flask apps)
EXPOSE 5000 5001 5003 5004

# Default command (can be overridden)
CMD ["python3", "--version"]
