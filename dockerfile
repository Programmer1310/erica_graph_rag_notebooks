# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /AI_Project

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Default command
CMD ["python", "app1.py"]

