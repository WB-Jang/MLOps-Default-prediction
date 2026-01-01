FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry==1.6.1

# Copy dependency files
COPY pyproject.toml ./

# Configure poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.api.main"]
