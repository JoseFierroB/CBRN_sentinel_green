# Lightweight Python Image for AgentBeats Phase 1
FROM python:3.11-slim

# Install system dependencies for RDKit and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Setup Work Directory
WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY . .

# Create unprivileged user
RUN useradd -m -u 1000 sentinel
USER sentinel

# Env vars
ENV PYTHONPATH=/app

# AgentBeats requires these args: --host, --port, --card-url
# Use a shell script entrypoint to properly handle args
ENTRYPOINT ["python", "-m", "src.server"]
