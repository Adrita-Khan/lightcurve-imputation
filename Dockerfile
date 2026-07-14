# ============================================================
# Dockerfile for lightcurve-imputation
# Produces a self-contained image that runs the full pipeline.
# ============================================================
FROM python:3.10-slim

# Metadata
LABEL maintainer="Adrita Khan"
LABEL description="Lightcurve imputation thesis reproducibility container"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full repository
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p data/results figures tables

# Default command: run the full reproducibility pipeline
CMD ["python", "run_all.py", "--config", "configs/experiment.yml"]
