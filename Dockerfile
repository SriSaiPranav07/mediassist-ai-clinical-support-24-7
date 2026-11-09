# ==============================================================================
# MediAssist - AI Clinical Decision Support System
# Production Docker Container Specification
# ==============================================================================

# Build Stage
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install essential system dependencies (libgl1 for Pillow/OpenCV, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user and group for security compliance
RUN groupadd -r mediassist && useradd -r -g mediassist -d /app -s /sbin/nologin mediassist

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . /app

# Ensure proper permissions for the non-root user
RUN chown -R mediassist:mediassist /app

# Switch to non-root execution
USER mediassist

# Expose Streamlit default network port
EXPOSE 8501

# Healthcheck to verify Streamlit application uptime
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch MediAssist Clinical Decision Support System
ENTRYPOINT ["streamlit", "run", "app.py"]
