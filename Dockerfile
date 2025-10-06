# 🚀 Space Intelligence Platform Docker Image
FROM python:3.11-slim

# Set metadata
LABEL org.opencontainers.image.title="Space Intelligence Platform"
LABEL org.opencontainers.image.description="Real-time space intelligence with NASA, NOAA, and ISS data"
LABEL org.opencontainers.image.version="3.0"
LABEL org.opencontainers.image.authors="Space Intelligence Team"

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Copy requirements first for better caching
COPY --chown=app:app requirements*.txt ./

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Create data directory for live updates
RUN mkdir -p data/live

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PATH="/home/app/.local/bin:${PATH}"

# Health check to ensure application is running
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the Space Intelligence Platform
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
