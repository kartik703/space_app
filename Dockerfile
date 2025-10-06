# 🚀 Space Intelligence Platform Docker Image
FROM python:3.11-slim

# Set build arguments for multi-platform support
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETARCH

# Set metadata
LABEL org.opencontainers.image.title="Space Intelligence Platform"
LABEL org.opencontainers.image.description="Real-time space intelligence with NASA, NOAA, and ISS data"
LABEL org.opencontainers.image.version="3.0"
LABEL org.opencontainers.image.authors="Space Intelligence Team"
LABEL org.opencontainers.image.source="https://github.com/kartik703/space_app"

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other requirements
# Use --no-install-recommends to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app \
    && chown -R app:app /app
    
# Copy requirements first for better caching
COPY requirements*.txt ./

# Install Python dependencies as root for better performance
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Switch to non-root user
USER app

# Copy application code
COPY --chown=app:app . .

# Create data directory for live updates
RUN mkdir -p data/live logs \
    && chmod 755 data/live logs

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
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
