# Use an official slim Python image
FROM python:3.9-slim

# Create a non-root user to run the app
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /home/appuser/app

# Copy application files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtual environment
RUN pip install --no-cache-dir --upgrade pip virtualenv && \
    virtualenv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PATH="/home/appuser/app/venv/bin:$PATH"
ENV STREAMLIT_PORT=8501
ENV STREAMLIT_HOME="/home/appuser/app"

# Ensure the app runs as non-root
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app\streamlit_app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
