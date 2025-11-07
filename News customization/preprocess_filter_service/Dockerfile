# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY preprocess_filter_service.py .

# Copy local model folder into container
COPY model/ ./model/

# Expose the port Cloud Run expects
ENV PORT 8080

# Run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "preprocess_filter_service:app"]