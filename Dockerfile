# Use a slimmed-down Python runtime as a base image
FROM python:3.11-slim

# Copy the requirements file
COPY requirements.txt ./

# Install build dependencies for any Python packages that may need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && apt-get purge -y --auto-remove build-essential libssl-dev \
    && python3 -m spacy download en_core_web_sm \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Run the application with Gunicorn, using Gevent workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "gevent", "app:app"]
