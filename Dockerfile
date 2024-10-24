# Use the official PyTorch image with CUDA 11.6 and Python 3.10
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Expose the port (optional)
EXPOSE 5000

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Run the application with Gunicorn, using a single worker
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "sync", "app:app"]
