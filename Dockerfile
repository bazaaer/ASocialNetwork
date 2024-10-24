# Use the official PyTorch image with CUDA 11.6 and Python 3.10
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3-dev \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install xformers from prebuilt wheel
RUN pip install --no-cache-dir xformers==0.0.16 --extra-index-url https://download.pytorch.org/whl/cu116

# Remove build dependencies
RUN apt-get purge -y --auto-remove build-essential cmake git ninja-build python3-dev libpthread-stubs0-dev

# Copy the rest of the application code
COPY . .

# Expose the port (optional)
EXPOSE 5000

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Run the application with Gunicorn, using a single worker
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "sync", "app:app"]
