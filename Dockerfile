# Use the official PyTorch image with CUDA 11.7 and PyTorch 2.0
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt ./

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY . .

# Expose the port (optional)
EXPOSE 5000

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Run the application with Gunicorn, using a single worker
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "sync", "app:app"]
