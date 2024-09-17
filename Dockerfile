# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
RUN conda create -n vllm python=3.10 -y

# Activate conda environment
SHELL ["conda", "run", "-n", "vllm", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio

# Install vLLM
RUN pip install vllm

# Install OpenAI Python client
RUN pip install openai

# Set working directory
WORKDIR /app

# Copy your application code (if any)
COPY . /app

# Set the default command to run when the container starts
CMD ["conda", "run", "--no-capture-output", "-n", "vllm", "python", "your_script.py"]

# Create a CPU-only version
FROM base AS cpu
RUN conda install -c pytorch cpuonly -y