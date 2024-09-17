# Comprehensive Guide: Setting Up vLLM on AWS using WinSCP and PuTTY

## Table of Contents
1. Introduction
2. Setting Up an AWS EC2 Instance
3. Connecting to Your EC2 Instance
   - Using PuTTY for SSH
   - Using WinSCP for File Transfer
4. Installing Docker and NVIDIA Docker
5. Creating and Uploading the Dockerfile
6. Building and Running the vLLM Docker Image
7. Verifying the Server
8. Using the OpenAI SDK for Inference
9. Advanced: Dynamically Switching Models
10. Troubleshooting and Best Practices

## 1. Introduction

This guide will walk you through the process of setting up a vLLM-based server on AWS that's compatible with the OpenAI API. We'll be using PuTTY for SSH connections and WinSCP for file transfers, making this guide particularly useful for Windows users.

## 2. Setting Up an AWS EC2 Instance

1. Log in to the AWS Management Console and navigate to EC2.
2. Click "Launch Instance".
3. Choose an appropriate instance type (e.g., g4dn.xlarge for GPU support).
4. Configure instance details, add storage, and set up security groups.
   - Ensure you allow inbound traffic on port 22 (for SSH) and port 8000 (for the vLLM server).
5. Create or select an existing key pair. Download the .pem file if you're creating a new one.

## 3. Connecting to Your EC2 Instance

### Using PuTTY for SSH

1. Download and install PuTTY if you haven't already.
2. Convert your .pem file to .ppk format:
   - Open PuTTYgen
   - Click "Load" and select your .pem file
   - Click "Save private key" to create a .ppk file
3. Open PuTTY:
   - In the "Host Name" field, enter: ubuntu@your-instance-public-ip
   - Navigate to Connection > SSH > Auth
   - Browse and select your .ppk file
   - Click "Open" to start the SSH session

### Using WinSCP for File Transfer

1. Download and install WinSCP if you haven't already.
2. Create a new session in WinSCP:
   - File protocol: SFTP
   - Host name: your-instance-public-ip
   - User name: ubuntu
   - In "Advanced" > "SSH" > "Authentication", browse and select your .ppk file
3. Click "Login" to connect to your EC2 instance

## 4. Installing Docker and NVIDIA Docker

Connect to your EC2 instance using PuTTY and run the following commands:

```bash
# Update package list
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 5. Creating and Uploading the Dockerfile

1. On your local machine, create a file named `Dockerfile` with the following content:

```dockerfile
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
```

2. Use WinSCP to upload this `Dockerfile` to your EC2 instance. You can place it in the `/home/ubuntu` directory.

## 6. Building and Running the vLLM Docker Image

Connect to your EC2 instance using PuTTY and run the following commands:

```bash
# Navigate to the directory containing the Dockerfile
cd /home/ubuntu

# Build the Docker image
sudo docker build -t vllm-server .

# Run the Docker container
sudo docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<your_huggingface_token>" \
    -p 8000:8000 \
    --ipc=host \
    vllm-server \
    conda run --no-capture-output -n vllm python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-v0.1
```

Replace `<your_huggingface_token>` with your actual Hugging Face token.

## 7. Verifying the Server

In your PuTTY session, run:

```bash
curl http://localhost:8000/v1/models
```

This should return a list of available models.

## 8. Using the OpenAI SDK for Inference

1. On your local machine, create a file named `vllm_inference.py` with the following content:

```python
import openai

# Set OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://<your_ec2_public_ip>:8000/v1"

def get_model_inference(prompt, model="mistralai/Mistral-7B-v0.1"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text

# Example usage
prompt = "Once upon a time"
result = get_model_inference(prompt)
print("Model Inference Result:", result)
```

2. Use WinSCP to upload this `vllm_inference.py` file to your EC2 instance.

3. In your PuTTY session, run:

```bash
python vllm_inference.py
```

## 9. Advanced: Dynamically Switching Models

1. On your local machine, create a file named `switch_model.py` with the following content:

```python
import os
import subprocess
import time
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://<your_ec2_public_ip>:8000/v1"

def stop_existing_container():
    result = subprocess.run(['sudo', 'docker', 'ps', '-q', '--filter', 'ancestor=vllm-server'], capture_output=True, text=True)
    container_id = result.stdout.strip()
    if container_id:
        subprocess.run(['sudo', 'docker', 'stop', container_id])
        print(f"Stopped container: {container_id}")
    else:
        print("No running vLLM container found.")

def start_new_container(model_name):
    command = [
        'sudo', 'docker', 'run', '--runtime', 'nvidia', '--gpus', 'all',
        '-v', os.path.expanduser('~/.cache/huggingface:/root/.cache/huggingface'),
        '--env', f"HUGGING_FACE_HUB_TOKEN={os.getenv('HUGGING_FACE_HUB_TOKEN')}",
        '-p', '8000:8000',
        '--ipc=host',
        'vllm-server',
        'conda', 'run', '--no-capture-output', '-n', 'vllm', 'python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model_name
    ]
    subprocess.run(command)
    print(f"Started new container with model: {model_name}")

def get_model_inference(prompt, model="mistralai/Mistral-7B-v0.1"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text

# Example usage
prompt = "Once upon a time"
result = get_model_inference(prompt)
print("Model Inference Result:", result)

# Unload the current model and load a new model
stop_existing_container()
time.sleep(5)  # Wait for the container to stop
start_new_container("meta-llama/Llama-2-7b-chat-hf")  # Replace with the desired model name

# Wait for the new container to start
time.sleep(30)

# Get inference from the new model
new_result = get_model_inference(prompt, model="meta-llama/Llama-2-7b-chat-hf")
print("New Model Inference Result:", new_result)
```

2. Use WinSCP to upload this `switch_model.py` file to your EC2 instance.

3. In your PuTTY session, run:

```bash
python switch_model.py
```

## 10. Troubleshooting and Best Practices

1. **File Permissions**: If you encounter permission issues, you may need to change file permissions. In PuTTY, use:
   ```bash
   chmod +x your_script.py
   ```

2. **Docker Permissions**: If you get a "permission denied" error when running Docker commands, add your user to the docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```
   Then, log out and log back in for the changes to take effect.

3. **Firewall Issues**: Ensure that your EC2 security group allows inbound traffic on port 8000.

4. **Model Downloads**: The first time you run a model, it may take some time to download. Ensure you have a stable internet connection.

5. **Monitoring**: Use the AWS CloudWatch service to monitor your EC2 instance's resource usage.

6. **Updates**: Regularly update your Docker image and vLLM installation to get the latest features and security updates.

7. **Security**: Never expose your Hugging Face token or other sensitive information in your code or Dockerfile. Use environment variables or AWS Secrets Manager for sensitive data.

8. **Backups**: Regularly backup your code and any important data using WinSCP.

By following this guide, you should now have a functional vLLM-based OpenAI-compatible inference server running on AWS, which you can manage using PuTTY and WinSCP. Whether you're a beginner or an experienced user, you can use this setup to run inferences on various language models using the familiar OpenAI SDK interface.