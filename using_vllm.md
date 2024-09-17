# Comprehensive Guide: Setting Up and Using vLLM for OpenAI-Compatible Inference on AWS

## Table of Contents
1. Introduction
2. Setting Up an AWS EC2 Instance
3. Installing Docker and NVIDIA Docker
4. Pulling and Running the vLLM Docker Image
5. Verifying the Server
6. Using the OpenAI SDK for Inference
7. Advanced: Dynamically Switching Models
8. Troubleshooting and Best Practices

## 1. Introduction

vLLM is a fast and easy-to-use library for LLM inference and serving. This guide will walk you through setting up a vLLM-based server that's compatible with the OpenAI API, allowing you to run inferences on various language models using familiar tools and SDKs.

## 2. Setting Up an AWS EC2 Instance

### Step 1: Launch an EC2 Instance
1. Go to the AWS Management Console and navigate to EC2.
2. Click "Launch Instance".
3. Choose an appropriate instance type (e.g., g4dn.xlarge for GPU support).
4. Configure instance details, add storage, and set up security groups.
   - Ensure you allow inbound traffic on port 8000 (or your chosen port).

### Step 2: Connect to Your Instance
Use SSH to connect to your instance:

```bash
ssh -i your-key.pem ubuntu@your-instance-public-ip
```

## 3. Installing Docker and NVIDIA Docker

### Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

### Install NVIDIA Docker (for GPU support)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 4. Pulling and Running the vLLM Docker Image

### Pull the vLLM Docker Image
```bash
sudo docker pull vllm/vllm-openai:latest
```

### Run the Docker Container
```bash
sudo docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<your_huggingface_token>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

Replace `<your_huggingface_token>` with your actual Hugging Face token.

## 5. Verifying the Server

Check if the server is running correctly:

```bash
curl http://localhost:8000/v1/models
```

This should return a list of available models.

## 6. Using the OpenAI SDK for Inference

### Install the OpenAI Python package
```bash
pip install openai
```

### Python Script for Model Inference

Create a file named `vllm_inference.py` with the following content:

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

Replace `<your_ec2_public_ip>` with your EC2 instance's public IP address.

Run the script:
```bash
python vllm_inference.py
```

## 7. Advanced: Dynamically Switching Models

vLLM doesn't support hot-swapping of entire models. To switch models, you need to stop the current container and start a new one with the desired model. Here's a Python script to automate this process:

Create a file named `switch_model.py` with the following content:

```python
import os
import subprocess
import time
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://<your_ec2_public_ip>:8000/v1"

def stop_existing_container():
    result = subprocess.run(['sudo', 'docker', 'ps', '-q', '--filter', 'ancestor=vllm/vllm-openai:latest'], capture_output=True, text=True)
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
        'vllm/vllm-openai:latest',
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
start_new_container("llama2")  # Replace with the desired model name

# Wait for the new container to start
time.sleep(30)

# Get inference from the new model
new_result = get_model_inference(prompt, model="llama2")
print("New Model Inference Result:", new_result)
```

Replace `<your_ec2_public_ip>` with your EC2 instance's public IP address.

To run this script:
```bash
python switch_model.py
```

## 8. Troubleshooting and Best Practices

1. **GPU Memory**: Ensure your EC2 instance has enough GPU memory for the model you're trying to run.

2. **Docker Permissions**: If you encounter permission issues, you may need to add your user to the docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```
   Log out and back in for changes to take effect.

3. **Model Downloads**: The first time you run a model, it may take some time to download. Ensure you have a stable internet connection.

4. **API Compatibility**: While vLLM aims to be OpenAI API compatible, there might be some differences. Refer to the vLLM documentation for specifics.

5. **Security**: Always use proper security practices. Don't expose your server directly to the internet without proper authentication and encryption.

6. **Monitoring**: Set up monitoring for your EC2 instance to track resource usage and uptime.

7. **Updating vLLM**: Regularly check for updates to the vLLM Docker image and update when new versions are available:
   ```bash
   sudo docker pull vllm/vllm-openai:latest
   ```

8. **Custom Models**: If you're using custom models, ensure they're compatible with vLLM and properly loaded into the Hugging Face model hub or accessible from your server.

By following this guide, you should now have a functional vLLM-based OpenAI-compatible inference server running on AWS. Whether you're a beginner or an experienced user, you can use this setup to run inferences on various language models using the familiar OpenAI SDK interface.