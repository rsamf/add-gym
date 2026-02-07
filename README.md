<p align="center">
  <h1 align="center">ADD Gym</h1>

  <p align="center">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.10-blue">
    <img alt="Manager" src="https://img.shields.io/badge/managed%20by-uv-purple">
  </p>

  <p align="center">
    An ML training repo for imitation training using the ADD learning robot, G1 robot, and physics simulators: Genesis and Mujoco Warp.
    <br>
  </p>

  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#deployment">Deployment</a> •
    <a href="#sagemaker">SageMaker Training</a>
  </p>
</p>

## Overview

This is my (@rsamf) project that I work on for humanoid robotics imitation training using the ADD algorithm. It is designed to be scalable, supporting both local development cycles and large-scale distributed training on AWS SageMaker. I have made this repo public to serve inspiration to others or create a starting point for myself and others to train other RL policies. If you happen to use this repo and find any issues, feel free to make an issue. I will address it ASAP. If you find this repo useful, make sure to leave a ⭐. It helps a lot.

I will also make model weights public once I get some good results :) 

## Getting Started

### Prerequisites

- **Python 3.10** is required.
- **uv**: We use `uv` for dependency management.

### Installation

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync dependencies**:
    ```bash
    uv sync
    ```
    This will create a virtual environment and install all necessary dependencies, including `genesis-world`.

## Local Development

To run the training loop locally:

```bash
uv run python -m add-gym.main
```

You can customize the run using Hydra configuration overrides:

```bash
uv run python -m add-gym.main engine.num_envs=4096
```

### Mujoco Warp

> [!CAUTION]
> Mujoco Warp integration is currently under development and is not yet rigorously tested!

To use Mujoco Warp, you'll first need to install the extra dependencies:

```bash
uv sync --all-extras
```
And then, simply change the engine config group to mjwarp. Example:

```bash
uv run python -m add-gym.main engine=mjwarp
```

## Deployment

### CI/CD Pipeline

The project includes GitHub Actions workflows that automatically:
* build and push the Docker image suitable for training
* optionally, tag to push your trained model weights to Hugging Face

**Crucial Setup:**
To ensure the build pipeline works, you must set the `BASE_IMAGE` secret in your GitHub repository settings.
-   **Value**: This should point to a valid Genesis or MjWarp base image (e.g., `account_id.dkr.ecr.region.amazonaws.com/genesis:latest`).
-   The workflow checks out the code, builds the training image using this base, and pushes it to your configured ECR repository.

In addition to `BASE_IMAGE`, you also need to set the following secrets:
* `AWS_REGION`
* `S3_BUCKET`
* `HF_TOKEN` (optional, if you want to push your trained model to Hugging Face)

## SageMaker Training

I provide a specialized workflow for submitting training jobs to SageMaker without manually handling config files. The submission script converts your local Hydra configuration into command-line arguments for the remote job. This requires Cloud deployable resources, which are specified in terraform.

### 1. Configure

* Update `deploy/sagemaker-job-config.yaml` with job specs.
* Update `deploy/train-config.yaml` with training overrides.

### 2. Customization

Update your terraform config in [terraform/terraform.tfvars](terraform/terraform.tfvars)

You can override job parameters in the Hydra config.

* For cloud-executed training runs, you can override it in [deploy/train-config](deploy/train-config).
* You can also specify specific parameters the hydra files in [add-gym/configs](add-gym/configs).


### 3. Train

Any commit pushed to a branch named models or models/* will trigger a job to build and Sagemaker job.
