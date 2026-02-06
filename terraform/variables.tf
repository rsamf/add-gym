variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
}

variable "project_name" {
  description = "Name of the project, used for resource naming"
  type        = string
  default     = "add-gym"
}

variable "github_repo" {
  description = "The GitHub repository to allow OIDC access (format: user/repo)"
  type        = string
}

variable "gym_ecr_name" {
    description = "Name of the ECR repository"
    type        = string
    default     = "add-gym"
}

variable "base_ecr_name" {
    description = "Name of the ECR repository"
    type        = string
    default     = "training-base"
}

# -------------------------------------------------------
# Self-Hosted Runner
# -------------------------------------------------------

variable "github_pat" {
  description = "GitHub Personal Access Token with repo scope, used to register the self-hosted runner"
  type        = string
  sensitive   = true
}

variable "runner_instance_type" {
  description = "EC2 instance type for the GitHub Actions runner"
  type        = string
  default     = "t3.large"
}

variable "runner_volume_size_gb" {
  description = "Root EBS volume size in GB for the runner (needs space for Docker builds)"
  type        = number
  default     = 50
}

variable "runner_labels" {
  description = "Comma-separated labels applied to the self-hosted runner"
  type        = string
  default     = "self-hosted,linux,x64"
}

variable "runner_ssh_public_key" {
  description = "SSH public key to authorize on the runner instance (contents of ~/.ssh/id_rsa.pub or similar)"
  type        = string
}