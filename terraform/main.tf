terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Create ECR repository
resource "aws_ecr_repository" "gym" {
  name                 = var.gym_ecr_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name    = var.gym_ecr_name
    Project = var.project_name
  }
}

# Create ECR repository
resource "aws_ecr_repository" "base" {
  name                 = var.base_ecr_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name    = var.base_ecr_name
    Project = var.project_name
  }
}

# -------------------------------------------------------
# Self-Hosted GitHub Actions Runner
# -------------------------------------------------------

# Look up the latest Ubuntu 22.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# SSH key pair for debugging the runner instance
resource "aws_key_pair" "runner" {
  key_name   = "${var.project_name}-runner-key"
  public_key = var.runner_ssh_public_key

  tags = {
    Name    = "${var.project_name}-runner-key"
    Project = var.project_name
  }
}

# Security group: allow SSH inbound + all outbound
resource "aws_security_group" "runner" {
  name        = "${var.project_name}-runner-sg"
  description = "Security group for GitHub Actions self-hosted runner"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-runner-sg"
    Project = var.project_name
  }
}

# IAM role for the runner EC2 instance (ECR access, SageMaker, etc.)
resource "aws_iam_role" "runner" {
  name = "${var.project_name}-runner-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Name    = "${var.project_name}-runner-role"
    Project = var.project_name
  }
}

resource "aws_iam_role_policy" "runner_policy" {
  name = "${var.project_name}-runner-policy"
  role = aws_iam_role.runner.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:AddTags",
          "iam:PassRole",
          "logs:*"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "runner" {
  name = "${var.project_name}-runner-profile"
  role = aws_iam_role.runner.name
}

# EC2 instance running the self-hosted GitHub Actions runner
resource "aws_instance" "runner" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.runner_instance_type
  key_name               = aws_key_pair.runner.key_name
  iam_instance_profile   = aws_iam_instance_profile.runner.name
  vpc_security_group_ids = [aws_security_group.runner.id]

  root_block_device {
    volume_size = var.runner_volume_size_gb
    volume_type = "gp3"
  }

  user_data = base64encode(templatefile("${path.module}/runner-user-data.sh.tpl", {
    github_repo  = var.github_repo
    github_pat   = var.github_pat
    runner_name  = "${var.project_name}-runner"
    runner_labels = var.runner_labels
  }))

  tags = {
    Name    = "${var.project_name}-runner"
    Project = var.project_name
  }
}
