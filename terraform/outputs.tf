output "gym_ecr_repository_url" {
  description = "URL of the gym ECR repository"
  value       = aws_ecr_repository.gym.repository_url
}

output "base_ecr_repository_url" {
  description = "URL of the base ECR repository"
  value       = aws_ecr_repository.base.repository_url
}

output "runner_instance_id" {
  description = "Instance ID of the self-hosted GitHub Actions runner"
  value       = aws_instance.runner.id
}

output "runner_public_ip" {
  description = "Public IP of the runner (empty if in private subnet)"
  value       = aws_instance.runner.public_ip
}

output "runner_arn" {
  description = "ARN of the runner instance"
  value       = aws_instance.runner.arn
}