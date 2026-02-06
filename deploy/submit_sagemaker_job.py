#!/usr/bin/env python3
"""
Submit SageMaker training job with Hydra configuration passed directly as arguments.
This script eliminates the need to upload configs to S3 by converting the Hydra
configuration to command-line overrides.
"""

import argparse
import boto3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import yaml


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> List[str]:
    """
    Flatten nested dictionary to Hydra override format.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys

    Returns:
        List of Hydra override strings in format "key.subkey=value"
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Handle lists by converting to Hydra list syntax [item1,item2]
            # Convert items to strings and join with commas
            # We assume items don't contain commas or brackets for now
            list_items = [str(item) for item in v]
            list_str = ",".join(list_items)
            items.append(f"{new_key}=[{list_str}]")
        elif isinstance(v, str):
            # Quote strings that contain spaces or slashes
            if " " in v or "/" in v:
                items.append(f"{new_key}='{v}'")
            else:
                items.append(f"{new_key}={v}")
        else:
            items.append(f"{new_key}={v}")

    return items


def yaml_to_hydra_overrides(yaml_path: Path) -> List[str]:
    """
    Convert YAML config file to list of Hydra command-line overrides.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        List of Hydra override strings
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    overrides = []

    # Handle 'defaults' separately as it uses special Hydra syntax
    if "defaults" in config:
        defaults = config.pop("defaults")
        # Convert defaults to Hydra config group syntax
        # Format: +group_name=config_name or just config_name
        for item in defaults:
            if isinstance(item, dict):
                # Format: {group_name: config_name}
                for group, cfg_name in item.items():
                    if cfg_name is not None:
                        overrides.append(f"+{group}={cfg_name}")
            elif isinstance(item, str):
                # Special items like '_self_' are skipped
                if item != "_self_":
                    overrides.append(item)

    # Handle 'hydra' section separately as it's for Hydra runtime config
    if "hydra" in config:
        hydra_config = config.pop("hydra")
        # Keep hydra config for flattening
        overrides.extend(flatten_dict({"hydra": hydra_config}))

    # Flatten the rest of the config
    overrides.extend(flatten_dict(config))

    return overrides


def get_aws_account_id() -> str:
    """Get AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def submit_training_job(
    job_config_path: Path,
    hydra_config_path: Path,
    append_timestamp: bool = True,
    override_job_name: str = None,
    override_aws_region: str = None,
    override_s3_bucket: str = None,
    override_image_uri: str = None,
) -> str:
    """
    Submit SageMaker training job with Hydra configuration.

    Args:
        job_config_path: Path to SageMaker job configuration YAML
        hydra_config_path: Path to Hydra overrides YAML
        append_timestamp: Whether to append timestamp to job name

    Returns:
        Training job name
    """
    # Load SageMaker job configuration
    with open(job_config_path, "r") as f:
        job_config = yaml.safe_load(f)

    # Extract configuration values
    job_name = str(override_job_name if override_job_name else job_config["job"]["name"])
    instance_type = job_config["job"]["instance_type"]
    instance_count = job_config["job"].get("instance_count", 1)
    volume_size = job_config["job"]["volume_size"]
    max_runtime = job_config["job"]["max_runtime"]

    aws_region = override_aws_region
    s3_bucket = override_s3_bucket
    s3_prefix = job_config["aws"]["s3_prefix"]
    spot_enabled = job_config["aws"].get("spot_enabled", False)

    tags = job_config.get("tags", {})

    # Check if distributed training is enabled
    distributed_enabled = instance_count > 1

    # Append timestamp if requested
    if append_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{job_name}-{timestamp}"

    # Get AWS account ID and construct image URI
    account_id = get_aws_account_id()
    image_uri = override_image_uri

    # Convert Hydra config to command-line overrides
    hydra_overrides = yaml_to_hydra_overrides(hydra_config_path)

    # Build container arguments
    # Pass Hydra overrides directly — the entrypoint/CMD in the Dockerfile handles execution
    container_arguments = []

    # Add all Hydra overrides
    for override in hydra_overrides:
        container_arguments.append(override)

    # Explicitly override experiment_name to match the SageMaker job name
    # This ensures logs are stored in a unique directory (e.g. s3://.../checkpoints/<commit_hash>/)
    container_arguments.append(f"experiment_name={job_name}")

    print(f"Creating SageMaker training job: {job_name}")
    print(f"Instance type: {instance_type}")
    print(f"Instance count: {instance_count}")
    print(f"Distributed training: {'Enabled' if distributed_enabled else 'Disabled'}")
    print(f"Training image: {image_uri}")
    print(f"S3 output: s3://{s3_bucket}/{s3_prefix}/output")
    print(f"\nHydra overrides ({len(hydra_overrides)}):")
    for override in hydra_overrides[:10]:  # Show first 10
        print(f"  {override}")
    if len(hydra_overrides) > 10:
        print(f"  ... and {len(hydra_overrides) - 10} more")

    # Create SageMaker client
    sagemaker = boto3.client("sagemaker", region_name=aws_region)

    # Prepare tags
    tag_list = [{"Key": k, "Value": str(v)} for k, v in tags.items()]

    stopping_condition = {
        "MaxRuntimeInSeconds": max_runtime,
    }
    if spot_enabled:
        stopping_condition["MaxWaitTimeInSeconds"] = max_runtime + 24 * 3600

    # Create training job
    response = sagemaker.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerArguments": container_arguments,
        },
        RoleArn=f"arn:aws:iam::{account_id}:role/sagemaker_execution_role",
        InputDataConfig=[
            {
                "ChannelName": "assets",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{s3_bucket}/robots/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/octet-stream",
                "CompressionType": "None",
            }
        ],
        ResourceConfig={
            "InstanceCount": instance_count,
            "InstanceType": instance_type,
            "VolumeSizeInGB": volume_size,
        },
        StoppingCondition=stopping_condition,
        OutputDataConfig={"S3OutputPath": f"s3://{s3_bucket}/{s3_prefix}/output"},
        CheckpointConfig={
            "S3Uri": f"s3://{s3_bucket}/{s3_prefix}/checkpoints/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=False,
        EnableManagedSpotTraining=spot_enabled,
        Tags=tag_list,
    )

    print(f"\n✓ SageMaker training job '{job_name}' created successfully")
    print(f"\nTo monitor the training job, run:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
    print(f"\nTo stream CloudWatch logs, run:")
    print(
        f"  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow"
    )

    return job_name


def main():
    parser = argparse.ArgumentParser(
        description="Submit SageMaker training job with Hydra configuration"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).parent / "sagemaker-job-config.yaml",
        help="Path to SageMaker job configuration YAML (default: ./sagemaker-job-config.yaml)",
    )
    parser.add_argument(
        "-hc",
        "--hydra-config",
        type=Path,
        default=Path(__file__).parent / "train-config.yaml",
        help="Path to Hydra configuration YAML (default: ./train-config.yaml)",
    )
    parser.add_argument(
        "-n",
        "--no-timestamp",
        action="store_true",
        help="Don't append timestamp to job name",
    )
    parser.add_argument(
        "--image-tag", type=str, help="Override image tag (default: from config)"
    )
    parser.add_argument(
        "--job-name", type=str, help="Override job name (default: from config)"
    )
    parser.add_argument(
        "--aws-region", type=str, help="Override AWS region (default: from config)"
    )
    parser.add_argument(
        "--s3-bucket", type=str, help="Override S3 bucket name (default: from config)"
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        help="Override full ECR image URI (default: constructed from account ID, region, repository, and tag)",
    )

    args = parser.parse_args()

    # Validate files exist
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        return 1

    if not args.hydra_config.exists():
        print(f"ERROR: Hydra config file not found: {args.hydra_config}")
        return 1

    try:
        submit_training_job(
            job_config_path=args.config,
            hydra_config_path=args.hydra_config,
            append_timestamp=not args.no_timestamp and not args.job_name,
            override_job_name=args.job_name,
            override_aws_region=args.aws_region,
            override_s3_bucket=args.s3_bucket,
            override_image_uri=args.image_uri,
        )
        return 0
    except Exception as e:
        print(f"ERROR: Failed to create training job: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
