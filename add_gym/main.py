import os
import subprocess
import os.path as osp
import hydra
import pickle
import torch
import torch._functorch.config
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from add_gym.learning.add.add_agent import ADDAgent


def _setup_training_env():
    """Setup common training environment settings."""
    # Enable TF32 for H100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _resolve_resume_path(path):
    """
    Resolve resume path, downloading from S3 if necessary.
    """
    if path is None:
        return None

    if path.startswith("s3://"):
        print(f"Downloading checkpoint from {path}...")
        local_dir = "/tmp/checkpoints"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(path))

        # Check if file already exists to avoid re-downloading (optional, but good for local/dev)
        # For now, let's aws cp it to be sure we have the latest or correct one.
        try:
            subprocess.check_call(["aws", "s3", "cp", path, local_path])
            print(f"Downloaded to {local_path}")
            return local_path
        except subprocess.CalledProcessError as e:
            print(f"Failed to download from S3: {e}")
            raise e

    return path


def _run_training(cfg: DictConfig, distributed: bool = False):
    """
    Run training loop (called by both single-instance and distributed training).

    Args:
        cfg: Hydra configuration
        distributed: Whether to run in distributed mode
    """
    exp_name = cfg["experiment_name"]

    is_main_process = True
    if distributed:
        is_main_process = dist.get_rank() == 0

    # Save Configurations (only on main process)
    if is_main_process:
        log_dir = Path(cfg.get("log_dir", "logs")) / f"{exp_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "cfgs.pkl", "wb") as f:
            pickle.dump(cfg, f)
        print(OmegaConf.to_yaml(cfg))
    else:
        log_dir = Path(cfg.get("log_dir", "logs")) / f"{exp_name}"

    # Create agent
    # We pass the distributed flag to the agent so it can wrap the model in DDP
    agent = ADDAgent(cfg, distributed=distributed)
    out_model_file = log_dir / "model.pt"
    log_file = log_dir / "log.txt"
    int_output_dir = log_dir / "intermediate_outputs"

    # Create directories (only on main process)
    if is_main_process:
        output_dir = osp.dirname(out_model_file)
        if output_dir != "" and (not osp.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)

        if int_output_dir != "" and (not osp.exists(int_output_dir)):
            os.makedirs(int_output_dir, exist_ok=True)

    # Synchronize all workers before loading checkpoints
    if distributed:
        dist.barrier()

    # Load checkpoint if exists (all workers load the same checkpoint)
    if osp.exists(out_model_file):
        # Used for Spot training resumption
        if is_main_process:
            print(
                f"Already found checkpoint in current log directory. Resuming training from {out_model_file}"
            )
        agent.load(out_model_file)
    else:
        resume_path = cfg.get("resume_path")
        resume_path = _resolve_resume_path(resume_path)
        if resume_path is not None:
            if is_main_process:
                print(f'Using "resume_path" and resuming training from {resume_path}')
            agent.load(resume_path)

    # Run training
    agent.train_model(
        out_model_file=out_model_file,
        int_output_dir=int_output_dir,
        log_file=log_file,
    )


def train_command(cfg: DictConfig):
    """
    Execute training with optional distributed training.

    Metrics Platform: TensorBoard

    Args:
        cfg: Hydra configuration
    """
    _setup_training_env()

    # Check if distributed training is enabled
    # Auto-detect if launched via torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not cfg.get("distributed_training", False):
            print(
                "[Distributed] Auto-detected torchrun environment. Enabling distributed training."
            )
        distributed_enabled = True
    else:
        distributed_enabled = cfg.get("distributed_training", False)

    if distributed_enabled:
        print("[Distributed] Distributed training enabled - using PyTorch DDP")

        # Initialize distributed process group
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Device Masking Pattern
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # Set environment variables for masking
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            os.environ["TI_VISIBLE_DEVICE"] = str(local_rank)
            os.environ["EGL_DEVICE_ID"] = str(local_rank)

            print(
                f"[Distributed] Device Masking Enabled: LOCAL_RANK={local_rank} mapped to Logical Device 0"
            )

            # Launched via torchrun
            # Must set device 0 BEFORE init_process_group when using specific binding with masking
            torch.cuda.set_device(0)

            torch.distributed.init_process_group(backend="nccl")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            print(
                f"[Distributed] Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}"
            )
        else:
            print(
                "[Distributed] Warning: Distributed enabled but not launched via torchrun. Running in single process mode."
            )
            distributed_enabled = False

    if distributed_enabled:
        # Run distributed training
        _run_training(cfg, distributed=True)

        # Cleanup
        torch.distributed.destroy_process_group()
    else:
        print("[Single Instance] Running single-instance training")
        # Run single-instance training
        _run_training(cfg, distributed=False)


@torch.inference_mode()
def test_command(cfg: DictConfig):
    agent = ADDAgent(cfg)

    resume_path = cfg.get("resume_path")
    resume_path = _resolve_resume_path(resume_path)
    if resume_path is not None:
        print(f'Using "resume_path" to test model {resume_path}')
        agent.load(resume_path)

    agent.test_model(100)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train_command(cfg)
    elif cfg.mode == "test":
        test_command(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Please choose 'train' or 'test'.")


if __name__ == "__main__":
    main()
