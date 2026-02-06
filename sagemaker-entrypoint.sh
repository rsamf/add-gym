#!/bin/bash
set -e

echo "=== SageMaker Training Container Started ==="
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Arguments: $@"
echo "Environment variables:"
env | grep -E "(SM_|SAGEMAKER_)" | head -10 || echo "No SageMaker env vars found"

# Distributed training setup
# Check for resourceconfig.json which contains the cluster information
RESOURCE_CONFIG=${SM_RESOURCE_CONFIG:-"/opt/ml/input/config/resourceconfig.json"}

if [ -f "$RESOURCE_CONFIG" ]; then
    dist_info=$(python -c "
import json, os, sys
try:
    with open('$RESOURCE_CONFIG', 'r') as f:
        config = json.load(f)
    
    hosts = config.get('hosts', [])
    if not isinstance(hosts, list):
        print('0;;0;0')
        sys.exit(0)
        
    nnodes = len(hosts)
    if nnodes <= 1:
        print(f'0;;0;0')
        sys.exit(0)
        
    # SageMaker hosts are usually sorted, but let's trust the strict list order
    master_addr = hosts[0]
    current_host = config.get('current_host', os.environ.get('SM_CURRENT_HOST', ''))
    
    try:
        node_rank = hosts.index(current_host)
    except ValueError:
        node_rank = 0
        
    print(f'1;{master_addr};{nnodes};{node_rank}')
except Exception:
    print('0;;0;0')
")
else
    # Fallback or single instance if config doesn't exist
    dist_info="0;;0;0"
fi

IFS=';' read -r IS_DISTRIBUTED MASTER_ADDR NNODES NODE_RANK <<< "$dist_info"

# Auto-detect local multi-GPU to force DDP even on single instance
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
# Trim whitespace
GPUS_PER_NODE=$(echo $GPUS_PER_NODE | xargs)
IS_DISTRIBUTED=$(echo $IS_DISTRIBUTED | xargs)
IS_DISTRIBUTED=${IS_DISTRIBUTED:-0}

if [ "$IS_DISTRIBUTED" == "0" ] && [ "$GPUS_PER_NODE" -gt 1 ]; then
    echo "Detected single node with $GPUS_PER_NODE GPUs. Enabling local DDP."
    IS_DISTRIBUTED="1"
    MASTER_ADDR="localhost"
    NNODES=1
    NODE_RANK=0
fi

if [ "$IS_DISTRIBUTED" == "1" ]; then
    echo "Multi-instance training detected"
    
    MASTER_PORT=${SM_MASTER_PORT:-7777}
    
    echo "Distributed Config:"
    echo "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  NNodes: $NNODES"
    echo "  Node Rank: $NODE_RANK"
    echo "  Current Host: $SM_CURRENT_HOST"

    # Set environment variables for torchrun/DDP
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$NNODES  # Total number of nodes
    export NODE_RANK=$NODE_RANK
    export RANK=$NODE_RANK     # For multi-node, rank is usually the node rank
    
    # Use torchrun to launch the training script
    # nproc_per_node should match the number of GPUs per instance (1 for g4dn.xlarge)
    # We can detect this or assume 1 based on our known config
    GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
    echo "  GPUs per node: $GPUS_PER_NODE"

    # Ensure we have a valid Run ID. If SAGEMAKER_JOB_NAME is missing, torchrun generates random IDs
    # causing RendezvousTimeoutError (nodes waiting for different IDs).
    RDZV_ID=${SAGEMAKER_JOB_NAME:-"add-gym-distributed-job"}
    echo "  Rendezvous ID: $RDZV_ID"

    # Filter out "python", "-m", "add_gym.main" or "add_gym/main.py" from arguments if present
    # SageMaker might pass the entrypoint command as arguments
    ARGS=("$@")
    CLEAN_ARGS=()
    SKIP=0
    for arg in "${ARGS[@]}"; do
        if [ $SKIP -gt 0 ]; then
            SKIP=$((SKIP-1))
            continue
        fi
        case "$arg" in
            "python"|"python3")
                ;;
            "-m")
                SKIP=1 # Skip the next argument (module name)
                ;;
            "add_gym.main"|"add_gym/main.py")
                ;;
            *)
                CLEAN_ARGS+=("$arg")
                ;;
        esac
    done

    echo "=== Executing torchrun ==="
    
    # Force NCCL to use the correct network interface
    # export NCCL_SOCKET_IFNAME=eth0
    
    # Disable NCCL P2P to fix "Cuda failure 217 'peer access is not supported'"
    # This forces data transfer via Shared Memory (SHM) which is stable with device masking
    export NCCL_P2P_DISABLE=1
    # Disable NCCL SHM because Device Masking prevents CUDA IPC (peer access)
    export NCCL_SHM_DISABLE=1
    export NCCL_IB_DISABLE=1 # We are on ethernet
    
    # Disable Taichi Offline Cache to prevent file lock contention/corruption across processes
    export TI_OFFLINE_CACHE=0
    # Force NCCL to use loopback interface for stable local communication
    export NCCL_SOCKET_IFNAME=lo
    
    export NCCL_DEBUG=INFO

    exec torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=$RDZV_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        add_gym/main.py "${CLEAN_ARGS[@]}" \
        hydra.output_subdir=null \
        hydra.run.dir=.

else
    echo "Single-instance training"
    if [ $# -eq 0 ]; then
        echo "=== Starting default training ==="
        exec python -m add_gym.main
    else
        echo "=== Executing provided command ==="
        exec "$@"
    fi
fi