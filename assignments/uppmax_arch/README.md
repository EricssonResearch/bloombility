## Explaination
- This directory contains solutions for Federated Learning and Split Learning (tba).
- These are single node solutions to use for a HPC-architecture.

## REQUIREMENT
- Singularity
- A venv environment named "venv" with pip libraries numpy, torch, torchvision installed.

## Runtime instructions
# For Singularity
- Build Singularity images by executing "./build_singularity".
- Boot up a singularity cluster to perform Federated Learning by executing "./start_with_singularity.sh".
# For Docker
- Build Docker images by executing (tba, maybe..)
- Boot up a docker cluster to perform Federated Learning by executing "./start_with_docker.sh".
- Stop and tear down the running docker containers by executing "./kill_docker_containers.sh"
