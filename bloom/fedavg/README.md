## Explaination
- This directory contains solutions for Federated Learning and Split Learning (tba).
- These are single node solutions to use for a HPC-architecture.

## Runtime instructions

# For using locally on machine using a conda environment
Requirements: A conda environment with libraries numpy, torch, torchvision installed.
INSTRUCTIONS: To deploy on a host machine, run "./start_locally.sh". Then specify number of clients to be used for training.
NOTE:
- The splitted datasets are located in the "datasets" folder.
- TODO: Federated learning parameters are currently changed manually through the server and client files. Implement a way to change these parameters through the congig file.

# For Singularity (OUTDATED)
- Build Singularity images by executing "./build_singularity".
- Boot up a singularity cluster to perform Federated Learning by executing "./start_with_singularity.sh".

# For Docker (OUTDATED)
- Build Docker images by executing (tba, maybe..)
- Boot up a docker cluster to perform Federated Learning by executing "./start_with_docker.sh".
- Stop and tear down the running docker containers by executing "./kill_docker_containers.sh"
