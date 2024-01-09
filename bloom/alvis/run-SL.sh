#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-1181 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:2  # We're launching 2 nodes with 8 Nvidia T4 GPUs each
#SBATCH -t 0-00:10:00

repo_name="bloom-MJ"
ROOT_DIR="/mimer/NOBACKUP/groups/bloom/${repo_name}/bloom/bloom"

# ROOT_DIR="/Users/marion/Code/bloom/bloom"
DATASET_DIR="${ROOT_DIR}/load_data/datasets"
main_script="${ROOT_DIR}/split/main.py"
server_script="${ROOT_DIR}/split/server/server.py"
client_script="${ROOT_DIR}/split/worker/client.py"

# Check if the correct number of arguments is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <num_clients>"
    exit 1
fi

# Extract command-line arguments
num_clients=$1

echo "Number of clients: ${num_clients}"

echo "Running main.py"
/mimer/NOBACKUP/groups/bloom/bloom-MJ/bloom/venv/bin/python "${main_script}" --num-workers "${num_clients}" & pid1=$!
wait $pid1

# echo "Running server.py"
# python "${server_script}" -n "${num_clients}" -c "${config_file}" &
# sleep 3

# # Run client.py in a loop with num_clients
# for i in `seq 1 ${num_clients}`; do
#     echo "Running client.py $i"
#     python "${client_script}" -c "${config_file}" --train "${DATASET_DIR}/train_dataset${i}_${num_clients}.pth" --test "${DATASET_DIR}/test_dataset.pth" &
# done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
