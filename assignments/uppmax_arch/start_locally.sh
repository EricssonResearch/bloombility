#!/bin/bash

# Brief: A Script which boots up a federated learning locally
# Author: Victor Hwasser
#
# NOTE: This is for testing on your personal PC only! It will not work on UPPMAX!

echo "How many devices to use for training?"
read n_devices

if [ $n_devices -gt 0 ]
then
	# Start data distributer
	echo "Starting data distributer"
	python3 data_distributer/data_dist.py $n_devices
	# Start server from "start_server" script
	echo "Starting server"
	# singularity instance start test_server/server.sif fl_server --command "python3 /app/server.py"
	python3 test_server/app/server.py &
	# Start all devices
	i=1
	while [ $i -le $n_devices ]
	do
		echo "Starting client$i"
		python3 test_client/app/client.py datasets/train_dataset${i}_${n_devices}.pt datasets/test_dataset.pt &
		let "i+=1"
	done
else
	echo "Error: There must be at least one device!"
fi
