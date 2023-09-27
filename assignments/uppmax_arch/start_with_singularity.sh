#!/bin/bash

# Brief: A Script which boots up a federated learning server and N clients using Singularity
# Author: Victor Hwasser
# TODO: The proper way to start is by using "singularity instance start .."
#       but this do not work for some reason, it starts an empty container,
#       but "singularity run .." works perfectly.

echo "How many devices to use for training?"
read n_devices

if [ $n_devices -gt 0 ]
then
	# Start server from "start_server" script
	echo "Starting server"
	# singularity instance start test_server/server.sif fl_server --command "python3 /app/server.py"
	singularity run test_server/server.sif &
	# Start all devices
	i=1
	while [ $i -le $n_devices ]
	do
		echo "Starting client$i"
		# singularity instance start test_client/client.sif fl_client$i --command "./test_client/entrypoint.sh" 
		singularity run --cpus 1 test_client/client.sif &
		let "i+=1"
	done
else
	echo "Error: There must be at least one device!"
fi
