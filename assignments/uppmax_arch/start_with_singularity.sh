#!/bin/bash

echo "How many devices to use for training?"
read n_devices

if [ $n_devices -gt 0 ]
then
	# Start server from "start_server" script
	singularity instance start test_server/server.sif fl_server
	# Start all devices
	i=1
	while [ $i -le $n_devices ]
	do
		echo "Starting client$i"
		singularity instance start test_client/client.sif fl_client$i
		let "i+=1"
	done
else
	echo "Error: There must be at least one device!"
fi
