#!/bin/bash

echo "How many devices to use for training?"
read n_devices

if [ $n_devices -gt 0 ]
then
	# Start server from "start_server" script
	./test_server/start_server.sh
	# Start all devices
	i=1
	while [ $i -le $n_devices ]
	do
		echo "Starting client$i"
		docker run -d --name client$i --net flower_network fl_client
		let "i+=1"
	done
else
	echo "Error: There must be at least one device!"
fi
