#!/bin/bash

# Script which builds the singularity images for clients and servers
# Author: Victor Hwasser

# Change these if the directory for client and server changes
CLIENT_DIR=test_client
SERVER_DIR=test_server

cd $SERVER_DIR
singularity build server.sif SingFile.def
cd ../$CLIENT_DIR
singularity build client.sif SingFile.def

