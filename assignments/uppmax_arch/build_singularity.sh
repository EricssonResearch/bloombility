#!/bin/bash

# Script which builds the singularity images for clients and servers
# Author: Victor Hwasser

# Change these if the directory for client and server changes
CLIENT_DIR=test_client
SERVER_DIR=test_server

singularity build $CLIENT_DIR/client.sif $CLIENT_DIR/SingFile.def
singularity build $SERVER_DIR/server.sif $SERVER_DIR/SingFile.def

