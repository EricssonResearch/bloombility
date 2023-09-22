# This script will stop and kill all active docker containers
docker stop $(docker ps -a -q) 
docker rm $(docker ps -a -q)
