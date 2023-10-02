docker network create flower_network
docker run -d --name flower_server --net flower_network -p 8080:8080 fl_server
