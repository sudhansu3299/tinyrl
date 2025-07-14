docker network create agent-net

docker build -t backend .

docker run -d \
  --name backend \
  --network agent-net \
  -p 8000:8000 \
  backend
