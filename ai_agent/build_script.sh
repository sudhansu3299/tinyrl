docker build -t ai-agent .

docker run -d \
  --name ai-agent \
  --network agent-net \
  -e BACKEND_URL=http://backend:8000 \
  ai-agent
