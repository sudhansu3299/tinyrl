docker build -t cloud-agent-vm2 .

docker run -d \
  --cpus="0.5" \
  --memory="256m" \
  --network agent-net \
  --memory-reservation="128m" \
  --tmpfs /app/tmp:rw,size=50m \
  -e BACKEND_URL="http://backend:8000" \
  -e HOST_ID="vm-2" \
  --name cloud-agent-vm2 \
  cloud-agent-vm2

