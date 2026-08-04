# Startup Guide (Intel XPU — Docker)

Steps used to build and run Physical AI Studio on this machine (Intel Battlemage G31 XPU).

## Prerequisites

- Docker Engine 24.0+ with Docker Compose v2.24.0+
- Intel XPU hardware (run `lspci | grep -i vga` to confirm)

## 1. Fix Docker file-descriptor limit

BuildKit containers default to 1024 file descriptors, which is too low for `uv sync`
to bytecode-compile PyTorch's `torchgen`. Raise it before building.

```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
EOF

sudo systemctl restart docker
```

Verify the fix:

```bash
docker run --rm busybox sh -c 'ulimit -n'
# should print 65536
```

## 2. Detect host device-group GIDs

```bash
cd application/docker
./setup-devices.sh --xpu
```

This writes `application/docker/.env` with `COMPOSE_PROFILES=xpu` and the
correct GIDs for `video`, `dialout`, `plugdev`, and `render` groups.

## 3. Build the Docker image

The standard `docker compose build` does not forward `--ulimit` to BuildKit,
so build manually with the flag:

```bash
docker build \
  --ulimit nofile=65536:65536 \
  --build-arg APP_UID=1000 \
  --build-arg APP_GID=1000 \
  --target physical-ai-studio-xpu \
  -f application/docker/Dockerfile \
  --build-context libs=library \
  --build-context plugin=application/plugin \
  -t ghcr.io/open-edge-platform/physical-ai-studio-xpu:main \
  .
```

Run this from the repository root.

## 4. Start the application

```bash
cd application/docker
docker compose --profile xpu up -d
```

## 5. Verify

```bash
curl http://localhost:7860/api/health
# {"status":"healthy"}
```

Open **http://localhost:7860** in a browser to access the UI.

## Stopping

```bash
cd application/docker
docker compose --profile xpu down
```

