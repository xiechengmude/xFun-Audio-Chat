#!/bin/bash
# Deploy PDF-AI Control Center to a Remote Server
# Usage: ./deploy-to-server.sh user@host [--runpod-key KEY]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./deploy-to-server.sh user@host [--runpod-key KEY]"
    echo ""
    echo "Example:"
    echo "  ./deploy-to-server.sh root@192.168.1.100 --runpod-key your_api_key"
    echo ""
    echo "This script will:"
    echo "  1. Install Docker on the remote server (if needed)"
    echo "  2. Copy the control center files"
    echo "  3. Configure with your RunPod API key"
    echo "  4. Start the control center"
    exit 1
fi

REMOTE_HOST="$1"
RUNPOD_KEY=""

shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --runpod-key|-k)
            RUNPOD_KEY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Prompt for RunPod key if not provided
if [ -z "$RUNPOD_KEY" ]; then
    echo -n "Enter your RunPod API Key: "
    read -s RUNPOD_KEY
    echo ""
fi

if [ -z "$RUNPOD_KEY" ]; then
    echo -e "${RED}Error: RunPod API key is required.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Deploying to Remote Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Target: ${GREEN}$REMOTE_HOST${NC}"
echo ""

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" "echo 'SSH OK'" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to $REMOTE_HOST${NC}"
    exit 1
fi
echo -e "${GREEN}SSH connection OK${NC}"

# Check/Install Docker
echo ""
echo -e "${YELLOW}Checking Docker installation...${NC}"
ssh "$REMOTE_HOST" bash << 'DOCKER_CHECK'
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi
echo "Docker version: $(docker --version)"

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin || \
    curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose
fi
echo "Docker Compose: OK"
DOCKER_CHECK

# Create remote directory
echo ""
echo -e "${YELLOW}Setting up remote directory...${NC}"
REMOTE_DIR="/opt/pdf-ai-control-center"
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Copy files
echo ""
echo -e "${YELLOW}Copying files to remote server...${NC}"
rsync -avz --progress \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    "$SCRIPT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

# Configure .env
echo ""
echo -e "${YELLOW}Configuring environment...${NC}"
ssh "$REMOTE_HOST" bash << CONFIGURE
cd $REMOTE_DIR
cp .env.example .env
sed -i "s/your_runpod_api_key_here/$RUNPOD_KEY/" .env
echo "Configuration updated"
CONFIGURE

# Start the service
echo ""
echo -e "${YELLOW}Starting control center...${NC}"
ssh "$REMOTE_HOST" bash << START
cd $REMOTE_DIR
docker-compose down 2>/dev/null || true
docker-compose up -d --build
sleep 5

# Check health
for i in {1..10}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "Control center is ready!"
        break
    fi
    echo "Waiting... (\$i/10)"
    sleep 2
done
START

# Get server IP
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Control Center is running at:"
echo "  http://$REMOTE_HOST:8080"
echo ""
echo "API Endpoints:"
echo "  - Status:  http://$REMOTE_HOST:8080/api/status"
echo "  - Deploy:  curl -X POST http://$REMOTE_HOST:8080/api/deploy -H 'Content-Type: application/json' -d '{\"gpu_type\": \"A40\"}'"
echo "  - Pods:    http://$REMOTE_HOST:8080/api/pods"
echo "  - Health:  http://$REMOTE_HOST:8080/health"
echo ""
echo "To trigger a deployment:"
echo "  curl -X POST http://$REMOTE_HOST:8080/api/deploy \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"gpu_type\": \"A40\", \"run_benchmark\": true}'"
echo ""
