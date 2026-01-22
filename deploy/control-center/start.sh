#!/bin/bash
# PDF-AI Control Center Quick Start Script
# Usage: ./start.sh [--gpu GPU_TYPE] [--benchmark]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GPU_TYPE="A40"
RUN_BENCHMARK=false
ACTION="deploy"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)
            GPU_TYPE="$2"
            shift 2
            ;;
        --benchmark|-b)
            RUN_BENCHMARK=true
            shift
            ;;
        --status|-s)
            ACTION="status"
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --logs|-l)
            ACTION="logs"
            shift
            ;;
        --help|-h)
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu, -g GPU_TYPE    GPU type (H100, A100, A40, RTX4090). Default: A40"
            echo "  --benchmark, -b       Run performance benchmark after deployment"
            echo "  --status, -s          Show deployment status"
            echo "  --stop                Stop control center"
            echo "  --logs, -l            Show logs"
            echo "  --help, -h            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}No .env file found. Creating from .env.example...${NC}"
        cp .env.example .env
        echo -e "${RED}Please edit .env and set RUNPOD_API_KEY, then run again.${NC}"
        exit 1
    else
        echo -e "${RED}Error: No .env or .env.example file found.${NC}"
        exit 1
    fi
fi

# Check for RUNPOD_API_KEY
source .env
if [ -z "$RUNPOD_API_KEY" ] || [ "$RUNPOD_API_KEY" = "your_runpod_api_key_here" ]; then
    echo -e "${RED}Error: RUNPOD_API_KEY not set in .env file.${NC}"
    echo "Get your API key from: https://runpod.io/console/user/settings"
    exit 1
fi

case $ACTION in
    deploy)
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}  PDF-AI Control Center${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        echo -e "GPU Type: ${GREEN}$GPU_TYPE${NC}"
        echo -e "Benchmark: ${GREEN}$RUN_BENCHMARK${NC}"
        echo ""

        # Start docker compose
        echo -e "${YELLOW}Starting control center...${NC}"
        docker-compose up -d --build

        # Wait for service to be ready
        echo -e "${YELLOW}Waiting for service to be ready...${NC}"
        sleep 5

        # Check health
        for i in {1..10}; do
            if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                echo -e "${GREEN}Control center is ready!${NC}"
                break
            fi
            echo "  Waiting... ($i/10)"
            sleep 2
        done

        # Trigger deployment
        echo ""
        echo -e "${YELLOW}Triggering deployment to RunPod...${NC}"

        BENCHMARK_FLAG="false"
        if [ "$RUN_BENCHMARK" = true ]; then
            BENCHMARK_FLAG="true"
        fi

        RESPONSE=$(curl -s -X POST http://localhost:8080/api/deploy \
            -H "Content-Type: application/json" \
            -d "{\"gpu_type\": \"$GPU_TYPE\", \"run_benchmark\": $BENCHMARK_FLAG}")

        echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

        echo ""
        echo -e "${GREEN}Deployment triggered!${NC}"
        echo ""
        echo "Monitor progress:"
        echo "  - Status: curl http://localhost:8080/api/status"
        echo "  - Logs:   docker-compose logs -f"
        echo "  - UI:     http://localhost:8080/api/info"
        ;;

    status)
        echo -e "${BLUE}Checking deployment status...${NC}"
        curl -s http://localhost:8080/api/status | python3 -m json.tool
        ;;

    stop)
        echo -e "${YELLOW}Stopping control center...${NC}"
        docker-compose down
        echo -e "${GREEN}Stopped.${NC}"
        ;;

    logs)
        docker-compose logs -f
        ;;
esac
