#!/bin/bash

# AstroScale - Complete System Startup Script
# This script starts all components of the AstroScale platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8080
FRONTEND_PORT=5173
CORE_PORT=8081
METRICS_PORT=9090

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Print banner
echo -e "${CYAN}"
cat << 'EOF'
   ___        __            _____            __
  / _ | ___ _/ /________   / ____|______ _ / /__
 / __ |/ _ | / __/ __/ _ \/\ \| |/ / __ | / /_ \
/_/ |_|\___/ \__/_/  \___/___/_/\_\_/ |_/_/\___|

    Stellar Redshift Prediction Platform
EOF
echo -e "${NC}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🚀 AstroScale System Startup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

MISSING_DEPS=0

if ! command_exists node; then
    echo -e "${RED}  ✗ Node.js not found${NC}"
    MISSING_DEPS=1
else
    echo -e "${GREEN}  ✓ Node.js found: $(node --version)${NC}"
fi

if ! command_exists dune; then
    echo -e "${RED}  ✗ Dune (OCaml build tool) not found${NC}"
    MISSING_DEPS=1
else
    echo -e "${GREEN}  ✓ Dune found: $(dune --version)${NC}"
fi

if ! command_exists python3; then
    echo -e "${RED}  ✗ Python3 not found${NC}"
    MISSING_DEPS=1
else
    echo -e "${GREEN}  ✓ Python3 found: $(python3 --version)${NC}"
fi

if [ ! -f "$HOME/.virtualenvs/py3.11/bin/python" ]; then
    echo -e "${YELLOW}  ⚠ Virtual environment not found at ~/.virtualenvs/py3.11${NC}"
    echo -e "${YELLOW}    Using system Python instead${NC}"
fi

if ! command_exists cargo; then
    echo -e "${YELLOW}  ⚠ Rust/Cargo not found (Core engine will not be available)${NC}"
else
    echo -e "${GREEN}  ✓ Cargo found: $(cargo --version | head -1)${NC}"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}✗ Missing required dependencies. Please install them first.${NC}"
    exit 1
fi

echo ""

# Check if ports are available
echo -e "${YELLOW}🔍 Checking ports...${NC}"

if check_port $BACKEND_PORT; then
    echo -e "${RED}  ✗ Port $BACKEND_PORT (Backend) is already in use${NC}"
    echo -e "${YELLOW}    Kill the process with: lsof -ti:$BACKEND_PORT | xargs kill -9${NC}"
    exit 1
else
    echo -e "${GREEN}  ✓ Port $BACKEND_PORT (Backend) available${NC}"
fi

if check_port $FRONTEND_PORT; then
    echo -e "${RED}  ✗ Port $FRONTEND_PORT (Frontend) is already in use${NC}"
    exit 1
else
    echo -e "${GREEN}  ✓ Port $FRONTEND_PORT (Frontend) available${NC}"
fi

echo ""

# Build backend if needed
echo -e "${YELLOW}🔨 Building backend...${NC}"
cd backend
if [ ! -d "_build" ] || [ ! -f "_build/default/bin/main.exe" ]; then
    echo -e "${CYAN}  Building OCaml backend...${NC}"
    dune build 2>&1 | sed 's/^/    /'
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Backend built successfully${NC}"
    else
        echo -e "${RED}  ✗ Backend build failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  ✓ Backend already built${NC}"
fi
cd ..

# Install frontend dependencies if needed
echo ""
echo -e "${YELLOW}📦 Checking frontend dependencies...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo -e "${CYAN}  Installing npm packages...${NC}"
    npm install 2>&1 | sed 's/^/    /'
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Dependencies installed${NC}"
    else
        echo -e "${RED}  ✗ npm install failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  ✓ Dependencies already installed${NC}"
fi
cd ..

# Build core engine if Rust is available
if command_exists cargo; then
    echo ""
    echo -e "${YELLOW}⚙️  Building core engine (optional)...${NC}"
    cd core
    if [ ! -f "target/release/astroscale-node" ]; then
        echo -e "${CYAN}  Building Rust core engine...${NC}"
        cargo build --release 2>&1 | grep -E "(Compiling|Finished)" | sed 's/^/    /'
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓ Core engine built successfully${NC}"
        else
            echo -e "${YELLOW}  ⚠ Core engine build had issues (non-critical)${NC}"
        fi
    else
        echo -e "${GREEN}  ✓ Core engine already built${NC}"
    fi
    cd ..
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ All components ready!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create log directory
mkdir -p logs

# Start backend
echo -e "${PURPLE}🚀 Starting Backend (OCaml)...${NC}"
cd backend
dune exec backend > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}  ✓ Backend started (PID: $BACKEND_PID)${NC}"
echo -e "${CYAN}    URL: http://localhost:$BACKEND_PORT${NC}"
echo -e "${CYAN}    Logs: logs/backend.log${NC}"
cd ..

# Wait for backend to be ready
echo -e "${YELLOW}  Waiting for backend to start...${NC}"
for i in {1..30}; do
    if check_port $BACKEND_PORT; then
        echo -e "${GREEN}  ✓ Backend is ready!${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}  ✗ Backend failed to start${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

echo ""

# Start frontend
echo -e "${PURPLE}🎨 Starting Frontend (Svelte)...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}  ✓ Frontend started (PID: $FRONTEND_PID)${NC}"
echo -e "${CYAN}    URL: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${CYAN}    Logs: logs/frontend.log${NC}"
cd ..

# Wait for frontend to be ready
echo -e "${YELLOW}  Waiting for frontend to start...${NC}"
for i in {1..30}; do
    if check_port $FRONTEND_PORT; then
        echo -e "${GREEN}  ✓ Frontend is ready!${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}  ✗ Frontend failed to start${NC}"
        kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
        exit 1
    fi
done

echo ""

# Optionally start core engine
if command_exists cargo && [ -f "core/target/release/astroscale-node" ]; then
    read -p "$(echo -e ${YELLOW}Start core engine (optional)? [y/N]:${NC} )" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${PURPLE}⚡ Starting Core Engine (Rust)...${NC}"
        cd core
        if [ ! -f "config.toml" ]; then
            ./target/release/astroscale-node init-config > /dev/null 2>&1
        fi
        ./target/release/astroscale-node start > ../logs/core.log 2>&1 &
        CORE_PID=$!
        echo -e "${GREEN}  ✓ Core engine started (PID: $CORE_PID)${NC}"
        echo -e "${CYAN}    API: http://localhost:$CORE_PORT${NC}"
        echo -e "${CYAN}    Metrics: http://localhost:$METRICS_PORT${NC}"
        echo -e "${CYAN}    Logs: logs/core.log${NC}"
        cd ..
        echo ""
    fi
fi

# Save PIDs for shutdown
echo "$BACKEND_PID" > logs/backend.pid
echo "$FRONTEND_PID" > logs/frontend.pid
if [ ! -z "$CORE_PID" ]; then
    echo "$CORE_PID" > logs/core.pid
fi

# Print summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ AstroScale is now running!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}🌐 Access Points:${NC}"
echo -e "   ${GREEN}Frontend:${NC}  http://localhost:$FRONTEND_PORT"
echo -e "   ${GREEN}Backend:${NC}   http://localhost:$BACKEND_PORT"
if [ ! -z "$CORE_PID" ]; then
    echo -e "   ${GREEN}Core API:${NC}  http://localhost:$CORE_PORT"
    echo -e "   ${GREEN}Metrics:${NC}   http://localhost:$METRICS_PORT"
fi
echo ""
echo -e "${CYAN}📊 Try it out:${NC}"
echo -e "   1. Open ${GREEN}http://localhost:$FRONTEND_PORT${NC} in your browser"
echo -e "   2. Click on a star preset (e.g., 'Sun-like Star')"
echo -e "   3. Click '${GREEN}Predict Redshift${NC}' button"
echo -e "   4. See the predicted redshift visualization!"
echo ""
echo -e "${CYAN}📝 Logs:${NC}"
echo -e "   Backend:  ${YELLOW}tail -f logs/backend.log${NC}"
echo -e "   Frontend: ${YELLOW}tail -f logs/frontend.log${NC}"
if [ ! -z "$CORE_PID" ]; then
    echo -e "   Core:     ${YELLOW}tail -f logs/core.log${NC}"
fi
echo ""
echo -e "${CYAN}🛑 To stop:${NC}"
echo -e "   ${YELLOW}./stop.sh${NC} or ${YELLOW}Ctrl+C${NC} (will create stop.sh automatically)"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create stop script
cat > stop.sh << 'STOP_SCRIPT'
#!/bin/bash
echo "🛑 Stopping AstroScale..."
if [ -f logs/backend.pid ]; then
    kill $(cat logs/backend.pid) 2>/dev/null && echo "  ✓ Backend stopped"
    rm logs/backend.pid
fi
if [ -f logs/frontend.pid ]; then
    kill $(cat logs/frontend.pid) 2>/dev/null && echo "  ✓ Frontend stopped"
    rm logs/frontend.pid
fi
if [ -f logs/core.pid ]; then
    kill $(cat logs/core.pid) 2>/dev/null && echo "  ✓ Core engine stopped"
    rm logs/core.pid
fi
echo "✅ All services stopped"
STOP_SCRIPT
chmod +x stop.sh

# Handle Ctrl+C
trap './stop.sh; exit' INT TERM

# Keep script running
echo -e "${YELLOW}Press Ctrl+C to stop all services...${NC}"
echo ""
wait
