#!/bin/bash

# Port settings
API_PORT=8000
FRONTEND_PORT=5173

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "🚀 STARTING NASDAQ-50 QUANT TRADING SIMULATOR WEB"
echo "=================================================="

# Function to clean up background processes on exit
cleanup() {
    echo -e "\n🛑 Stopping servers..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null
    fi
    if [ -n "$VITE_PID" ]; then
        kill "$VITE_PID" 2>/dev/null
    fi
    exit 0
}

# Register the cleanup function for Ctrl+C (SIGINT) and exit (SIGTERM)
trap cleanup SIGINT SIGTERM

# 1. Start FastAPI backend
echo "Starting FastAPI backend on port $API_PORT..."
python3 -m uvicorn api_server:app --port $API_PORT --host 127.0.0.1 > /dev/null 2>&1 &
API_PID=$!

# Wait a second for API to boot
sleep 2

# Verify API is running
if ps -p $API_PID > /dev/null; then
    echo "✅ Backend started successfully (PID: $API_PID)"
else
    echo "❌ Failed to start backend server."
    cleanup
fi

# 2. Start Vite frontend
echo "Starting Vite frontend..."
cd frontend
npm run dev -- --port $FRONTEND_PORT > /dev/null 2>&1 &
VITE_PID=$!

sleep 2

# Verify Vite is running
if ps -p $VITE_PID > /dev/null; then
    echo "✅ Frontend started successfully (PID: $VITE_PID)"
    echo "👉 Open your browser at: http://localhost:$FRONTEND_PORT"
    echo "=================================================="
    echo "Press Ctrl+C to shut down both servers."
else
    echo "❌ Failed to start Vite frontend."
    cleanup
fi

# Keep script running to maintain processes
wait
