#!/bin/bash

echo "🚀 Starting QuickBooks Data Cleaner Portal..."

# Kill existing processes on port 5003
echo "🔄 Cleaning up existing processes..."
lsof -ti:5003 | xargs kill -9 2>/dev/null || true
pkill -f "python app.py" 2>/dev/null || true
pkill -f "flask" 2>/dev/null || true

# Wait a moment for processes to fully terminate
sleep 2

# Navigate to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found in current directory"
    exit 1
fi

# Start the server
echo "🌐 Starting server on http://localhost:5003"
echo "📝 Press Ctrl+C to stop the server"
echo "=================================================="

python app.py 