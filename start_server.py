#!/usr/bin/env python3
"""
Startup script for QuickBooks Data Cleaner Portal
"""

import os
import sys
import subprocess
import time
import signal
import psutil

def kill_processes_on_port(port):
    """Kill any processes running on the specified port."""
    try:
        # Find processes using the port
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections']:
                    if conn.laddr.port == port:
                        print(f"🔄 Killing process {proc.info['pid']} using port {port}")
                        proc.kill()
                        time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"⚠️ Error killing processes: {e}")

def kill_python_processes():
    """Kill any Python processes that might be running the app."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('app.py' in arg for arg in cmdline):
                    print(f"🔄 Killing existing app.py process {proc.info['pid']}")
                    proc.kill()
                    time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"⚠️ Error killing Python processes: {e}")

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import pandas
        import flask
        import anthropic
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def start_server():
    """Start the Flask server."""
    print("🚀 Starting QuickBooks Data Cleaner Portal...")
    
    # Kill existing processes
    print("🔄 Cleaning up existing processes...")
    kill_processes_on_port(5003)
    kill_python_processes()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies")
        return False
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Check if virtual environment exists
    venv_path = os.path.join(project_dir, 'venv')
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found. Please create it first.")
        return False
    
    # Start the server
    try:
        print("🌐 Starting server on http://localhost:5003")
        print("📝 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the Flask app
        subprocess.run([
            os.path.join(venv_path, 'bin', 'python'),
            'app.py'
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    
    return True

if __name__ == "__main__":
    success = start_server()
    sys.exit(0 if success else 1) 