#!/usr/bin/env python3
"""
Launcher for the Blueprint Editor
Starts the server and opens the editor in a browser
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blueprint_server import app  # noqa: E402


def start_server():
    """Start the Flask server in a separate thread"""
    port = 8000
    # use_reloader=False is required when running in a separate thread
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def main():
    """Launch the blueprint editor"""
    print("Starting RayRoom Blueprint Editor...")

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(1)

    # Open browser
    url = 'http://localhost:8000'
    print(f"Opening {url} in your browser...")
    webbrowser.open(url)

    print("\nServer is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
