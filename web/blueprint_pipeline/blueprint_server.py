#!/usr/bin/env python3
"""
Blueprint Editor Backend Server (Flask Implementation)
Executes RayRoom pipelines from node-based blueprints
"""

import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from executor import BlueprintExecutor

# Add parent directory to path to allow importing rayroom modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper to determine paths
BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR.parent
PROJECT_ROOT = WEB_DIR.parent


@app.route('/')
@app.route('/blueprint_editor.html')
def serve_editor():
    """Serve the blueprint editor HTML"""
    return send_from_directory(BASE_DIR, 'blueprint_editor.html')


@app.route('/api/audio')
def serve_audio():
    """Serve audio files with support for range requests"""
    file_path = request.args.get('file')
    if not file_path:
        return "Missing file parameter", 400

    # Decode the file path
    file_path = file_path.replace('%2F', '/').replace('%5C', '\\')

    full_path = None
    if not os.path.isabs(file_path):
        # Try multiple locations
        # 1. Relative to web directory
        path_candidate = WEB_DIR / file_path
        if path_candidate.exists():
            full_path = path_candidate
        else:
            # 2. Relative to project root
            path_candidate = PROJECT_ROOT / file_path
            if path_candidate.exists():
                full_path = path_candidate
            else:
                # 3. In examples directory
                path_candidate = PROJECT_ROOT / 'examples' / file_path
                if path_candidate.exists():
                    full_path = path_candidate
    else:
        full_path = Path(file_path)

    if full_path and full_path.exists() and full_path.suffix.lower() in ['.wav', '.mp3', '.ogg']:
        return send_file(full_path)
    else:
        return f"Error - Audio file not found: {file_path}", 404


@app.route('/api/execute', methods=['POST'])
def execute_blueprint():
    """Execute the blueprint"""
    try:
        blueprint = request.get_json()
        if not blueprint:
            return jsonify({'success': False, 'error': 'Invalid JSON'}), 400

        executor = BlueprintExecutor()
        result = executor.execute(blueprint)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def main():
    """Run the blueprint server"""
    port = 8000
    print(f"Blueprint Editor Server (Flask) running on http://localhost:{port}")
    print(f"Open http://localhost:{port} in your browser")
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
