import os
import sys
import json
import numpy as np
from importlib import resources
import jinja2

from rayroom.room.database import (
    MedicalRoom4_5M, MedicalRoom6M, MedicalRoom8M, MedicalRoom9_5M,
    MedicalRoom12M, MedicalRoom15M, MedicalRoom16MConsulting, MedicalRoom16MExamination,
    MedicalRoom18M, MedicalRoom20M,
    MedicalRoom24M, MedicalRoom32M
)
from demo_utils import (
    generate_layouts,
    save_room_mesh,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def save_room_json(room, output_dir, room_name):
    """
    Saves the room configuration as a JSON file compatible with the Room Creator.
    """
    # 1. Extract Room Dimensions & Walls
    # Assuming 'Floor' wall exists and has vertices on z=0
    floor_wall = next((w for w in room.walls if w.name == "Floor"), None)
    
    walls_data = []
    height = 3.0 # Default
    
    if floor_wall:
        # Extract 2D points (x, y)
        # Vertices are typically (x, y, 0)
        # Note: Depending on creation, winding might need check, but raw points are fine for now.
        for v in floor_wall.vertices:
            walls_data.append({"x": float(v[0]), "y": float(v[1])})
    
    # Try to determine height from 'Ceiling' or room bounding box
    if hasattr(room, 'corners') and len(room.corners) == 3:
        height = float(room.corners[2])
    else:
        # Fallback: Find max Z in any wall vertex
        max_z = 0.0
        for w in room.walls:
            z_vals = w.vertices[:, 2]
            if np.max(z_vals) > max_z:
                max_z = np.max(z_vals)
        if max_z > 0:
            height = float(max_z)

    # 2. Extract Objects
    objects_data = []
    
    all_objects = room.furniture + room.sources + room.receivers
    
    for i, obj in enumerate(all_objects):
        obj_type = 'furniture'
        # Basic heuristic for type
        if hasattr(obj, 'power'): obj_type = 'source'
        elif hasattr(obj, 'radius') and not hasattr(obj, 'vertices'): obj_type = 'receiver' # Receivers don't usually have complex meshes
        
        # Get rotation if available (Furniture has rotation_z)
        rotation = getattr(obj, 'rotation_z', 0.0)
        
        objects_data.append({
            "id": i + 1, # Simple ID
            "class": type(obj).__name__,
            "position": [float(obj.position[0]), float(obj.position[1]), float(obj.position[2])],
            "rotation": float(rotation)
        })

    export_data = {
        "room_name": room_name,
        "room_dims": height,
        "walls": walls_data,
        "objects": objects_data
    }
    
    json_path = os.path.join(output_dir, f"{room_name}.json")
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"  Saved JSON config: {json_path}")


def create_viewer_index(output_dir='outputs/medical_room_meshes'):
    """
    Scans for generated medical room viewers and creates a main index.html
    to easily switch between them using a template.
    """
    print(f"Scanning for room viewers in: {output_dir}")

    if not os.path.isdir(output_dir):
        print(f"Error: Output directory not found at '{output_dir}'.")
        print("Please run 'generate_medical_room_meshes.py' first to create the mesh files.")
        return

    room_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    viewer_pages = []
    for room_name in sorted(room_dirs):
        viewer_filename = f"{room_name}_mesh_viewer.html"
        viewer_path = os.path.join(output_dir, room_name, viewer_filename)

        if os.path.exists(viewer_path):
            relative_path = os.path.join(room_name, viewer_filename)
            viewer_pages.append({
                "name": room_name.replace('_', ' ').replace('MedicalRoom', 'Medical Room'),
                "path": relative_path
            })
            print(f"  Found viewer for: {room_name}")
        else:
            print(f"  Warning: Viewer not found for {room_name} at {viewer_path}")

    if not viewer_pages:
        print("No viewer pages found. Please run generate_medical_room_meshes.py first.")
        return

    # Load template
    try:
        template_str = resources.read_text('rayroom.room.templates', 'viewer_index_template.html')
        template = jinja2.Template(template_str)
    except FileNotFoundError:
        print("Error: Could not find 'viewer_index_template.html'.")
        print("Ensure the template file exists in 'rayroom/room/templates/'.")
        return

    # Render the template with the found pages
    html_content = template.render(viewer_pages=viewer_pages)

    # Generate the main index.html
    index_html_path = os.path.join(output_dir, "index.html")
    with open(index_html_path, 'w') as f:
        f.write(html_content)

    print(f"\nSuccessfully created index page: {index_html_path}")
    print("Open this file in your browser to view the rooms.")


def main(output_dir='outputs/medical_room_meshes'):

    medical_rooms = [
        MedicalRoom4_5M, MedicalRoom6M, MedicalRoom8M, MedicalRoom9_5M,
        MedicalRoom12M, MedicalRoom15M, MedicalRoom16MConsulting, MedicalRoom16MExamination,
        MedicalRoom18M, MedicalRoom20M,
        MedicalRoom24M, MedicalRoom32M
    ]

    # 3. Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for room_class in medical_rooms:

        print(f"Processing {room_class.__name__}...")
        room, _, _ = room_class(mic_type='mono').create_room()

        save_room_json(room, output_dir, room_class.__name__)


if __name__ == "__main__":
    main()
