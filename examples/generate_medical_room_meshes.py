import os
import sys
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
        room_name = room_class.__name__
        room_output_dir = os.path.join(output_dir, room_name)
        if not os.path.exists(room_output_dir):
            os.makedirs(room_output_dir)

        print(f"Processing {room_name}...")
        room, _, _ = room_class(mic_type='mono').create_room()

        # Save layout visualization
        generate_layouts(room, room_output_dir, room_name)

        # Save mesh if requested
        save_room_mesh(room, room_output_dir, room_name)
        print(f"Finished processing {room_name}. Files saved in {room_output_dir}")

    print("\n" + "="*50)
    print("All medical room meshes have been generated.")
    print("="*50 + "\n")

    create_viewer_index(output_dir)


if __name__ == "__main__":
    main()
