import os
import sys
import json
import inspect
import webbrowser

# Add parent directory to path to allow importing rayroom
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import rayroom.room.objects as objs
    from rayroom.room.objects import Furniture, Source, Receiver
except ImportError:
    print(
        "Error: Could not import rayroom. Ensure you are running this from "
        "the examples directory or have installed the package."
    )
    sys.exit(1)


def generate_asset_library():
    library = {}

    # Get all classes in objects module
    for name, cls in inspect.getmembers(objs):
        if not inspect.isclass(cls):
            continue

        # Determine type
        obj_type = None
        if issubclass(cls, Furniture) and cls is not Furniture:
            obj_type = 'furniture'
        elif issubclass(cls, Source) and cls is not Source:
            obj_type = 'source'
        elif issubclass(cls, Receiver) and cls is not Receiver:
            obj_type = 'receiver'

        if not obj_type:
            continue

        # Prepare entry
        entry = {
            "type": obj_type
        }

        # Try to extract geometry for visualization
        if obj_type == 'furniture':

            # Instantiate with dummy values to get geometry
            # Most furniture takes (name, position, ...)
            # Some might have extra required args, we'll try/except
            instance = cls(name="dummy", position=[0, 0, 0])

            if hasattr(instance, 'vertices') and hasattr(instance, 'faces'):
                entry['vertices'] = instance.vertices.tolist()
                entry['faces'] = instance.faces  # Faces are usually list of lists

        library[name] = entry

    return library


def main():
    print("Generating Room Creator UI...")

    # 1. Generate Asset Library
    asset_library = generate_asset_library()
    print(f"Found {len(asset_library)} assets.")

    asset_library_json = json.dumps(asset_library)

    # 2. Load Template
    try:
        # Try reading from local file system if running from source
        template_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'rayroom',
            'room',
            'templates',
            'room_creator_template.html'
        )
        with open(template_path, 'r') as f:
            template_content = f.read()
    except FileNotFoundError:
        print("Template file not found.")
        return

    # 3. Render Template (Simple string replace for now as Jinja might not be needed for just one variable,
    #    but let's use replace since we used {{ asset_library_json }} syntax)
    html_content = template_content.replace('{{ asset_library_json }}', asset_library_json)

    # 4. Save Output
    output_path = os.path.join(os.path.dirname(__file__), 'room_creator.html')
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Successfully created UI at: {output_path}")

    # 5. Open in Browser
    webbrowser.open('file://' + os.path.abspath(output_path))


if __name__ == "__main__":
    main()
