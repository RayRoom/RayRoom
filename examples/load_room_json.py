import json
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from rayroom.room import Room
    import rayroom.room.objects as objs
    from rayroom.room.visualize import plot_room
except ImportError:
    print("Error: Could not import rayroom.")
    sys.exit(1)


def load_room_from_json(json_path):
    """
    Loads a room configuration from a JSON file created by the Room Creator UI.
    """
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loading room: {data.get('room_name', 'Untitled')}")

    # 1. Create Room Geometry
    height = data.get('room_dims', 3.0)
    walls_data = data.get('walls', [])
    
    if len(walls_data) < 3:
        print("Error: Invalid room geometry (less than 3 wall points).")
        return None
        
    corners = [(w['x'], w['y']) for w in walls_data]
    
    # Create the base room
    room = Room.create_from_corners(corners, height)
    
    # 2. Add Objects
    objects_data = data.get('objects', [])
    print(f"Adding {len(objects_data)} objects...")
    
    for obj_data in objects_data:
        class_name = obj_data.get('class')
        position = obj_data.get('position') # [x, y, z]
        rotation_z = obj_data.get('rotation', 0)
        
        # Get class from rayroom.room.objects
        if not hasattr(objs, class_name):
            print(f"Warning: Unknown object class '{class_name}'. Skipping.")
            continue
            
        obj_class = getattr(objs, class_name)
        
        try:
            # Instantiate object
            # Most objects support (name, position, rotation_z=...)
            # Source and Receiver might be different
            
            if issubclass(obj_class, objs.Source):
                # Source(name, position, ...)
                instance = obj_class(name=f"{class_name}_{obj_data.get('id', '')}", position=position)
                room.add_source(instance)
                
            elif issubclass(obj_class, objs.Receiver):
                # Receiver(name, position, ...)
                instance = obj_class(name=f"{class_name}_{obj_data.get('id', '')}", position=position)
                room.add_receiver(instance)
                
            elif issubclass(obj_class, objs.Furniture):
                # Furniture(name, position, rotation_z=...)
                instance = obj_class(name=f"{class_name}", position=position, rotation_z=rotation_z)
                room.add_furniture(instance)
                
            else:
                print(f"Warning: Class '{class_name}' is not a valid room object type.")
                
        except Exception as e:
            print(f"Error creating '{class_name}': {e}")
            
    return room


def main():
    # Example usage: check if a file arg is provided, else look for room_config.json
    import argparse
    parser = argparse.ArgumentParser(description="Load a room from JSON.")
    parser.add_argument("json_file", nargs='?', default="room_config.json", help="Path to the JSON room file.")
    args = parser.parse_args()
    
    # If default file doesn't exist, warn user
    if args.json_file == "room_config.json" and not os.path.exists("room_config.json"):
        print("Usage: python load_room_json.py <path_to_json>")
        print("No 'room_config.json' found in current directory.")
        print("Please generate a room using 'launch_room_creator.py' and export the JSON first.")
        return

    room = load_room_from_json(args.json_file)
    
    if room:
        print("Room loaded successfully.")
        print(f"  Walls: {len(room.walls)}")
        print(f"  Furniture: {len(room.furniture)}")
        print(f"  Sources: {len(room.sources)}")
        print(f"  Receivers: {len(room.receivers)}")
        
        # Visualize
        print("Plotting room...")
        plot_room(room)


if __name__ == "__main__":
    main()

