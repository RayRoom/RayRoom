# RayRoom Creator

A visual web-based editor for designing and configuring acoustic rooms for RayRoom audio simulations. Create custom room geometries, place furniture, audio sources, and receivers with an intuitive 2D/3D interface.

![Room Creator Screenshot](./Screenshot.png)

## Features

- **Visual Room Design**: Create custom room geometries using a 2D top-down view with polygon drawing tools
- **Room Templates**: Quick-start with pre-configured shoebox room templates
- **Asset Library**: Browse and place furniture, audio sources, and receivers from the RayRoom object library
- **3D Preview**: Real-time 3D visualization of your room design
- **Interactive Editing**: Click and drag to position objects, adjust properties in real-time
- **Export Configuration**: Save room configurations as JSON for use in RayRoom simulations

## Room Creation

### Shoebox Rooms

Quickly create rectangular rooms using pre-configured templates:
- **4m x 5m Shoebox**: Standard small room template
- **6m x 8m Shoebox**: Medium-sized room template

### Custom Dimensions

Create rectangular rooms with custom dimensions:
1. Enter width, length, and height in meters
2. Click "Create Room with Custom Dimensions"
3. The room will be created with the specified dimensions

### Polygon Rooms

Draw custom room shapes:
1. Click "Draw Polygon" to enter drawing mode
2. Click on the canvas to place vertices
3. Click "Finish" to complete the polygon (minimum 3 points required)
4. Click "Cancel" to abort drawing

## Object Placement

### Furniture

Place furniture objects from the asset library:
1. Select the "Furniture" tab in the asset panel
2. Click on a furniture item to select it
3. Click on the 2D canvas to place it at that location
4. Use the Properties panel to adjust position, rotation, and other parameters

### Audio Sources

Add audio sources to your room:
1. Select the "Sources" tab in the asset panel
2. Choose a source type
3. Click on the canvas to place the source
4. Configure position, gain, and other properties in the Properties panel

### Receivers

Place microphone/receiver objects:
1. Select the "Receivers" tab in the asset panel
2. Choose a receiver type (mono or ambisonic)
3. Click on the canvas to place the receiver
4. Adjust position and microphone type in the Properties panel

## Interface

### Left Sidebar

- **Room Tab**: Room creation controls, dimensions, templates, and statistics
- **Properties Tab**: Object selection and property editing

### Main Canvas

- **2D Top-Down View**: Primary editing interface
  - Click to place objects
  - Click and drag to move objects
  - Scroll to zoom in/out
  - Click and drag background to pan

### Right Sidebar

- **3D Preview**: Real-time 3D visualization of the room
- **Scene List**: Hierarchical list of all objects in the scene
  - Click to select objects
  - Right-click for context menu (delete, etc.)

## Usage

### Starting the Room Creator

```bash
python web/room_creator/launch_room_creator.py
```

This will:
1. Scan the RayRoom object library for available assets (furniture, sources, receivers)
2. Generate the HTML interface with the asset library embedded
3. Open the room creator in your default web browser

### Creating a Room

1. **Choose a Template or Draw Custom**:
   - Use a shoebox template for quick rectangular rooms
   - Or draw a custom polygon for irregular shapes

2. **Place Objects**:
   - Select an object type from the asset library
   - Click on the canvas to place it
   - Adjust properties in the Properties panel

3. **Export Configuration**:
   - Use the export functionality to save your room as JSON
   - The exported file can be used with RayRoom simulation engines

### Keyboard Shortcuts

- **Delete/Backspace**: Delete selected object
- **Scroll Wheel**: Zoom in/out on 2D canvas
- **Click + Drag**: Pan the 2D view or move objects

## Asset Library

The asset library is automatically generated from RayRoom's object classes:

- **Furniture**: All classes inheriting from `Furniture` (tables, chairs, etc.)
- **Sources**: All classes inheriting from `Source` (various audio source types)
- **Receivers**: All classes inheriting from `Receiver` (microphone types)

Each asset includes:
- 3D geometry (vertices and faces) for visualization
- Type information for proper instantiation
- Default properties

## Export Format

The exported JSON configuration includes:
- Room geometry (wall vertices, height)
- Object placements (furniture, sources, receivers)
- Object properties (position, rotation, materials, etc.)

This format is compatible with RayRoom's room loading functionality.

## Architecture

### Frontend (`room_creator.html`)

- **2D Canvas**: HTML5 Canvas for top-down room view and object placement
- **3D Preview**: Three.js-based 3D visualization
- **Asset Library**: Dynamically generated from RayRoom objects
- **State Management**: JavaScript state object tracks room and object data

### Backend (`launch_room_creator.py`)

- **Asset Discovery**: Uses Python introspection to find all RayRoom object classes
- **Template Rendering**: Injects asset library JSON into HTML template
- **File Generation**: Creates standalone HTML file with embedded data

## Troubleshooting

### Assets not appearing

- Ensure RayRoom is properly installed or the project root is in your Python path
- Check that object classes are properly defined in `rayroom.room.objects`
- Verify the template file exists at `rayroom/room/templates/room_creator_template.html`

### Room not displaying

- Check browser console for JavaScript errors
- Ensure the room has at least 3 wall vertices
- Verify room height is greater than 0

### Objects not placing

- Make sure you've selected an object from the asset library first
- Check that you're clicking within the 2D canvas area
- Verify the room has been created before placing objects

### Export not working

- Check browser console for errors
- Ensure all required fields are filled
- Verify the export format matches RayRoom's expected structure

## Integration with RayRoom

The exported room configuration can be used with:

- **Room Loading**: Load custom rooms in Python scripts
- **Blueprint Editor**: Import room configurations into the Blueprint Editor
- **Simulation Engines**: Use with Hybrid, Radiosity, Spectral, or Raytracing renderers

Example usage:

```python
import json
from rayroom.room.base import Room

# Load exported room configuration
with open('my_room.json', 'r') as f:
    room_config = json.load(f)

# Create RayRoom Room object from configuration
room = Room.from_dict(room_config)
```

## Future Enhancements

- Material assignment for walls and objects
- Room import from CAD files
- Undo/redo functionality
- Object rotation controls
- Grid snapping
- Measurement tools
- Room validation and acoustic analysis preview
- Integration with Blueprint Editor for complete pipeline design

