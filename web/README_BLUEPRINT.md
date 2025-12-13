# RayRoom Blueprint Editor

A visual node-based editor for creating RayRoom audio simulation pipelines, inspired by Unreal Engine's Blueprint system.

## Features

- **Visual Node Editor**: Drag-and-drop interface with nodes for Room, Sources, Receivers, Renderers, Audio Inputs, Metrics, Effects, and Output
- **Blueprint-Style UI**: Dark theme with rounded nodes and connection lines, similar to Unreal Engine Blueprints
- **Pipeline Execution**: Execute complete audio simulation pipelines directly from the visual graph
- **Save/Load**: Save and load blueprint configurations as JSON files

## Node Types

### Room Node
- **Type**: Room configuration
- **Parameters**:
  - Room Type: DemoRoom, TestBenchRoom, MedicalRoom8M, MedicalRoom12M, or Custom
  - Mic Type: mono or ambisonic
- **Outputs**: Room object

### Source Node
- **Type**: Audio source configuration
- **Inputs**: Room
- **Parameters**:
  - Name: Source identifier
  - Position: x,y,z coordinates
  - Gain: Audio gain level
- **Outputs**: Source object

### Receiver Node
- **Type**: Microphone/receiver configuration
- **Inputs**: Room
- **Parameters**:
  - Name: Receiver identifier
  - Position: x,y,z coordinates
  - Mic Type: mono or ambisonic
- **Outputs**: Receiver object

### Renderer Node
- **Type**: Audio rendering engine
- **Inputs**: Room, Sources (multiple), Receiver
- **Parameters**:
  - Renderer: HybridRenderer, RadiosityRenderer, SpectralRenderer, or RaytracingRenderer
  - Sample Rate: Audio sample rate (default: 44100)
  - Temperature: Ambient temperature in °C
  - Humidity: Relative humidity in %
  - ISM Order: Image Source Method order
  - N Rays: Number of rays for ray tracing
  - Max Hops: Maximum ray bounces
  - RIR Duration: Room Impulse Response duration in seconds
- **Outputs**: Audio output, RIR (Room Impulse Response)

### Audio Input Node
- **Type**: Audio file input
- **Parameters**:
  - Audio File: Path to WAV/MP3 file
- **Outputs**: Audio data

### Metrics Node
- **Type**: Acoustic and psychoacoustic metrics computation
- **Inputs**: Audio, RIR
- **Parameters**:
  - Acoustic Metrics: RT60, C50, C80, DRR, EDT
  - Psychoacoustic: Loudness, Sharpness, Roughness
  - Performance: Runtime and memory usage
- **Outputs**: Metrics data

### Effects Node
- **Type**: Post-processing audio effects
- **Inputs**: Audio
- **Parameters**:
  - Effect: original, reverb, echo, lowpass, highpass
- **Outputs**: Processed audio

### Output Node
- **Type**: Save results to disk
- **Inputs**: Audio, RIR, Metrics (optional)
- **Parameters**:
  - Output Directory: Where to save files
  - Save Audio: Save rendered audio
  - Save RIR: Save Room Impulse Response
  - Save Metrics: Save computed metrics
  - Save Mesh: Save room geometry as OBJ

## Usage

### Starting the Editor

```bash
python web/launch_blueprint_editor.py
```

This will:
1. Start the backend server on `http://localhost:8000`
2. Open the blueprint editor in your default browser

### Creating a Pipeline

1. **Add Nodes**: Click the node buttons in the toolbar to add nodes to the canvas
2. **Configure Nodes**: Click on nodes to select them, then modify parameters in the node panel
3. **Connect Nodes**: Click and drag from output sockets (right side, blue) to input sockets (left side, cyan)
4. **Execute**: Click "▶ Execute Pipeline" to run the simulation

### Example Pipeline

A typical pipeline might look like:

```
[Room] → [Source] → [Renderer] → [Metrics] → [Output]
         ↑          ↑            ↑
    [Audio Input] [Receiver]  [Effects]
```

### Keyboard Shortcuts

- **Delete/Backspace**: Delete selected node
- **Right-click**: Context menu (coming soon)

### Saving and Loading

- **Save**: Click "💾 Save" to export your blueprint as JSON
- **Load**: Click "📂 Load" to import a previously saved blueprint

## Architecture

### Frontend (`blueprint_editor.html`)
- HTML5 Canvas for connection lines
- DOM elements for nodes
- JavaScript for interaction and pipeline serialization

### Backend (`blueprint_server.py`)
- HTTP server for API requests
- Pipeline executor that:
  - Parses the node graph
  - Performs topological sort to determine execution order
  - Executes nodes in dependency order
  - Handles data flow between nodes

## Connection Rules

- Output sockets can only connect to input sockets
- Input and output types must match (room → room, audio → audio, etc.)
- Multiple sources can connect to a single renderer
- Circular dependencies are detected and prevented

## Troubleshooting

### Server not starting
- Make sure port 8000 is not in use
- Check that all RayRoom dependencies are installed

### Pipeline execution fails
- Verify all required nodes are present (Room, Renderer, Output)
- Check that audio file paths are correct (relative to examples directory)
- Ensure all connections are properly made

### Nodes not connecting
- Make sure you're connecting output (blue) to input (cyan) sockets
- Verify the data types match (room, source, audio, rir, etc.)

## Future Enhancements

- File upload for audio files
- Real-time preview of audio
- Visual feedback during execution
- More node types (custom room geometry, advanced effects)
- Undo/redo functionality
- Node grouping and organization

