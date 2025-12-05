# RayRoom: 3D Room Acoustics Simulator

A Python-based ray tracing acoustics simulator supporting complex room geometries, materials, and furniture.

## Features

- **Ray Tracing Engine**: Stochastic ray tracing for impulse response estimation.
- **Room Creation**: Create shoebox rooms or complex polygons from corner lists.
- **Materials**: Frequency-dependent absorption, transmission (transparency), and scattering coefficients.
- **Objects**: Support for furniture, people (blockers), sources, and receivers (microphones).
- **Transparency**: Walls can be partially transparent (transmission).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Simple Shoebox Room

```python
from rayroom import Room, Source, Receiver, RayTracer, get_material

# Create Room
room = Room.create_shoebox([5, 4, 3])

# Add Source and Receiver
room.add_source(Source("Speaker", [1, 1, 1.5]))
room.add_receiver(Receiver("Mic", [4, 3, 1.5]))

# Run Simulation
tracer = RayTracer(room)
tracer.run(n_rays=10000)

# Access Data
print(room.receivers[0].energy_histogram)
```

### Complex Geometry

See `examples/polygon_room.py` for creating rooms from 2D floor plans.

## Structure

- `rayroom/core.py`: Main simulation engine.
- `rayroom/room.py`: Room and wall definitions.
- `rayroom/objects.py`: Source, Receiver, Furniture classes.
- `rayroom/materials.py`: Material properties.
- `rayroom/geometry.py`: Vector math and intersection tests.
