## Comparison & Demonstrations

Below we compare the output of **RayRoom**'s various rendering engines against **PyRoomAcoustics** (with ray tracing on).

The audio samples illustrate how RayRoom captures richer late reverberation, diffuse energy, and frequency-dependent phenomena compared to standard ISM.

### PyRoomAcoustics (Baseline)

Uses standard Ray Tracing. This serves as a baseline for comparison.

<audio controls src="https://github.com/RayRoom/RayRoom/raw/main/docs/_static/pyroomacoustics.mp4"></audio>

### RayRoom Hybrid

Combines the Image Source Method (ISM) for early reflections with Ray Tracing for late reverberation. This approach balances accuracy and performance.

<audio controls src="https://github.com/RayRoom/RayRoom/raw/main/hybrid_simulation.wav"></audio>

### RayRoom Radiosity

Focuses on Diffuse Energy modeling. This method is excellent for simulating the diffuse reverberation field using energy exchange between surface patches.

<audio controls src="https://github.com/RayRoom/RayRoom/raw/main/radiosity_simulation.wav"></audio>

### RayRoom Spectral

Utilizes a Spectral approach, combining Wave physics (FDTD) for low frequencies and Geometric methods (ISM + Ray Tracing) for high frequencies. This provides the highest fidelity across the frequency spectrum.

<audio controls src="https://github.com/RayRoom/RayRoom/raw/main/spectral_simulation.wav"></audio>

### RayRoom Small Room

Demonstrates Ray Tracing in a smaller acoustic space.

<audio controls src="https://github.com/RayRoom/RayRoom/raw/main/small_room_simulation.wav"></audio>
