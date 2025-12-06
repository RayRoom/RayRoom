## Comparison & Demonstrations

Below we compare the output of **RayRoom**'s various rendering engines against **PyRoomAcoustics** (with ray tracing on).

The audio samples illustrate how RayRoom captures richer late reverberation, diffuse energy, and frequency-dependent phenomena compared to standard ISM.

### PyRoomAcoustics (Baseline)

Uses standard Ray Tracing. This serves as a baseline for comparison.

<video controls src="../docs/_static/pyroomacoustics.mp4"></video>

### RayRoom Hybrid

Combines the Image Source Method (ISM) for early reflections with Ray Tracing for late reverberation. This approach balances accuracy and performance.

<video controls src="../docs/_static/hybrid_simulation.mp4"></video>

### RayRoom Radiosity

Focuses on Diffuse Energy modeling. This method is excellent for simulating the diffuse reverberation field using energy exchange between surface patches.

<video controls src="../docs/_static/radiosity_simulation.mp4"></video>

### RayRoom Spectral

Utilizes a Spectral approach, combining Wave physics (FDTD) for low frequencies and Geometric methods (ISM + Ray Tracing) for high frequencies. This provides the highest fidelity across the frequency spectrum.

<video controls src="../docs/_static/spectral_simulation.mp4"></video>

### RayRoom Small Room

Demonstrates Ray Tracing in a smaller acoustic space.

<video controls src="../docs/_static/small_room_simulation.mp4"></video>
