## Comparison & Demonstrations

Below we compare the output of **RayRoom**'s various rendering engines against **PyRoomAcoustics** (with ray tracing on).

The audio samples illustrate how RayRoom captures richer late reverberation, diffuse energy, and frequency-dependent phenomena compared to standard ISM.

| Engine | Rendering Method | Audio |
|--------|------------------|-------|
| **PyRoomAcoustics** (Baseline) | Ray Tracing | <video controls width="300" src="https://raw.githubusercontent.com/RayRoom/RayRoom/main/docs/_static/pyroomacoustics.mp4"></video> |
| **RayRoom Hybrid** | ISM + Ray Tracing | <video controls width="300" src="https://raw.githubusercontent.com/RayRoom/RayRoom/main/docs/_static/hybrid_simulation.mp4"></video> |
| **RayRoom Radiosity** | Diffuse Energy | <video controls width="300" src="https://raw.githubusercontent.com/RayRoom/RayRoom/main/docs/_static/radiosity_simulation.mp4"></video> |
| **RayRoom Spectral** | Wave + Geometric | <video controls width="300" src="https://raw.githubusercontent.com/RayRoom/RayRoom/main/docs/_static/spectral_simulation.mp4"></video> |
| **RayRoom Small Room** | Ray Tracing | <video controls width="300" src="https://raw.githubusercontent.com/RayRoom/RayRoom/main/docs/_static/small_room_simulation.mp4"></video> |
