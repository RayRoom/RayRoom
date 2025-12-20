## Comparison & Demonstrations

Below we compare the output of **RayRoom**'s various rendering engines against **PyRoomAcoustics** (with ray tracing on).

The audio samples illustrate how RayRoom captures richer late reverberation, diffuse energy, and frequency-dependent phenomena compared to standard ISM.

### PyRoomAcoustics (Baseline)

Uses standard hybrid simulator based on image source method (ISM) and ray tracing (RT). This serves as a baseline for comparison.



https://github.com/user-attachments/assets/a214fc2d-d7dc-41da-85a0-74e18ee6e3c6




### RayRoom Hybrid

Combines the Image Source Method (ISM) for early reflections with Ray Tracing for late reverberation. This approach balances accuracy and performance.



https://github.com/user-attachments/assets/da7e8a1c-91dc-4d5a-af3a-147174342b15





### RayRoom Spectral (FDTD)

Utilizes a Spectral approach, combining Wave physics (FDTD) for low frequencies and Geometric methods (ISM + Ray Tracing) for high frequencies. This provides the highest fidelity across the frequency spectrum.



https://github.com/user-attachments/assets/36011eff-1454-4ed0-9e0d-651710636384





### RayRoom Radiosity

Focuses on Diffuse Energy modeling. This method is excellent for simulating the diffuse reverberation field using energy exchange between surface patches.



https://github.com/user-attachments/assets/1a335cbc-0d81-421f-a7ff-32c0091177b4





### RayRoom Small Room

Demonstrates Ray Tracing in a smaller acoustic space.



https://github.com/user-attachments/assets/7e01f141-912f-4981-bd41-2183ff2a3420




