.. RayRoom documentation master file, created by
   sphinx-quickstart on Mon Dec  8 13:42:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RayRoom's documentation!
===================================

RayRoom is a powerful Python-based acoustics simulator designed for flexibility and accuracy. It supports complex room geometries, a variety of materials, and the inclusion of objects like furniture.

Features
--------

- **Multiple Rendering Engines**: RayRoom integrates several simulation techniques, including stochastic ray tracing, the Image Source Method (ISM) for deterministic reflections, acoustic radiosity for diffuse fields, and an FDTD solver for low-frequency wave physics.
- **Hybrid Simulation**: For comprehensive analysis, RayRoom offers hybrid engines that combine the strengths of different methods, such as ISM for early reflections and ray tracing for late reverberation.
- **Complex Geometries**: Model simple shoebox rooms or complex polygonal spaces.
- **Detailed Acoustics**: Define materials with frequency-dependent properties and simulate both mono and first-order Ambisonic audio.
- **In-depth Analysis**: The library includes tools for calculating key acoustic metrics like reverberation time (RT60), clarity (C50, C80), and Direct-to-Reverberant Ratio (DRR).

.. toctree::
   :caption: Contents:

   modules
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

