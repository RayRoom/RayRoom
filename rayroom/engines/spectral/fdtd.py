import numpy as np
from tqdm import tqdm

from ...core.constants import C_SOUND


class FDTDSolver:
    """
    Finite Difference Time Domain (FDTD) solver for low-frequency room acoustics.
    Solves the scalar wave equation on a 3D grid.
    """

    def __init__(self, room, max_freq=1000.0, ppw=6.0, cfl=1.0 / np.sqrt(3)):
        """
        Initialize the FDTD solver.

        :param room: RayRoom Room object.
        :param max_freq: Maximum frequency to simulate (determines grid size).
        :param ppw: Points per wavelength (dispersion control). Typically 6-10.
        :param cfl: Courant number (stability). Max 1/sqrt(3) for 3D.
        """
        self.room = room
        self.max_freq = max_freq

        # 1. Grid Setup
        # lambda_min = c / f_max
        # dx <= lambda_min / ppw
        lambda_min = C_SOUND / max_freq
        self.dx = lambda_min / ppw

        # dt <= dx * CFL / c
        self.dt = (self.dx * cfl) / C_SOUND

        print(f"FDTD Setup: f_max={max_freq}Hz, dx={self.dx * 100:.2f}cm, dt={self.dt * 1000:.4f}ms")

        # 2. Voxelize Room
        self._voxelize_room()

        # 3. State Arrays
        # p[n+1], p[n], p[n-1]
        self.p = np.zeros(self.grid_shape, dtype=np.float32)
        self.p_prev = np.zeros(self.grid_shape, dtype=np.float32)
        self.p_next = np.zeros(self.grid_shape, dtype=np.float32)

        # Precompute coefficients
        # Standard wave eq update:
        # p_next = 2*p - p_prev + (c*dt/dx)^2 * Laplacian(p)
        self.courant_sq = (C_SOUND * self.dt / self.dx) ** 2

        # 4. Boundary Setup
        # Instead of global damping, we use a reflection coefficient at the walls.
        # This simulates impedance/admittance boundaries (Robin condition approximation).
        # reflection_coeff (R):
        # R = 1.0 (Rigid/Hard wall)
        # R = -1.0 (Pressure release/Open window)
        # R = 0.9 - 0.98 (Typical wall absorption)
        
        # We estimate an average R from the room materials.
        # Absorption alpha = 1 - R^2  => R = sqrt(1 - alpha)
        
        avg_alpha = 0.0
        total_area = 0.0
        for w in self.room.walls:
            mat = w.material
            alpha = np.mean(mat.absorption) if np.ndim(mat.absorption) > 0 else mat.absorption
            
            # Estimate area
            # Simple approximation for weighting
            area = 1.0 
            avg_alpha += alpha * area
            total_area += area
        
        if total_area > 0:
            avg_alpha /= total_area
        else:
            avg_alpha = 0.1 # Default small absorption

        self.reflection_coeff = np.sqrt(1.0 - avg_alpha)
        # Clamp R to avoid instability or weirdness
        self.reflection_coeff = np.clip(self.reflection_coeff, 0.5, 0.995)
        
        print(f"FDTD Boundary: Average Alpha={avg_alpha:.2f}, Reflection Coeff={self.reflection_coeff:.4f}")

    def _voxelize_room(self):
        """
        Discretize the room geometry into a boolean grid (Air/Solid).
        And precompute neighbor masks for boundary handling.
        """
        # Find bounds
        all_verts = []
        for w in self.room.walls:
            all_verts.extend(w.vertices)
        all_verts = np.array(all_verts)

        min_bounds = np.min(all_verts, axis=0) - 2 * self.dx  # Padding
        max_bounds = np.max(all_verts, axis=0) + 2 * self.dx

        self.origin = min_bounds
        size = max_bounds - min_bounds
        self.grid_shape = np.ceil(size / self.dx).astype(int)

        grid_volume_million = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2] / 1e6
        print(f"Grid shape: {self.grid_shape} ({grid_volume_million:.2f} M voxels)")

        x = np.arange(self.grid_shape[0]) * self.dx + self.origin[0]
        y = np.arange(self.grid_shape[1]) * self.dx + self.origin[1]
        z = np.arange(self.grid_shape[2]) * self.dx + self.origin[2]

        self.is_air = np.zeros(self.grid_shape, dtype=bool)

        print("Voxelizing room (assuming convex for speed)...")

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack((X, Y, Z), axis=-1)

        mask = np.ones(self.grid_shape, dtype=bool)

        for wall in self.room.walls:
            v0 = wall.vertices[0]
            n = wall.normal

            dot_val = points[:, :, :, 0] * n[0] + points[:, :, :, 1] * n[1] + points[:, :, :, 2] * n[2]
            threshold = np.dot(v0, n)

            mask_wall = dot_val >= threshold - 1e-5
            mask = np.logical_and(mask, mask_wall)

        self.is_air = mask

        # Also handle furniture (as obstacles/solids)
        for furn in self.room.furniture:
            if not hasattr(furn, 'vertices') or len(furn.vertices) == 0:
                continue

            # Get the Axis-Aligned Bounding Box (AABB) of the furniture
            min_bounds = np.min(furn.vertices, axis=0)
            max_bounds = np.max(furn.vertices, axis=0)

            # Convert world coordinates to grid indices
            min_idx = self._world_to_grid(min_bounds)
            max_idx = self._world_to_grid(max_bounds)

            # Clip to grid dimensions
            min_idx = np.maximum(0, min_idx)
            max_idx = np.minimum(np.array(self.grid_shape) - 1, max_idx)

            # Set the region to solid (not air)
            if np.all(max_idx > min_idx):
                self.is_air[
                    min_idx[0]:max_idx[0]+1,
                    min_idx[1]:max_idx[1]+1,
                    min_idx[2]:max_idx[2]+1
                ] = False

        print(f"Voxelization complete. Air volume fraction: {np.mean(self.is_air):.2%}")
        
        # Precompute boundary neighbor masks
        # True if neighbor is WALL (not air)
        # Slices correspond to the 'center' region [1:-1, 1:-1, 1:-1]
        
        # x+1
        self.is_wall_x1 = ~self.is_air[2:, 1:-1, 1:-1]
        # x-1
        self.is_wall_x0 = ~self.is_air[:-2, 1:-1, 1:-1]
        
        # y+1
        self.is_wall_y1 = ~self.is_air[1:-1, 2:, 1:-1]
        # y-1
        self.is_wall_y0 = ~self.is_air[1:-1, :-2, 1:-1]
        
        # z+1
        self.is_wall_z1 = ~self.is_air[1:-1, 1:-1, 2:]
        # z-1
        self.is_wall_z0 = ~self.is_air[1:-1, 1:-1, :-2]

    def _world_to_grid(self, pos):
        idx = (np.array(pos) - self.origin) / self.dx
        return np.round(idx).astype(int)

    def run(self, duration, sources, receivers, source_signals=None):
        steps = int(duration / self.dt)

        src_indices = []
        for src in sources:
            idx = self._world_to_grid(src.position)
            idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
            src_indices.append((src, idx))

        rec_indices = []
        rec_signals = {r: np.zeros(steps) for r in receivers}
        for r in receivers:
            idx = self._world_to_grid(r.position)
            idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
            rec_indices.append((r, idx))

        # C^2
        C2 = self.courant_sq
        R = self.reflection_coeff

        print(f"Running FDTD for {steps} steps...")

        for t in tqdm(range(steps)):

            # 1. Inject Source
            for i, (src, idx) in enumerate(src_indices):
                val = 0.0
                if source_signals and src in source_signals:
                    sig = source_signals[src]
                    if t < len(sig):
                        val = sig[t]
                else:
                    # Gaussian Pulse
                    width = 5
                    val = np.exp(-((t - 20) ** 2) / (2 * width ** 2))

                self.p[idx[0], idx[1], idx[2]] += val

            # 2. Update Field (Vectorized)
            p_c = self.p[1:-1, 1:-1, 1:-1]
            p_prev_c = self.p_prev[1:-1, 1:-1, 1:-1]

            # Get neighbors from grid (values in walls are 0 due to masking at end of step)
            p_x1 = self.p[2:, 1:-1, 1:-1]
            p_x0 = self.p[:-2, 1:-1, 1:-1]
            p_y1 = self.p[1:-1, 2:, 1:-1]
            p_y0 = self.p[1:-1, :-2, 1:-1]
            p_z1 = self.p[1:-1, 1:-1, 2:]
            p_z0 = self.p[1:-1, 1:-1, :-2]

            # Apply boundary correction
            # If neighbor is wall, use R * p_c effectively
            # Since p_neighbor is 0 in wall, we add R * p_c where is_wall is True
            
            term_x1 = p_x1 + (self.is_wall_x1 * R * p_c)
            term_x0 = p_x0 + (self.is_wall_x0 * R * p_c)
            term_y1 = p_y1 + (self.is_wall_y1 * R * p_c)
            term_y0 = p_y0 + (self.is_wall_y0 * R * p_c)
            term_z1 = p_z1 + (self.is_wall_z1 * R * p_c)
            term_z0 = p_z0 + (self.is_wall_z0 * R * p_c)

            lap_sum = (term_x1 + term_x0 + term_y1 + term_y0 + term_z1 + term_z0 - 6.0 * p_c)

            self.p_next[1:-1, 1:-1, 1:-1] = 2.0 * p_c - p_prev_c + C2 * lap_sum

            # 3. Apply Mask
            # Forces 0 inside walls (keeping them clean for next step's "p_neighbor is 0" assumption)
            self.p_next *= self.is_air

            # 4. Record Receivers
            for r, idx in rec_indices:
                rec_signals[r][t] = self.p_next[idx[0], idx[1], idx[2]]

            # Cycle buffers
            self.p_prev[:] = self.p[:]
            self.p[:] = self.p_next[:]

        return rec_signals, (1.0 / self.dt)
