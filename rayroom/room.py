import numpy as np
from .objects import Furniture, Source, Receiver
from .materials import get_material

class Wall:
    def __init__(self, name, vertices, material):
        self.name = name
        self.vertices = np.array(vertices)
        self.material = material
        
        # Compute normal
        p0 = self.vertices[0]
        p1 = self.vertices[1]
        p2 = self.vertices[2]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            self.normal = normal / norm
        else:
            self.normal = np.array([0, 0, 1])

class Room:
    def __init__(self, walls=None):
        self.walls = walls if walls else []
        self.furniture = []
        self.sources = []
        self.receivers = []
        
    def add_furniture(self, item):
        self.furniture.append(item)
        
    def add_source(self, source):
        self.sources.append(source)
        
    def add_receiver(self, receiver):
        self.receivers.append(receiver)

    def plot(self, filename=None, show=True, view='3d'):
        """
        Plot the room geometry and objects.
        view: '3d' or '2d'
        """
        from .visualize import plot_room, plot_room_2d
        if view == '2d':
             plot_room_2d(self, filename, show)
        else:
             plot_room(self, filename, show)

    @classmethod
    def create_shoebox(cls, dimensions, materials=None):
        """
        Create a shoebox room.
        dimensions: [width, depth, height]
        materials: dict or Material (applied to all) or list of 6 materials.
                   keys: floor, ceiling, north, south, east, west
        """
        w, d, h = dimensions
        
        if materials is None:
            mat_def = get_material("concrete")
            mats = {k: mat_def for k in ["floor", "ceiling", "front", "back", "left", "right"]}
        elif isinstance(materials, dict):
            # fill missing with default
            default = get_material("concrete")
            mats = {k: materials.get(k, default) for k in ["floor", "ceiling", "front", "back", "left", "right"]}
        else:
            # Single material
            mats = {k: materials for k in ["floor", "ceiling", "front", "back", "left", "right"]}
            
        # Vertices
        # 0: 0,0,0
        # 1: w,0,0
        # 2: w,d,0
        # 3: 0,d,0
        # 4: 0,0,h
        # 5: w,0,h
        # 6: w,d,h
        # 7: 0,d,h
        
        v = [
            [0,0,0], [w,0,0], [w,d,0], [0,d,0],
            [0,0,h], [w,0,h], [w,d,h], [0,d,h]
        ]
        
        walls = []
        # Floor (normal up) 0-3-2-1
        walls.append(Wall("Floor", [v[0], v[3], v[2], v[1]], mats["floor"]))
        # Ceiling (normal down) 4-5-6-7
        walls.append(Wall("Ceiling", [v[4], v[5], v[6], v[7]], mats["ceiling"]))
        # Front (y=0) 0-1-5-4 -> Normal pointing inside (positive y)
        walls.append(Wall("Front", [v[0], v[1], v[5], v[4]], mats["front"]))
        # Back (y=d) 2-3-7-6 -> Normal pointing inside (negative y)
        walls.append(Wall("Back", [v[2], v[3], v[7], v[6]], mats["back"]))
        # Left (x=0) 3-0-4-7 -> Normal pointing inside (positive x)
        walls.append(Wall("Left", [v[3], v[0], v[4], v[7]], mats["left"]))
        # Right (x=w) 1-2-6-5 -> Normal pointing inside (negative x)
        walls.append(Wall("Right", [v[1], v[2], v[6], v[5]], mats["right"]))
        
        return cls(walls)

    @classmethod
    def create_from_corners(cls, corners, height, materials=None):
        """
        Create room from floor corners (2D) and height.
        corners: list of (x,y) tuples, in order (ccw or cw).
        """
        # Determine winding order to ensure normals point inward.
        # Assuming standard counter-clockwise usually means normals out? 
        # For a room, we want normals pointing *in*.
        
        # Make floor and ceiling
        # Walls connecting them.
        
        if materials is None:
             mat_def = get_material("concrete")
             mats = {"floor": mat_def, "ceiling": mat_def, "walls": mat_def}
        
        walls = []
        
        # Convert to 3D
        floor_verts = [np.array([c[0], c[1], 0.0]) for c in corners]
        ceil_verts = [np.array([c[0], c[1], height]) for c in corners]
        
        # Floor: Normal should be Up (0,0,1)
        # If corners are CCW, standard cross product gives Up.
        # Let's assume CCW.
        walls.append(Wall("Floor", floor_verts, mats.get("floor", get_material("concrete"))))
        
        # Ceiling: Normal Down (0,0,-1). Reverse order.
        walls.append(Wall("Ceiling", ceil_verts[::-1], mats.get("ceiling", get_material("concrete"))))
        
        n = len(corners)
        wall_mat = mats.get("walls", get_material("concrete"))
        
        for i in range(n):
            p1 = floor_verts[i]
            p2 = floor_verts[(i+1)%n]
            p3 = ceil_verts[(i+1)%n]
            p4 = ceil_verts[i]
            
            # Wall rectangle p1, p2, p3, p4
            # If floor is CCW, p1->p2 is along boundary. Up is z.
            # Cross(p2-p1, Up) points Inward if CCW.
            # So Normal = Cross(Right, Up) -> Inward.
            # Vertices order for Inward Normal: p1, p2, p3, p4
            
            walls.append(Wall(f"Wall_{i}", [p1, p2, p3, p4], wall_mat))
            
        return cls(walls)

