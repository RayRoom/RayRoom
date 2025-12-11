import numpy as np
from .materials import get_material


class Object3D:
    """
    Base class for 3D objects in the simulation.
    """

    def __init__(self, name, position, material=None):
        """
        Initialize an Object3D.

        :param name: Name of the object.
        :type name: str
        :param position: [x, y, z] coordinates of the object's center or reference point.
        :type position: list or np.ndarray
        :param material: Material properties of the object. Defaults to "default" material.
        :type material: rayroom.materials.Material, optional
        """
        self.name = name
        self.position = np.array(position, dtype=float)
        self.material = material if material else get_material("default")


class Source(Object3D):
    """
    Represents a sound source.
    """

    def __init__(self, name, position, power=1.0, orientation=None, directivity="omnidirectional"):
        """
        Initialize a Source.

        :param name: Name of the source.
        :type name: str
        :param position: [x, y, z] coordinates.
        :type position: list or np.ndarray
        :param power: Sound power of the source. Defaults to 1.0.
        :type power: float
        :param orientation: [x, y, z] vector pointing in the forward direction of the source.
        :type orientation: list or np.ndarray, optional
        :param directivity: Directivity pattern. Options: "omnidirectional", "cardioid",
                            "hypercardioid", "bidirectional", "subcardioid".
        :type directivity: str
        """
        super().__init__(name, position)
        self.power = power  # Scalar or array for bands
        self.orientation = np.array(orientation if orientation else [1, 0, 0], dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm
        self.directivity = directivity


class Receiver(Object3D):
    """
    Represents a microphone or listening point.
    """

    def __init__(self, name, position, radius=0.1):
        """
        Initialize a Receiver.

        :param name: Name of the receiver.
        :type name: str
        :param position: [x, y, z] coordinates.
        :type position: list or np.ndarray
        :param radius: Radius of the receiver sphere for ray intersection. Defaults to 0.1.
        :type radius: float
        """
        super().__init__(name, position)
        self.radius = radius
        self.amplitude_histogram = []  # To store arriving energy packets (time, amplitude)

    def record(self, time, energy):
        """
        Record an energy packet arrival.

        :param time: Arrival time in seconds.
        :type time: float
        :param energy: Energy value of the arriving packet.
        :type energy: float or np.ndarray
        """
        # Convert energy to amplitude
        if energy >= 0:
            self.amplitude_histogram.append((time, np.sqrt(energy)))


class AmbisonicReceiver(Object3D):
    """
    Represents a first-order Ambisonic microphone.
    """

    def __init__(self, name, position, orientation=None, radius=0.01):
        """
        Initialize an AmbisonicReceiver.

        :param name: Name of the receiver.
        :type name: str
        :param position: [x, y, z] coordinates.
        :type position: list or np.ndarray
        :param orientation: [x, y, z] vector pointing in the forward direction (X-axis).
        :type orientation: list or np.ndarray, optional
        :param radius: Radius for ray intersection tests.
        :type radius: float
        """
        super().__init__(name, position)
        self.radius = radius

        # Histograms for W, X, Y, Z channels
        self.w_histogram = []
        self.x_histogram = []
        self.y_histogram = []
        self.z_histogram = []

        # Define the orientation of the microphone capsules
        self.orientation = np.array(orientation if orientation is not None else [1, 0, 0], dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm

        # Create an orthonormal basis for the microphone's local coordinate system
        self.x_axis = self.orientation  # Forward

        # Ensure the up vector is not parallel to the forward vector
        up_global = np.array([0., 0., 1.])
        if np.allclose(np.abs(np.dot(self.x_axis, up_global)), 1.0):
            # If forward is aligned with global Z, use global Y as up
            up_global = np.array([0., 1., 0.])

        self.y_axis = np.cross(up_global, self.x_axis)  # Left
        self.y_axis /= np.linalg.norm(self.y_axis)

        self.z_axis = np.cross(self.x_axis, self.y_axis)  # Up
        self.z_axis /= np.linalg.norm(self.z_axis)

    def record(self, time, energy, direction):
        """
        Record an energy packet arrival from a specific direction.

        :param time: Arrival time in seconds.
        :type time: float
        :param energy: Energy value of the arriving packet.
        :type energy: float or np.ndarray
        :param direction: Normalized vector indicating the direction of arrival.
        :type direction: np.ndarray
        """
        if energy < 0:
            return

        amplitude = np.sqrt(energy)

        # W channel (omnidirectional)
        gain_w = 1.0
        self.w_histogram.append((time, amplitude * gain_w))

        # X, Y, Z channels (figure-of-eight / bidirectional)
        # Gain is the projection of the arrival direction onto the capsule's axis
        gain_x = np.dot(direction, self.x_axis)
        gain_y = np.dot(direction, self.y_axis)
        gain_z = np.dot(direction, self.z_axis)

        self.x_histogram.append((time, amplitude * gain_x))
        self.y_histogram.append((time, amplitude * gain_y))
        self.z_histogram.append((time, amplitude * gain_z))


class Furniture(Object3D):
    """
    Represents a complex 3D object (mesh) in the room, like a table or chair.
    """

    def __init__(self, name, position, vertices, faces, material=None, rotation_z=0):
        """
        Initialize Furniture.

        :param name: Name of the object.
        :type name: str
        :param position: [x, y, z] coordinates for the object's center.
        :type position: list or np.ndarray
        :param vertices: List of [x, y, z] coordinates for the mesh vertices, relative to the object's center.
        :type vertices: list
        :param faces: List of faces, where each face is a list of vertex indices.
        :type faces: list[list[int]]
        :param material: Material properties.
        :type material: rayroom.materials.Material, optional
        :param rotation_z: Rotation in degrees around the Z-axis.
        :type rotation_z: float, optional
        """
        super().__init__(name, position, material)
        self.rotation_z = rotation_z
        vertices = np.array(vertices)

        # Apply rotation around Z-axis if specified
        if rotation_z != 0:
            theta = np.deg2rad(rotation_z)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta,  cos_theta, 0],
                [0,          0,         1]
            ])
            vertices = np.dot(vertices, rotation_matrix.T)

        # Translate vertices to the final world position
        self.vertices = vertices + self.position
        self.faces = faces

        # Precompute normals and plane equations for faces
        self.face_normals = []
        self.face_planes = []  # Point on plane

        for face in self.faces:
            p0 = self.vertices[face[0]]
            p1 = self.vertices[face[1]]
            p2 = self.vertices[face[2]]

            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            self.face_normals.append(normal)
            self.face_planes.append(p0)


class Person(Furniture):
    """
    Represents a person as a blocky, Minecraft-style character.
    """

    def __init__(self, name, position, rotation_z=0, height=1.7, width=0.5, depth=0.3, material_name="human"):
        """
        Initialize a Person object.

        :param name: Name of the person.
        :type name: str
        :param position: [x, y, z] coordinates of the feet center.
        :type position: list or np.ndarray
        :param rotation_z: Rotation in degrees around the Z-axis.
        :type rotation_z: float, optional
        :param height: Height of the person in meters. Defaults to 1.7.
        :type height: float
        :param width: Width of the person (shoulder width) in meters. Defaults to 0.5.
        :type width: float
        :param depth: Depth of the person (chest depth) in meters. Defaults to 0.3.
        :type depth: float
        :param material_name: Name of the material to use. Defaults to "human".
        :type material_name: str
        """
        parts = []

        # Proportions (approximate)
        head_h = height * 0.15
        torso_h = height * 0.45
        leg_h = height * 0.40

        head_w = width * 0.5
        torso_w = width
        leg_w = width * 0.25

        head_d = depth * 0.8
        torso_d = depth
        arm_d = depth * 0.8
        leg_d = depth * 0.8

        # Torso
        torso_pos = [0, 0, leg_h]
        parts.append(_create_box_vertices_faces([torso_w, torso_d, torso_h], torso_pos))

        # Head
        head_pos = [0, 0, leg_h + torso_h]
        parts.append(_create_box_vertices_faces([head_w, head_d, head_h], head_pos))

        # Legs
        left_leg_pos = [-torso_w / 4, 0, 0]
        right_leg_pos = [torso_w / 4, 0, 0]
        parts.append(_create_box_vertices_faces([leg_w, leg_d, leg_h], left_leg_pos))
        parts.append(_create_box_vertices_faces([leg_w, leg_d, leg_h], right_leg_pos))

        # Arms
        arm_h = torso_h * 0.9
        arm_w = width * 0.2
        left_arm_pos = [-torso_w / 2 - arm_w / 2, 0, leg_h]
        right_arm_pos = [torso_w / 2 + arm_w / 2, 0, leg_h]
        parts.append(_create_box_vertices_faces([arm_w, arm_d, arm_h], left_arm_pos))
        parts.append(_create_box_vertices_faces([arm_w, arm_d, arm_h], right_arm_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


def _create_box_vertices_faces(dimensions, center_bottom_pos=(0, 0, 0)):
    """
    Creates vertices and faces for a box.

    :param dimensions: [width, depth, height]
    :type dimensions: list or np.ndarray
    :param center_bottom_pos: [x, y, z] of the center of the bottom face.
    :type center_bottom_pos: list or np.ndarray
    :return: Tuple of (vertices, faces)
    :rtype: (np.ndarray, list)
    """
    x, y, z = center_bottom_pos
    w, d, h = dimensions

    verts = [
        [x - w / 2, y - d / 2, z], [x + w / 2, y - d / 2, z],
        [x + w / 2, y + d / 2, z], [x - w / 2, y + d / 2, z],  # Bottom
        [x - w / 2, y - d / 2, z + h], [x + w / 2, y - d / 2, z + h],
        [x + w / 2, y + d / 2, z + h], [x - w / 2, y + d / 2, z + h]  # Top
    ]

    faces = [
        [0, 1, 2, 3],  # Bottom
        [4, 7, 6, 5],  # Top
        [0, 4, 5, 1],  # Front
        [1, 5, 6, 2],  # Right
        [2, 6, 7, 3],  # Back
        [3, 7, 4, 0]   # Left
    ]

    return np.array(verts), faces


def _create_composite_object(parts):
    """
    Combines multiple vertex/face lists into a single mesh.

    :param parts: A list of tuples, where each tuple is (vertices, faces).
    :type parts: list
    :return: Tuple of (combined_vertices, combined_faces)
    :rtype: (np.ndarray, list)
    """
    all_verts = []
    all_faces = []
    num_verts = 0

    for verts, faces in parts:
        all_verts.append(verts)
        # Handle ragged face arrays (e.g., from cylinders with tri and quad faces)
        offset_faces = [[vertex_index + num_verts for vertex_index in face] for face in faces]
        all_faces.extend(offset_faces)
        num_verts += len(verts)

    if not all_verts:
        return np.array([]), []

    return np.vstack(all_verts), all_faces


def _create_cylinder_vertices_faces(radius, height, resolution=16, center_bottom_pos=(0, 0, 0)):
    """
    Creates vertices and faces for a cylinder with inward-facing normals.

    :param radius: Radius of the cylinder.
    :type radius: float
    :param height: Height of the cylinder.
    :type height: float
    :param resolution: Number of vertices in the circular cross-section.
    :type resolution: int, optional
    :param center_bottom_pos: [x, y, z] of the center of the bottom face.
    :type center_bottom_pos: list or np.ndarray
    :return: Tuple of (vertices, faces)
    :rtype: (np.ndarray, list)
    """
    x, y, z = center_bottom_pos

    verts = []

    # Bottom circle vertices (indices 0 to resolution-1)
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        verts.append([x + radius * np.cos(angle), y + radius * np.sin(angle), z])

    # Top circle vertices (indices resolution to 2*resolution-1)
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        verts.append([x + radius * np.cos(angle), y + radius * np.sin(angle), z + height])

    # Bottom center (index 2*resolution)
    verts.append([x, y, z])
    bottom_center_idx = 2 * resolution

    # Top center (index 2*resolution + 1)
    verts.append([x, y, z + height])
    top_center_idx = 2 * resolution + 1

    faces = []

    # Side faces (quads, inward normal)
    for i in range(resolution):
        p_bl = i
        p_br = (i + 1) % resolution
        p_tl = i + resolution
        p_tr = (i + 1) % resolution + resolution
        faces.append([p_bl, p_tl, p_tr, p_br])

    # Bottom cap faces (triangles, inward normal pointing +z)
    for i in range(resolution):
        p1 = i
        p2 = (i + 1) % resolution
        faces.append([bottom_center_idx, p1, p2])

    # Top cap faces (triangles, inward normal pointing -z)
    for i in range(resolution):
        p1 = i + resolution
        p2 = (i + 1) % resolution + resolution
        faces.append([top_center_idx, p2, p1])

    return np.array(verts), faces


def _create_ring_vertices_faces(outer_radius, inner_radius, height, resolution=16, center_bottom_pos=(0, 0, 0)):
    """
    Creates vertices and faces for a ring (a cylinder with a hole).
    """
    x, y, z = center_bottom_pos
    verts = []

    # Bottom outer, bottom inner, top outer, top inner
    for r in [outer_radius, inner_radius]:
        for h_offset in [0, height]:
            for i in range(resolution):
                angle = 2 * np.pi * i / resolution
                verts.append([x + r * np.cos(angle), y + r * np.sin(angle), z + h_offset])

    faces = []
    res = resolution
    # Outer wall
    for i in range(res):
        p1 = i
        p2 = (i + 1) % res
        p3 = i + 2 * res
        p4 = (i + 1) % res + 2 * res
        faces.append([p1, p2, p4, p3])

    # Inner wall
    for i in range(res):
        p1 = i + res
        p2 = (i + 1) % res + res
        p3 = i + 3 * res
        p4 = (i + 1) % res + 3 * res
        faces.append([p1, p3, p4, p2])

    # Top surface
    for i in range(res):
        p1 = i + 2 * res
        p2 = (i + 1) % res + 2 * res
        p3 = i + 3 * res
        p4 = (i + 1) % res + 3 * res
        faces.append([p1, p3, p4, p2])

    # Bottom surface
    for i in range(res):
        p1 = i
        p2 = (i + 1) % res
        p3 = i + res
        p4 = (i + 1) % res + res
        faces.append([p1, p3, p4, p2])

    return np.array(verts), faces


def _create_torus_vertices_faces(major_radius, minor_radius, major_resolution=24, minor_resolution=12):
    """
    Creates vertices and faces for a torus.
    """
    verts = []
    # Generate vertices
    for i in range(major_resolution + 1):
        u = 2 * np.pi * i / major_resolution
        cos_u, sin_u = np.cos(u), np.sin(u)
        for j in range(minor_resolution + 1):
            v = 2 * np.pi * j / minor_resolution
            cos_v, sin_v = np.cos(v), np.sin(v)

            x = (major_radius + minor_radius * cos_v) * cos_u
            y = (major_radius + minor_radius * cos_v) * sin_u
            z = minor_radius * sin_v
            verts.append([x, y, z])

    faces = []
    # Generate faces
    for i in range(major_resolution):
        for j in range(minor_resolution):
            p1_idx = i * (minor_resolution + 1) + j
            p2_idx = (i + 1) * (minor_resolution + 1) + j
            p3_idx = (i + 1) * (minor_resolution + 1) + (j + 1)
            p4_idx = i * (minor_resolution + 1) + (j + 1)
            faces.append([p1_idx, p2_idx, p3_idx, p4_idx])

    return np.array(verts), faces


def _create_sphere_vertices_faces(radius, lat_res=16, lon_res=16, center=(0, 0, 0)):
    """
    Creates vertices and faces for a sphere with inward-facing normals.
    """
    verts = []
    # Generate vertices
    for i in range(lat_res + 1):
        lat_angle = np.pi * i / lat_res
        for j in range(lon_res + 1):
            lon_angle = 2 * np.pi * j / lon_res
            x = radius * np.sin(lat_angle) * np.cos(lon_angle)
            y = radius * np.sin(lat_angle) * np.sin(lon_angle)
            z = radius * np.cos(lat_angle)
            verts.append([x, y, z])

    faces = []
    # Generate faces
    for i in range(lat_res):
        for j in range(lon_res):
            p1 = i * (lon_res + 1) + j
            p2 = p1 + 1
            p3 = (i + 1) * (lon_res + 1) + j
            p4 = p3 + 1
            # For inward normals, reverse winding order
            faces.append([p1, p3, p4, p2])

    verts = np.array(verts) + np.array(center)
    return np.array(verts), faces


class Chair(Furniture):
    """
    Represents a simple chair made of a seat, a back, and four legs.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        """
        Initialize a Chair object.

        :param name: Name of the chair.
        :type name: str
        :param position: [x, y, z] coordinates of the center of the chair on the floor.
        :type position: list or np.ndarray
        :param rotation_z: Rotation in degrees around the Z-axis.
        :type rotation_z: float, optional
        :param material_name: Name of the material. Defaults to "wood".
        :type material_name: str
        """
        parts = []

        # Seat
        seat_dims = [0.5, 0.5, 0.05]
        seat_pos = [0, 0, 0.4]
        parts.append(_create_box_vertices_faces(seat_dims, seat_pos))

        # Back
        back_dims = [0.5, 0.05, 0.5]
        back_pos = [0, -0.225, 0.425]
        parts.append(_create_box_vertices_faces(back_dims, back_pos))

        # Legs
        leg_dims = [0.04, 0.04, 0.4]
        leg_positions = [
            [-0.23, -0.23, 0],
            [0.23, -0.23, 0],
            [-0.23, 0.23, 0],
            [0.23, 0.23, 0],
        ]
        for pos in leg_positions:
            parts.append(_create_box_vertices_faces(leg_dims, pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class DiningTable(Furniture):
    """
    Represents a dining table with a tabletop and four legs.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        parts = []

        # Tabletop
        top_dims = [1.5, 0.8, 0.05]
        top_pos = [0, 0, 0.7]
        parts.append(_create_box_vertices_faces(top_dims, top_pos))

        # Legs
        leg_dims = [0.05, 0.05, 0.7]
        leg_positions = [
            [-0.7, -0.35, 0],
            [0.7, -0.35, 0],
            [-0.7, 0.35, 0],
            [0.7, 0.35, 0],
        ]
        for pos in leg_positions:
            parts.append(_create_box_vertices_faces(leg_dims, pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class CoffeeTable(Furniture):
    """
    Represents a coffee table with a tabletop and four legs.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood", legs_to_remove=None):
        """
        Initialize a CoffeeTable object.

        :param name: Name of the table.
        :type name: str
        :param position: [x, y, z] coordinates for the object's center.
        :type position: list or np.ndarray
        :param rotation_z: Rotation in degrees around the Z-axis.
        :type rotation_z: float, optional
        :param material_name: Name of the material. Defaults to "wood".
        :type material_name: str
        :param legs_to_remove: A list of leg indices (0-3) to remove. 0:FL, 1:FR, 2:BL, 3:BR
        :type legs_to_remove: list, optional
        """
        parts = []

        # Tabletop
        top_dims = [1.0, 0.5, 0.04]
        top_pos = [0, 0, 0.36]
        parts.append(_create_box_vertices_faces(top_dims, top_pos))

        # Legs
        leg_dims = [0.04, 0.04, 0.36]
        leg_positions = [
            [-0.45, -0.22, 0],  # 0: front-left
            [0.45, -0.22, 0],   # 1: front-right
            [-0.45, 0.22, 0],   # 2: back-left
            [0.45, 0.22, 0],    # 3: back-right
        ]

        if legs_to_remove is None:
            legs_to_remove = []

        for i, pos in enumerate(leg_positions):
            if i not in legs_to_remove:
                parts.append(_create_box_vertices_faces(leg_dims, pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class TV(Furniture):
    """
    Represents a simple TV as a thin box.
    """
    def __init__(self, name, position, rotation_z=0, material_name="glass"):
        dims = [1.2, 0.05, 0.7]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Desk(Furniture):
    """
    Represents a desk with a tabletop and side/back panels.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        parts = []
        self.width, self.depth, self.height = 1.2, 0.6, 0.75
        panel_thickness = 0.05
        top_thickness = 0.05
        panel_height = self.height - top_thickness

        # Tabletop
        top_dims = [self.width, self.depth, top_thickness]
        top_pos = [0, 0, panel_height]
        parts.append(_create_box_vertices_faces(top_dims, top_pos))

        # Side panels
        side_panel_dims = [panel_thickness, self.depth, panel_height]
        left_pos = [-self.width / 2 + panel_thickness / 2, 0, 0]
        right_pos = [self.width / 2 - panel_thickness / 2, 0, 0]
        parts.append(_create_box_vertices_faces(side_panel_dims, left_pos))
        parts.append(_create_box_vertices_faces(side_panel_dims, right_pos))

        # Back panel
        back_panel_dims = [self.width, panel_thickness, panel_height]
        back_pos = [0, -self.depth / 2 + panel_thickness / 2, 0]
        parts.append(_create_box_vertices_faces(back_panel_dims, back_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


def _create_couch_geometry(width, options=None):
    """Helper function to create couch geometry."""
    if options is None:
        options = {}

    parts = []
    depth = options.get("depth", 0.9)
    seat_height = options.get("seat_height", 0.4)
    back_height = options.get("back_height", 0.4)
    arm_height = options.get("arm_height", 0.2)
    arm_width = options.get("arm_width", 0.2)

    # Base
    base_dims = [width, depth, seat_height]
    parts.append(_create_box_vertices_faces(base_dims, [0, 0, 0]))

    # Backrest
    if not options.get("no_backrest", False):
        back_dims = [width, 0.2, back_height]
        back_pos = [0, -depth / 2 + 0.1, seat_height]
        parts.append(_create_box_vertices_faces(back_dims, back_pos))

    # Armrests
    if not options.get("no_armrests", False):
        arm_dims = [arm_width, depth, arm_height]
        left_arm_pos = [-width / 2 + arm_width / 2, 0, seat_height]
        right_arm_pos = [width / 2 - arm_width / 2, 0, seat_height]
        parts.append(_create_box_vertices_faces(arm_dims, left_arm_pos))
        parts.append(_create_box_vertices_faces(arm_dims, right_arm_pos))

    vertices, faces = _create_composite_object(parts)
    return vertices, faces


class ThreeSeatCouch(Furniture):
    """
    Represents a three-seat couch with base, backrest, and armrests.
    """
    def __init__(self, name, position, rotation_z=0, material_name="fabric", options=None):
        vertices, faces = _create_couch_geometry(2.0, options)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class TwoSeatCouch(Furniture):
    """
    Represents a two-seat couch with base, backrest, and armrests.
    """
    def __init__(self, name, position, rotation_z=0, material_name="fabric", options=None):
        vertices, faces = _create_couch_geometry(1.5, options)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class OneSeatCouch(Furniture):
    """
    Represents a one-seat couch (armchair) with base, backrest, and armrests.
    """
    def __init__(self, name, position, rotation_z=0, material_name="fabric", options=None):
        vertices, faces = _create_couch_geometry(1.0, options)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SquareCarpet(Furniture):
    """
    Represents a square of thick carpet as a flat box.
    """
    def __init__(self, name, position, rotation_z=0, material_name="thick_carpet"):
        dims = [2.0, 2.0, 0.02]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Subwoofer(Furniture):
    """
    Represents a subwoofer as a cube with a large driver cone.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood", resolution=16):
        parts = []
        width, depth, height = 0.4, 0.4, 0.4

        # Main cabinet
        cabinet_dims = [width, depth, height]
        parts.append(_create_box_vertices_faces(cabinet_dims, center_bottom_pos=[0, 0, 0]))

        # Driver (cylinder) on the front face (positive y)
        driver_thickness = 0.02
        front_y = depth / 2
        driver_radius = 0.15
        driver_z = height / 2

        driver_verts, driver_faces = _create_cylinder_vertices_faces(
            driver_radius, driver_thickness, resolution, [0, 0, -driver_thickness/2]
        )

        # Rotate to face forward
        angle = np.deg2rad(90)
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        driver_verts = np.dot(driver_verts, rotation_matrix.T)
        driver_verts += np.array([0, front_y, driver_z])
        parts.append((driver_verts, driver_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class FloorstandingSpeaker(Furniture):
    """
    Represents a floorstanding speaker as a tall box with speaker drivers.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood", resolution=12):
        parts = []
        width, depth, height = 0.3, 0.4, 1.0

        # Main cabinet
        cabinet_dims = [width, depth, height]
        parts.append(_create_box_vertices_faces(cabinet_dims, center_bottom_pos=[0, 0, 0]))

        # Drivers (cylinders) on the front face (positive y)
        driver_thickness = 0.02
        front_y = depth / 2

        # Define a rotation matrix to make drivers face forward (+y)
        angle = np.deg2rad(90)
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

        # Tweeter
        tweeter_radius = 0.03
        tweeter_z = height * 0.85
        tweeter_verts, tweeter_faces = _create_cylinder_vertices_faces(
            tweeter_radius, driver_thickness, resolution, [0, 0, -driver_thickness / 2]
        )
        tweeter_verts = np.dot(tweeter_verts, rotation_matrix.T)
        tweeter_verts += np.array([0, front_y, tweeter_z])
        parts.append((tweeter_verts, tweeter_faces))

        # Mid-range driver 1
        mid_radius = 0.06
        mid1_z = height * 0.65
        mid1_verts, mid1_faces = _create_cylinder_vertices_faces(
            mid_radius, driver_thickness, resolution, [0, 0, -driver_thickness / 2]
        )
        mid1_verts = np.dot(mid1_verts, rotation_matrix.T)
        mid1_verts += np.array([0, front_y, mid1_z])
        parts.append((mid1_verts, mid1_faces))

        # Mid-range driver 2
        mid2_z = height * 0.45
        mid2_verts, mid2_faces = _create_cylinder_vertices_faces(
            mid_radius, driver_thickness, resolution, [0, 0, -driver_thickness / 2]
        )
        mid2_verts = np.dot(mid2_verts, rotation_matrix.T)
        mid2_verts += np.array([0, front_y, mid2_z])
        parts.append((mid2_verts, mid2_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Window(Furniture):
    """
    Represents a single rectangular window as a thin box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=1.0, height=1.5, thickness=0.05, material_name="glass"):
        dims = [width, thickness, height]
        # Center the box on its position
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -height / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class DoubleRectangleWindow(Furniture):
    """
    Represents a double rectangular window as two thin boxes side-by-side.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=1.8, height=1.5, thickness=0.05,
                 frame_width=0.1, material_name="glass"):
        parts = []
        pane_width = (width - frame_width) / 2

        # Left pane
        left_pane_dims = [pane_width, thickness, height]
        left_pane_pos = [-width/2 + pane_width/2, 0, -height/2]
        parts.append(_create_box_vertices_faces(left_pane_dims, left_pane_pos))

        # Right pane
        right_pane_dims = [pane_width, thickness, height]
        right_pane_pos = [width/2 - pane_width/2, 0, -height/2]
        parts.append(_create_box_vertices_faces(right_pane_dims, right_pane_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SquareWindow(Window):
    """
    Represents a square window as a thin box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, size=1.0, thickness=0.05, material_name="glass"):
        super().__init__(name, position, rotation_z, width=size, height=size,
                         thickness=thickness, material_name=material_name)


class Painting(Furniture):
    """
    Represents a painting as a thin box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=0.8, height=0.6, thickness=0.05, material_name="fabric"):
        dims = [width, thickness, height]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -height / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class FloatingTVShelf(Furniture):
    """
    Represents a floating TV shelf as a box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=1.5, depth=0.4, height=0.2, material_name="wood"):
        dims = [width, depth, height]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -height / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class WallShelf(Furniture):
    """
    Represents a wall shelf as a box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=1.0, depth=0.3, height=0.05, material_name="wood"):
        dims = [width, depth, height]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -height / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class KitchenCabinet(Furniture):
    """
    Represents a kitchen cabinet as a box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=0.8, depth=0.4, height=0.9, material_name="wood"):
        dims = [width, depth, height]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -height / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Clock(Furniture):
    """
    Represents a wall clock as a thin cylinder.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, diameter=0.3, thickness=0.05,
                 material_name="plastic", resolution=16):
        radius = diameter / 2
        # Create a cylinder oriented along the Z-axis, centered at the origin
        verts, faces = _create_cylinder_vertices_faces(
            radius, thickness, resolution=resolution, center_bottom_pos=[0, 0, -thickness / 2]
        )

        # Rotate the cylinder to be flat against a wall (Y-axis as normal)
        # This rotates it from being upright on XY plane to being flat on XZ plane.
        angle = np.deg2rad(-90)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        verts = np.dot(verts, rotation_matrix.T)

        # The Furniture class init will handle rotation_z and final positioning.
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Door(Furniture):
    """
    Represents a standard door as a thin box.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, width=0.9, height=2.0,
                 thickness=0.05, material_name="wood", handle=True, handle_side='negative'):
        # Create door geometry
        door_dims = [width, thickness, height]
        door_verts, door_faces = _create_box_vertices_faces(door_dims, center_bottom_pos=[0, 0, 0])
        parts = [(door_verts, door_faces)]

        if handle:
            # Handle parameters
            handle_height = 1.0
            handle_x_offset = width / 2 - 0.1
            knob_radius = 0.03
            shaft_radius = 0.01
            shaft_length = 0.05

            if handle_side == 'positive':
                y_dir = 1
            else:  # negative
                y_dir = -1

            # Handle Shaft (cylinder)
            shaft_pos_y = y_dir * (thickness / 2)
            shaft_verts, shaft_faces = _create_cylinder_vertices_faces(
                shaft_radius, shaft_length, resolution=6, center_bottom_pos=[0, 0, 0]
            )
            # rotate around X to point along Y
            angle = np.deg2rad(-90 * y_dir)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            shaft_verts = np.dot(shaft_verts, rotation_matrix.T)
            shaft_verts += np.array([handle_x_offset, shaft_pos_y, handle_height])
            parts.append((shaft_verts, shaft_faces))

            # Handle Knob (sphere)
            knob_pos_y = y_dir * (thickness / 2 + shaft_length)
            knob_center = [handle_x_offset, knob_pos_y, handle_height]
            knob_verts, knob_faces = _create_sphere_vertices_faces(
                knob_radius, lat_res=6, lon_res=6, center=knob_center
            )
            parts.append((knob_verts, knob_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Smartphone(Furniture):
    """
    Represents a smartphone as a thin box.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="glass"):
        dims = [0.07, 0.15, 0.008]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Tablet(Furniture):
    """
    Represents a tablet as a thin box.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="glass"):
        dims = [0.17, 0.25, 0.007]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class EchoDot5(Furniture):
    """
    Represents an Amazon Echo Dot (5th Gen) as a sphere-like box.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="fabric"):
        dims = [0.1, 0.1, 0.086]  # diameter, diameter, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class EchoDot2(Furniture):
    """
    Represents an Amazon Echo Dot (2nd Gen) as a short cylinder (box).
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.084, 0.084, 0.032]  # diameter, diameter, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class EchoShow5(Furniture):
    """
    Represents an Amazon Echo Show 5 as a wedge-like box.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.147, 0.082, 0.091]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class GoogleNestMini(Furniture):
    """
    Represents a Google Nest Mini as a puck-like box.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="fabric"):
        dims = [0.098, 0.098, 0.042]  # diameter, diameter, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class AmazonEcho2(Furniture):
    """
    Represents an Amazon Echo (2nd Gen) as a cylinder.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="fabric", resolution=16):
        """
        Initialize an AmazonEcho2 object.

        :param name: Name of the object.
        :type name: str
        :param position: [x, y, z] coordinates for the object's bottom center.
        :type position: list or np.ndarray
        :param rotation_z: Rotation in degrees around the Z-axis.
        :type rotation_z: float, optional
        :param material_name: Name of the material. Defaults to "fabric".
        :type material_name: str
        :param resolution: The number of vertices for the circular cross-section. Defaults to 16.
        :type resolution: int, optional
        """
        radius = 0.088 / 2
        height = 0.148
        verts, faces = _create_cylinder_vertices_faces(
            radius, height, resolution=resolution, center_bottom_pos=[0, 0, 0]
        )
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class CRTMonitor(Furniture):
    """
    Represents a CRT monitor with a screen, a bulky back, and a stand.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []

        # Base foot
        base_dims = [0.25, 0.25, 0.04]
        parts.append(_create_box_vertices_faces(base_dims, [0, 0, 0]))

        # Stand/neck
        neck_dims = [0.1, 0.1, 0.06]
        parts.append(_create_box_vertices_faces(neck_dims, [0, 0, 0.04]))

        monitor_bottom_z = 0.1

        # Front part (bezel and screen)
        front_dims = [0.4, 0.05, 0.35]
        front_pos = [0, 0.175, monitor_bottom_z]  # Centered at y=0.175 to make front face at y=0.2
        parts.append(_create_box_vertices_faces(front_dims, front_pos))

        # Back box for CRT
        back_dims = [0.35, 0.35, 0.3]
        back_pos_y = -0.025  # Centered to be behind front part
        back_pos_z = monitor_bottom_z + (front_dims[2] - back_dims[2]) / 2  # Vertically centered with front
        back_pos = [0, back_pos_y, back_pos_z]
        parts.append(_create_box_vertices_faces(back_dims, back_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class LCDMonitor(Furniture):
    """
    Represents an LCD monitor with a screen, stand, and base.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        screen_bottom_z = 0.1

        # Base
        base_dims = [0.25, 0.2, 0.02]
        parts.append(_create_box_vertices_faces(base_dims, [0, 0, 0]))

        # Stand/Neck
        neck_dims = [0.04, 0.04, screen_bottom_z]
        parts.append(_create_box_vertices_faces(neck_dims, [0, 0, 0.02]))

        # Screen
        screen_dims = [0.5, 0.05, 0.3]
        screen_pos = [0, 0, screen_bottom_z]
        parts.append(_create_box_vertices_faces(screen_dims, screen_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class iMac(Furniture):
    """
    Represents an iMac-style computer with a screen and L-shaped stand.
    """
    def __init__(self, name, position, rotation_z=0, material_name="glass"):
        parts = []

        # L-shaped stand
        # Foot part
        foot_dims = [0.2, 0.2, 0.01]
        parts.append(_create_box_vertices_faces(foot_dims, [0, 0, 0]))
        # Hinge part
        hinge_dims = [0.05, 0.02, 0.1]
        hinge_pos = [0, -0.09, 0.01]  # At the back of the foot
        parts.append(_create_box_vertices_faces(hinge_dims, hinge_pos))

        # Screen
        screen_dims = [0.55, 0.05, 0.4]
        # Create screen geometry centered at the origin for easier rotation
        screen_verts, screen_faces = _create_box_vertices_faces(
            screen_dims, [0, 0, -screen_dims[2] / 2]
        )

        # Tilt the screen back slightly
        angle = np.deg2rad(10)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        screen_verts = np.dot(screen_verts, rotation_matrix.T)

        # Position the screen to sit on the hinge
        hinge_top_z = hinge_pos[2] + hinge_dims[2]
        screen_min_z_after_rot = np.min(screen_verts[:, 2])

        translation = np.array([
            0,  # Center the screen on the stand's x
            hinge_pos[1],  # Align the screen's y-center with the hinge's y-center
            hinge_top_z - screen_min_z_after_rot  # Move screen up to sit on the hinge
        ])

        screen_verts += translation
        parts.append((screen_verts, screen_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Laptop(Furniture):
    """
    Represents a laptop in an open position.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        # Base
        base_dims = [0.35, 0.25, 0.02]
        parts.append(_create_box_vertices_faces(base_dims, [0, 0, 0]))
        # Screen (at an angle)
        screen_dims = [0.35, 0.01, 0.25]
        screen_verts, screen_faces = _create_box_vertices_faces(screen_dims, [0, 0, 0])
        # Rotate screen part
        angle = np.deg2rad(25)  # degrees open
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        screen_verts = np.dot(screen_verts, rotation_matrix.T)
        # Position screen part
        screen_verts += np.array([0, -0.125, 0.02])
        parts.append((screen_verts, screen_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name=name,
                         position=position,
                         vertices=vertices.tolist(),
                         faces=faces,
                         material=get_material(material_name),
                         rotation_z=rotation_z)


class Printer(Furniture):
    """
    Represents a standard office printer.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        width, depth, height = 0.4, 0.45, 0.3

        # Lower part (main body with paper tray)
        lower_h = height * 0.6
        lower_dims = [width, depth, lower_h]
        lower_pos = [0, 0, 0]
        parts.append(_create_box_vertices_faces(lower_dims, lower_pos))

        # Upper part (scanner unit)
        upper_h = height * 0.4
        upper_depth = depth * 0.8
        upper_dims = [width, upper_depth, upper_h]
        # Position it on top of the lower part, flush with the back
        upper_pos = [0, -(depth - upper_depth) / 2, lower_h]
        parts.append(_create_box_vertices_faces(upper_dims, upper_pos))

        # Paper output tray lip (a small detail)
        lip_h = 0.02
        lip_depth = 0.05
        lip_dims = [width * 0.7, lip_depth, lip_h]
        lip_pos = [0, depth / 2, lower_h * 0.7]  # Sticking out the front
        parts.append(_create_box_vertices_faces(lip_dims, lip_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class StackOfPaper(Furniture):
    """
    Represents a stack of A4 paper.
    """
    def __init__(self, name, position, rotation_z=0, material_name="paper"):
        dims = [0.210, 0.297, 0.05]  # A4 size, 5cm high
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class TissueBox(Furniture):
    """
    Represents a tissue box.
    """
    def __init__(self, name, position, rotation_z=0, material_name="paper"):
        dims = [0.22, 0.12, 0.1]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class RoundBin(Furniture):
    """
    Represents a round trash bin.
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        radius = 0.15
        height = 0.4
        verts, faces = _create_cylinder_vertices_faces(radius, height, resolution=16, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SquareBin(Furniture):
    """
    Represents a square trash bin.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.3, 0.3, 0.4]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class CeilingFan(Furniture):
    """
    Represents a ceiling fan.
    The object's position is where it attaches to the ceiling.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        parts = []
        # Mount
        mount_radius = 0.08
        mount_height = 0.1
        mount_verts, mount_faces = _create_cylinder_vertices_faces(
            mount_radius, mount_height, resolution=12, center_bottom_pos=[0, 0, -mount_height]
        )
        parts.append((mount_verts, mount_faces))
        # Motor
        motor_radius = 0.12
        motor_height = 0.2
        motor_verts, motor_faces = _create_cylinder_vertices_faces(
            motor_radius, motor_height, resolution=16, center_bottom_pos=[0, 0, -mount_height - motor_height]
        )
        parts.append((motor_verts, motor_faces))
        # Blades (simple boxes)
        num_blades = 5
        blade_dims = [0.5, 0.1, 0.01]  # length, width, thickness
        for i in range(num_blades):
            angle = 2 * np.pi * i / num_blades

            # Create blade centered at origin, aligned with x-axis
            blade_verts, blade_faces = _create_box_vertices_faces(
                blade_dims, center_bottom_pos=[0, 0, -blade_dims[2] / 2]
            )

            # Rotate blade to point outwards
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            blade_verts = np.dot(blade_verts, rotation_matrix.T)

            # Translate blade to its position on the fan
            blade_radius = motor_radius + blade_dims[0] / 2
            blade_pos = np.array([
                blade_radius * np.cos(angle),
                blade_radius * np.sin(angle),
                -mount_height - motor_height / 2  # Center of motor height
            ])
            blade_verts += blade_pos
            parts.append((blade_verts, blade_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name=name,
                         position=position,
                         vertices=vertices.tolist(),
                         faces=faces,
                         material=get_material(material_name),
                         rotation_z=rotation_z)


class ACWallUnit(Furniture):
    """
    Represents an AC Wall Unit.
    Position is center of the face attached to the wall.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.8, 0.25, 0.3]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, -dims[1] / 2, -dims[2] / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class TallFanOnFoot(Furniture):
    """
    Represents a tall fan on a foot.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        # Base
        base_verts, base_faces = _create_cylinder_vertices_faces(
            0.2, 0.05, resolution=16, center_bottom_pos=[0, 0, 0]
        )
        parts.append((base_verts, base_faces))
        # Pole
        pole_verts, pole_faces = _create_cylinder_vertices_faces(
            0.02, 1.2, resolution=8, center_bottom_pos=[0, 0, 0.05]
        )
        parts.append((pole_verts, pole_faces))

        # Fan Head Assembly (created at origin, pointing along Z-axis)
        head_assembly_parts = []
        head_radius = 0.2
        head_depth = 0.15

        # Motor
        motor_verts, motor_faces = _create_cylinder_vertices_faces(
            head_radius * 0.3, head_depth, resolution=12, center_bottom_pos=[0, 0, -head_depth/2]
        )
        head_assembly_parts.append((motor_verts, motor_faces))

        # Blades
        num_blades = 3
        blade_len = head_radius * 1.6
        blade_wid = head_radius * 0.5
        blade_pitch_deg = 20
        motor_radius = head_radius * 0.3

        for i in range(num_blades):
            # 1. Create blade centered at origin, aligned with X-axis
            blade_verts, blade_faces = _create_box_vertices_faces(
                [blade_len, blade_wid, 0.01],
                center_bottom_pos=[0, 0, -0.005]
            )

            # 2. Add pitch by rotating around X-axis
            pitch_angle = np.deg2rad(blade_pitch_deg)
            pitch_rot_mat = np.array([
                [1, 0, 0],
                [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
                [0, np.sin(pitch_angle), np.cos(pitch_angle)]
            ])
            blade_verts = np.dot(blade_verts, pitch_rot_mat.T)

            # 3. Translate blade away from center along X-axis
            blade_pos_x = motor_radius + blade_len / 2
            blade_verts += np.array([blade_pos_x, 0, 0])

            # 4. Rotate blade to its final position around the fan's Z-axis
            angle = 2 * np.pi * i / num_blades
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            z_rot_mat = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            blade_verts = np.dot(blade_verts, z_rot_mat.T)

            head_assembly_parts.append((blade_verts, blade_faces))

        head_verts, head_faces = _create_composite_object(head_assembly_parts)

        # Rotate head assembly to be vertical (blades vertical, fan points forward in Y)
        rot = np.deg2rad(90)
        rot_mat = np.array([[1, 0, 0], [0, np.cos(rot), -np.sin(rot)], [0, np.sin(rot), np.cos(rot)]])
        head_verts = np.dot(head_verts, rot_mat.T)
        # Position the head
        head_verts += np.array([0, 0, 1.25])
        parts.append((head_verts, head_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SmallFanOnFoot(Furniture):
    """
    Represents a small fan on a foot.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        # Base
        base_verts, base_faces = _create_cylinder_vertices_faces(
            0.1, 0.03, resolution=16, center_bottom_pos=[0, 0, 0]
        )
        parts.append((base_verts, base_faces))
        # Pole
        pole_verts, pole_faces = _create_cylinder_vertices_faces(
            0.015, 0.4, resolution=8, center_bottom_pos=[0, 0, 0.03]
        )
        parts.append((pole_verts, pole_faces))

        # Fan Head Assembly (created at origin, pointing along Z-axis)
        head_assembly_parts = []
        head_radius = 0.15
        head_depth = 0.1

        # Motor
        motor_verts, motor_faces = _create_cylinder_vertices_faces(
            head_radius * 0.3, head_depth, resolution=12, center_bottom_pos=[0, 0, -head_depth/2]
        )
        head_assembly_parts.append((motor_verts, motor_faces))

        # Blades
        num_blades = 3
        blade_len = head_radius * 1.6
        blade_wid = head_radius * 0.5
        blade_pitch_deg = 20
        motor_radius = head_radius * 0.3

        for i in range(num_blades):
            # 1. Create blade centered at origin, aligned with X-axis
            blade_verts, blade_faces = _create_box_vertices_faces(
                [blade_len, blade_wid, 0.01],
                center_bottom_pos=[0, 0, -0.005]
            )

            # 2. Add pitch by rotating around X-axis
            pitch_angle = np.deg2rad(blade_pitch_deg)
            pitch_rot_mat = np.array([
                [1, 0, 0],
                [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
                [0, np.sin(pitch_angle), np.cos(pitch_angle)]
            ])
            blade_verts = np.dot(blade_verts, pitch_rot_mat.T)

            # 3. Translate blade away from center along X-axis
            blade_pos_x = motor_radius + blade_len / 2
            blade_verts += np.array([blade_pos_x, 0, 0])

            # 4. Rotate blade to its final position around the fan's Z-axis
            angle = 2 * np.pi * i / num_blades
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            z_rot_mat = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            blade_verts = np.dot(blade_verts, z_rot_mat.T)

            head_assembly_parts.append((blade_verts, blade_faces))

        head_verts, head_faces = _create_composite_object(head_assembly_parts)

        # Rotate head assembly to be vertical (blades vertical, fan points forward in Y)
        rot = np.deg2rad(90)
        rot_mat = np.array([[1, 0, 0], [0, np.cos(rot), -np.sin(rot)], [0, np.sin(rot), np.cos(rot)]])
        head_verts = np.dot(head_verts, rot_mat.T)
        # Position the head
        head_verts += np.array([0, 0, 0.43])
        parts.append((head_verts, head_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class HospitalBed(Furniture):
    """
    Represents a hospital bed with a frame, mattress, headboard, footboard, and side rails.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        # Dimensions
        bed_width, bed_depth, bed_height = 0.9, 2.0, 0.6
        mattress_height = 0.2
        head_foot_board_height = 0.4
        rail_height = 0.2
        # Frame and Mattress
        frame_dims = [bed_width, bed_depth, bed_height - mattress_height]
        parts.append(_create_box_vertices_faces(frame_dims, [0, 0, 0]))
        mattress_dims = [bed_width, bed_depth, mattress_height]
        parts.append(_create_box_vertices_faces(mattress_dims, [0, 0, bed_height - mattress_height]))
        # Headboard and Footboard
        board_dims = [bed_width, 0.05, head_foot_board_height]
        headboard_pos = [0, -bed_depth / 2, bed_height]
        footboard_pos = [0, bed_depth / 2, bed_height]
        parts.append(_create_box_vertices_faces(board_dims, headboard_pos))
        parts.append(_create_box_vertices_faces(board_dims, footboard_pos))
        # Side Rails
        rail_dims = [0.05, bed_depth * 0.7, rail_height]
        left_rail_pos = [-bed_width / 2, 0, bed_height]
        right_rail_pos = [bed_width / 2, 0, bed_height]
        parts.append(_create_box_vertices_faces(rail_dims, left_rail_pos))
        parts.append(_create_box_vertices_faces(rail_dims, right_rail_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class ExaminingTable(Furniture):
    """
    Represents an examining table with a padded top and a base frame.
    """
    def __init__(self, name, position, rotation_z=0, material_name="leather"):
        parts = []
        # Dimensions
        top_width, top_depth, top_height = 0.7, 1.8, 0.1
        base_height = 0.7
        leg_width, leg_depth = 0.05, 0.05
        # Padded Top
        top_pos = [0, 0, base_height]
        parts.append(_create_box_vertices_faces([top_width, top_depth, top_height], top_pos))
        # Base Frame/Legs
        frame_width = top_width - 0.1
        frame_depth = top_depth - 0.1
        leg_positions = [
            [-frame_width / 2, -frame_depth / 2, 0],
            [frame_width / 2, -frame_depth / 2, 0],
            [-frame_width / 2, frame_depth / 2, 0],
            [frame_width / 2, frame_depth / 2, 0]
        ]
        for pos in leg_positions:
            parts.append(_create_box_vertices_faces([leg_width, leg_depth, base_height], pos))
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class DentalChair(Furniture):
    """
    Represents a dental chair with a base, seat, backrest, headrest, leg rest, and armrests.
    """
    def __init__(self, name, position, rotation_z=0, material_name="leather"):
        parts = []
        # Dimensions
        seat_width, seat_depth, seat_thickness = 0.6, 0.6, 0.1
        seat_z = 0.5
        # Base and Pole
        parts.append(_create_cylinder_vertices_faces(0.3, 0.1, resolution=16, center_bottom_pos=[0, 0, 0]))
        parts.append(_create_cylinder_vertices_faces(0.1, 0.4, resolution=8, center_bottom_pos=[0, 0, 0.1]))

        # Seat
        seat_pos = [0, 0, seat_z]
        parts.append(_create_box_vertices_faces([seat_width, seat_depth, seat_thickness], seat_pos))

        # Backrest
        back_dims = [seat_width, 0.1, 0.8]
        back_verts, back_faces = _create_box_vertices_faces(back_dims, center_bottom_pos=[0, 0, 0])
        angle = np.deg2rad(45)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        back_verts = np.dot(back_verts, rotation_matrix.T)
        back_translation = np.array([0, -seat_depth / 2, seat_z])
        back_verts += back_translation
        parts.append((back_verts, back_faces))

        # Headrest
        head_dims = [0.3, 0.08, 0.2]
        head_verts, head_faces = _create_box_vertices_faces(head_dims, center_bottom_pos=[0, 0, 0])
        head_verts = np.dot(head_verts, rotation_matrix.T)
        top_of_back_local = np.array([0, 0, back_dims[2]])
        headrest_translation = back_translation + np.dot(top_of_back_local, rotation_matrix.T)
        head_verts += headrest_translation
        parts.append((head_verts, head_faces))

        # Leg Rest
        leg_dims = [seat_width, 0.42, 0.1]
        leg_verts, leg_faces = _create_box_vertices_faces(leg_dims, center_bottom_pos=[0, 0, 0])
        angle_leg = np.deg2rad(-45)
        rotation_matrix_leg = np.array([
            [1, 0, 0],
            [0, np.cos(angle_leg), -np.sin(angle_leg)],
            [0, np.sin(angle_leg), np.cos(angle_leg)]
        ])
        leg_verts = np.dot(leg_verts, rotation_matrix_leg.T)
        leg_translation = np.array([0, seat_depth / 2, seat_z])
        leg_verts += leg_translation
        parts.append((leg_verts, leg_faces))

        # Armrests
        arm_dims = [0.05, 0.5, 0.15]
        arm_z = seat_z + seat_thickness
        left_arm_pos = [-seat_width/2 - arm_dims[0]/2, 0, arm_z]
        right_arm_pos = [seat_width/2 + arm_dims[0]/2, 0, arm_z]
        parts.append(_create_box_vertices_faces(arm_dims, left_arm_pos))
        parts.append(_create_box_vertices_faces(arm_dims, right_arm_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class MedicalStool(Furniture):
    """
    Represents a medical stool with a wheeled base.
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        parts = []
        # Dimensions
        seat_radius = 0.2
        seat_height = 0.05
        pole_radius = 0.02
        pole_height = 0.45
        base_center_radius = pole_radius * 2.5
        base_center_height = 0.05
        leg_height = 0.02

        # Central part of the base
        base_center_verts, base_center_faces = _create_cylinder_vertices_faces(
            base_center_radius, base_center_height, resolution=12, center_bottom_pos=[0, 0, 0]
        )
        parts.append((base_center_verts, base_center_faces))

        # Base with 5 legs
        num_legs = 5
        leg_dims = [0.2, 0.04, leg_height]  # length, width, height
        for i in range(num_legs):
            angle = 2 * np.pi * i / num_legs
            # Create leg aligned with x-axis
            leg_verts, leg_faces = _create_box_vertices_faces(
                leg_dims, center_bottom_pos=[leg_dims[0] / 2, 0, 0]
            )

            # Rotate leg
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            leg_verts = np.dot(leg_verts, rotation_matrix.T)

            # Translate leg to start from the edge of the central base
            start_pos = np.array([base_center_radius * cos_a, base_center_radius * sin_a, leg_height / 2])
            leg_verts += start_pos
            parts.append((leg_verts, leg_faces))

        # Pole
        pole_verts, pole_faces = _create_cylinder_vertices_faces(
            pole_radius, pole_height, resolution=8, center_bottom_pos=[0, 0, base_center_height]
        )
        parts.append((pole_verts, pole_faces))

        # Seat
        seat_pos_z = base_center_height + pole_height
        seat_verts, seat_faces = _create_cylinder_vertices_faces(
            seat_radius, seat_height, resolution=16, center_bottom_pos=[0, 0, seat_pos_z]
        )
        parts.append((seat_verts, seat_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name=name,
                         position=position,
                         vertices=vertices.tolist(),
                         faces=faces,
                         material=get_material(material_name),
                         rotation_z=rotation_z)


class HorizontalCabinets(Furniture):
    """
    Represents horizontal wall-mounted cabinets.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        dims = [1.5, 0.4, 0.6]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class VerticalCabinets(Furniture):
    """
    Represents vertical, floor-standing cabinets.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        dims = [0.8, 0.6, 1.8]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Sink(Furniture):
    """
    Represents a sink within a cabinet, featuring a semi-cylindrical basin.
    """
    def __init__(self, name, position, rotation_z=0, material_name="ceramic"):
        parts = []
        # --- Dimensions ---
        cab_w, cab_d, cab_h = 0.6, 0.5, 0.8  # Cabinet dimensions
        top_h = 0.05  # Countertop height
        basin_w, basin_d, basin_depth = 0.45, 0.35, 0.15  # Basin dimensions

        # --- Cabinet and Countertop (with a hole) ---
        # 1. Main cabinet box
        cabinet_dims = [cab_w, cab_d, cab_h]
        parts.append(_create_box_vertices_faces(cabinet_dims, center_bottom_pos=[0, 0, 0]))

        # 2. Countertop pieces around the basin hole
        # Front piece
        front_strip_d = (cab_d - basin_d) / 2
        front_dims = [cab_w, front_strip_d, top_h]
        front_pos = [0, (basin_d + front_strip_d) / 2, cab_h]
        parts.append(_create_box_vertices_faces(front_dims, front_pos))
        # Back piece
        back_pos = [0, -(basin_d + front_strip_d) / 2, cab_h]
        parts.append(_create_box_vertices_faces(front_dims, back_pos))
        # Left piece
        side_strip_w = (cab_w - basin_w) / 2
        side_dims = [side_strip_w, basin_d, top_h]
        left_pos = [-(basin_w + side_strip_w) / 2, 0, cab_h]
        parts.append(_create_box_vertices_faces(side_dims, left_pos))
        # Right piece
        right_pos = [(basin_w + side_strip_w) / 2, 0, cab_h]
        parts.append(_create_box_vertices_faces(side_dims, right_pos))

        # --- Basin (Semi-cylinder) ---
        verts = []
        faces = []
        radius = basin_w / 2
        resolution = 12  # Number of segments for the curve

        # Generate vertices for the front and back semi-circular arcs
        for y_offset in [-basin_d / 2, basin_d / 2]:
            for i in range(resolution + 1):
                angle = np.pi + (np.pi * i / resolution)  # Range [pi, 2*pi] for U-shape
                x = radius * np.cos(angle)
                z = basin_depth + radius * np.sin(angle)  # basin_depth is the bottom
                verts.append([x, y_offset, z + cab_h - basin_depth])

        # Create faces for the curved basin wall
        for i in range(resolution):
            p1 = i
            p2 = i + 1
            p3 = i + 1 + (resolution + 1)
            p4 = i + (resolution + 1)
            faces.append([p1, p4, p3, p2])  # Inward-facing normal

        # Create faces for the flat side walls of the basin
        faces.append([0, resolution + 1, resolution + 1 + resolution, resolution])
        faces.append([0, 1, resolution + 2, resolution + 1])

        # Add the basin geometry to the parts list
        parts.append((np.array(verts), faces))

        # --- Faucet (simple representation) ---
        faucet_base_verts, faucet_base_faces = _create_cylinder_vertices_faces(
            0.02, 0.1, 8, [0, -basin_d/2 - 0.05, cab_h]
        )
        parts.append((faucet_base_verts, faucet_base_faces))
        spout_verts, spout_faces = _create_box_vertices_faces(
            [0.02, 0.15, 0.02], [0, -basin_d/2, cab_h + 0.1]
        )
        parts.append((spout_verts, spout_faces))

        # --- Final Assembly ---
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Wheelchair(Furniture):
    """
    Represents a wheelchair with a seat, backrest, armrests, large wheels, and footrests.
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        parts = []
        seat_width, seat_depth, seat_height = 0.5, 0.5, 0.45

        # Seat
        seat_dims = [seat_width, seat_depth, 0.05]
        parts.append(_create_box_vertices_faces(seat_dims, [0, 0, seat_height]))

        # Backrest
        back_dims = [seat_width, 0.05, 0.5]
        back_pos = [0, -seat_depth / 2, seat_height]
        parts.append(_create_box_vertices_faces(back_dims, back_pos))

        # Armrests
        arm_dims = [0.05, seat_depth, 0.2]
        left_arm_pos = [-seat_width / 2, 0, seat_height]
        right_arm_pos = [seat_width / 2, 0, seat_height]
        parts.append(_create_box_vertices_faces(arm_dims, left_arm_pos))
        parts.append(_create_box_vertices_faces(arm_dims, right_arm_pos))

        # Large Wheels
        wheel_radius = 0.3
        wheel_thickness = 0.02
        wheel_verts, wheel_faces = _create_cylinder_vertices_faces(
            wheel_radius, wheel_thickness, 16, [0, 0, 0]
        )
        rot = np.deg2rad(90)
        # Rotate around Y-axis to make wheels vertical and parallel to seat depth
        rot_mat = np.array([
            [np.cos(rot), 0, np.sin(rot)],
            [0, 1, 0],
            [-np.sin(rot), 0, np.cos(rot)]
        ])

        left_wheel_verts = np.dot(
            wheel_verts, rot_mat.T) + np.array([-seat_width / 2 - wheel_thickness, 0, wheel_radius])
        parts.append((left_wheel_verts, wheel_faces))

        right_wheel_verts = np.dot(
            wheel_verts, rot_mat.T) + np.array([seat_width / 2, 0, wheel_radius])
        parts.append((right_wheel_verts, wheel_faces))

        # Footrests
        footrest_dims = [0.15, 0.2, 0.02]
        footrest_y = seat_depth / 2 + 0.1
        footrest_z = 0.15
        left_foot_pos = [-seat_width / 4, footrest_y, footrest_z]
        right_foot_pos = [seat_width / 4, footrest_y, footrest_z]
        parts.append(_create_box_vertices_faces(footrest_dims, left_foot_pos))
        parts.append(_create_box_vertices_faces(footrest_dims, right_foot_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Walker(Furniture):
    """
    Represents a walker.
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        dims = [0.6, 0.5, 0.8]
        # Simplified as a box frame
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Defibrillator(Furniture):
    """
    Represents a defibrillator unit on a stand.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        # Box
        box_dims = [0.3, 0.3, 0.2]
        parts.append(_create_box_vertices_faces(box_dims, [0, 0, 0.8]))
        # Pole
        pole_verts, pole_faces = _create_cylinder_vertices_faces(
            0.02, 0.8, 8, [0, 0, 0]
        )
        parts.append((pole_verts, pole_faces))
        # Base
        base_dims = [0.4, 0.4, 0.05]
        parts.append(_create_box_vertices_faces(base_dims, [0, 0, 0]))
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class WeighingScale(Furniture):
    """
    Represents a medical weighing scale.
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        parts = []
        # Pole
        pole_verts, pole_faces = _create_cylinder_vertices_faces(
            0.015, 1.8, 8, [0, 0, 0]
        )
        parts.append((pole_verts, pole_faces))
        # Base (simplified)
        base_dims = [0.4, 0.4, 0.05]
        parts.append(_create_box_vertices_faces(base_dims, [0, 0, 0]))
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class MRIScanner(Furniture):
    """
    Represents an MRI scanner with a cylindrical gantry, patient bed, and base.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        parts = []
        # Gantry (Cylindrical Ring)
        gantry_verts, gantry_faces = _create_ring_vertices_faces(
            outer_radius=1.0, inner_radius=0.4, height=0.8, resolution=32, center_bottom_pos=[0, -0.4, 0]
        )
        # Rotate gantry to be upright
        angle = np.deg2rad(90)
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        gantry_verts = np.dot(gantry_verts, rotation_matrix_x.T)
        gantry_verts += np.array([0, 0, 1.0])  # Lift it up
        parts.append((gantry_verts, gantry_faces))
        # Base for the patient bed
        bed_base_dims = [0.5, 1.5, 0.7]  # width, depth, height
        bed_base_pos = [0, 0.8, 0]
        parts.append(_create_box_vertices_faces(bed_base_dims, bed_base_pos))
        # Patient Bed
        bed_dims = [0.6, 2.5, 0.1]
        bed_pos = [0, -0.2, 0.7]  # On top of the base
        parts.append(_create_box_vertices_faces(bed_dims, bed_pos))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Ventilator(Furniture):
    """
    Represents a ventilator machine on a stand.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.5, 0.5, 1.2]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class UltrasoundMachine(Furniture):
    """
    Represents an ultrasound machine.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.6, 0.7, 1.3]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class ECG(Furniture):
    """
    Represents an ECG machine on a cart.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.5, 0.6, 1.0]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class OperatingRoomLight(Furniture):
    """
    Represents an operating room light fixture.
    Position is where it attaches to the ceiling.
    """
    def __init__(self, name, position, rotation_z=0, material_name="glass"):
        parts = []
        # Arm
        arm_verts, arm_faces = _create_cylinder_vertices_faces(
            0.05, 1.0, 8, [0, 0, -1.0]
        )
        parts.append((arm_verts, arm_faces))
        # Light head
        head_verts, head_faces = _create_cylinder_vertices_faces(
            0.3, 0.1, 16, [0, 0, -1.1]
        )
        parts.append((head_verts, head_faces))
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Keyboard(Furniture):
    """
    Represents a computer keyboard.
    The object's position is the center of its bottom face.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.4, 0.15, 0.02]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SittingPerson(Furniture):
    """
    Represents a person sitting down, for example on a chair.
    """
    def __init__(self, name, position, rotation_z=0, material_name="human"):
        """
        :param position: [x, y, z] coordinates of the hip center, assuming they are sitting.
        """
        parts = []
        # Torso
        torso_dims = [0.4, 0.2, 0.6]
        parts.append(_create_box_vertices_faces(torso_dims, [0, 0, 0]))
        # Head
        head_dims = [0.2, 0.2, 0.25]
        parts.append(_create_box_vertices_faces(head_dims, [0, 0, 0.6]))
        # Thighs
        thigh_dims = [0.18, 0.5, 0.18]  # w, d, h
        parts.append(_create_box_vertices_faces(thigh_dims, [-0.1, 0.25, -0.09]))
        parts.append(_create_box_vertices_faces(thigh_dims, [0.1, 0.25, -0.09]))
        # Shins
        shin_dims = [0.15, 0.15, 0.4]
        parts.append(_create_box_vertices_faces(shin_dims, [-0.1, 0.5, -0.4]))
        parts.append(_create_box_vertices_faces(shin_dims, [0.1, 0.5, -0.4]))
        # Arms
        arm_dims = [0.1, 0.4, 0.1]
        parts.append(_create_box_vertices_faces(arm_dims, [-0.25, 0.2, 0.2]))
        parts.append(_create_box_vertices_faces(arm_dims, [0.25, 0.2, 0.2]))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class LayingPerson(Furniture):
    """
    Represents a person laying down, for example on a bed.
    The person is oriented along the y-axis by default.
    """
    def __init__(self, name, position, rotation_z=0, height=1.7, width=0.5, depth=0.3, material_name="human"):
        """
        :param position: [x, y, z] coordinates of the center of the person's body.
        """
        parts = []
        # Same proportions as Person
        head_h = height * 0.15
        torso_h = height * 0.45
        leg_h = height * 0.40
        head_w, torso_w, leg_w = width * 0.5, width, width * 0.25
        head_d, torso_d, arm_d, leg_d = depth * 0.8, depth, depth * 0.8, depth * 0.8
        arm_h, arm_w = torso_h * 0.9, width * 0.2

        # Create parts as if standing
        torso_pos = [0, 0, leg_h]
        parts.append(_create_box_vertices_faces([torso_w, torso_d, torso_h], torso_pos))
        head_pos = [0, 0, leg_h + torso_h]
        parts.append(_create_box_vertices_faces([head_w, head_d, head_h], head_pos))
        left_leg_pos, right_leg_pos = [-torso_w / 4, 0, 0], [torso_w / 4, 0, 0]
        parts.append(_create_box_vertices_faces([leg_w, leg_d, leg_h], left_leg_pos))
        parts.append(_create_box_vertices_faces([leg_w, leg_d, leg_h], right_leg_pos))
        left_arm_pos = [-torso_w / 2 - arm_w / 2, 0, leg_h]
        right_arm_pos = [torso_w / 2 + arm_w / 2, 0, leg_h]
        parts.append(_create_box_vertices_faces([arm_w, arm_d, arm_h], left_arm_pos))
        parts.append(_create_box_vertices_faces([arm_w, arm_d, arm_h], right_arm_pos))

        vertices, faces = _create_composite_object(parts)

        # Rotate to be lying flat along y-axis
        center = np.mean(vertices, axis=0)
        vertices -= center
        angle = np.deg2rad(90)
        # Rotate around X-axis
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        vertices = np.dot(vertices, rotation_matrix.T)

        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class PrivacyCurtain(Furniture):
    """
    Represents a privacy curtain, like in a hospital.
    Position is the center of the bottom face.
    """
    def __init__(self, name, position, rotation_z=0, width=2.0, height=2.0, thickness=0.02, material_name="fabric"):
        dims = [width, thickness, height]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class NoticeBoard(Furniture):
    """
    Represents a notice board on a wall.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, width=1.2, height=0.9, thickness=0.05, material_name="wood"):
        dims = [width, thickness, height]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -height / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class WallMountedWashHandBasin(Furniture):
    """
    Represents a wall-mounted wash-hand basin.
    Position is center of the face attached to the wall.
    """
    def __init__(self, name, position, rotation_z=0, material_name="ceramic"):
        parts = []
        dims = [0.5, 0.4, 0.2]
        # Use center_bottom_pos to position the back of the basin at y=0, for easier wall mounting placement
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, -dims[1] / 2, -dims[2] / 2])
        parts.append((verts, faces))

        # Faucet
        faucet_base_dims = [0.05, 0.05, 0.1]
        faucet_base_verts, faucet_base_faces = _create_box_vertices_faces(
            faucet_base_dims, center_bottom_pos=[0, -dims[1]/2-faucet_base_dims[1]/2, dims[2]/2]
        )
        parts.append((faucet_base_verts, faucet_base_faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SackHolder(Furniture):
    """
    Represents a sack holder (trash bag holder).
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        parts = []
        height = 0.8
        ring_verts, ring_faces = _create_ring_vertices_faces(
            outer_radius=0.2, inner_radius=0.18, height=0.05, resolution=16, center_bottom_pos=[0, 0, height]
        )
        parts.append((ring_verts, ring_faces))
        pole_verts, pole_faces = _create_cylinder_vertices_faces(0.02, height, 8, [0, 0, 0])
        parts.append((pole_verts, pole_faces))
        base_verts, base_faces = _create_cylinder_vertices_faces(0.2, 0.02, 16, [0, 0, 0])
        parts.append((base_verts, base_faces))
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class PaperTowelDispenser(Furniture):
    """
    Represents a wall-mounted paper towel dispenser.
    Position is center of the face attached to the wall.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.3, 0.15, 0.4]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, -dims[1] / 2, -dims[2] / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class GloveDispenser(Furniture):
    """
    Represents a wall-mounted glove dispenser.
    Position is center of the face attached to the wall.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.25, 0.1, 0.15]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, -dims[1] / 2, -dims[2] / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SuppliesTrolley(Furniture):
    """
    Represents a supplies trolley/cart.
    """
    def __init__(self, name, position, rotation_z=0, material_name="metal"):
        parts = []
        width, depth, height = 0.8, 0.5, 0.9
        shelf_thickness = 0.02
        leg_dims = [0.03, 0.03, height]

        # Legs
        leg_positions = [
            [-width/2, -depth/2, 0], [width/2, -depth/2, 0],
            [-width/2, depth/2, 0], [width/2, depth/2, 0]
        ]
        for pos in leg_positions:
            parts.append(_create_box_vertices_faces(leg_dims, pos))

        # Shelves
        shelf_dims = [width, depth, shelf_thickness]
        parts.append(_create_box_vertices_faces(shelf_dims, [0, 0, 0.1]))
        parts.append(_create_box_vertices_faces(shelf_dims, [0, 0, height / 2]))
        parts.append(_create_box_vertices_faces(shelf_dims, [0, 0, height - shelf_thickness]))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class CeilingMountedExaminationLight(Furniture):
    """
    Represents a ceiling mounted examination light fixture.
    Position is where it attaches to the ceiling.
    """
    def __init__(self, name, position, rotation_z=0, material_name="glass"):
        parts = []
        # Arm
        arm_verts, arm_faces = _create_cylinder_vertices_faces(
            0.05, 1.0, 8, [0, 0, -1.0]
        )
        parts.append((arm_verts, arm_faces))
        # Light head
        head_verts, head_faces = _create_cylinder_vertices_faces(
            0.3, 0.1, 16, [0, 0, -1.1]
        )
        parts.append((head_verts, head_faces))
        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class ComputerStation(Furniture):
    """
    Represents a computer station with a desk, monitor, and keyboard.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        parts = []
        # Desk
        desk = Desk("desk_part", [0, 0, 0], 0, material_name)
        parts.append((desk.vertices, desk.faces))
        # Monitor
        monitor = LCDMonitor("monitor_part", [0, 0, 0.75], 0)  # 0.75 is desk height
        parts.append((monitor.vertices, monitor.faces))
        # Keyboard
        keyboard = Keyboard("keyboard_part", [0, 0.2, 0.75], 0)
        parts.append((keyboard.vertices, keyboard.faces))

        vertices, faces = _create_composite_object(parts)
        super().__init__(name, position, vertices.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class SoapDispenser(Furniture):
    """
    Represents a wall-mounted soap dispenser.
    Position is center of the face attached to the wall.
    """
    def __init__(self, name, position, rotation_z=0, material_name="plastic"):
        dims = [0.1, 0.1, 0.2]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, -dims[1] / 2, -dims[2] / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)
