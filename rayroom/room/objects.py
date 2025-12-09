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
        offset_faces = (np.array(faces) + num_verts).tolist()
        all_faces.extend(offset_faces)
        num_verts += len(verts)

    if not all_verts:
        return np.array([]), []

    return np.vstack(all_verts), all_faces


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
    def __init__(self, name, position, rotation_z=0, material_name="default"):
        dims = [1.2, 0.05, 0.7]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Desk(Furniture):
    """
    Represents a simple desk as a single box.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        dims = [1.2, 0.6, 0.75]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


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
    Represents a square carpet as a flat box.
    """
    def __init__(self, name, position, rotation_z=0, material_name="carpet"):
        dims = [2.0, 2.0, 0.02]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class Subwoofer(Furniture):
    """
    Represents a subwoofer as a cube.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        dims = [0.4, 0.4, 0.4]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


class FloorstandingSpeaker(Furniture):
    """
    Represents a floorstanding speaker as a tall box.
    """
    def __init__(self, name, position, rotation_z=0, material_name="wood"):
        dims = [0.3, 0.4, 1.0]  # width, depth, height
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, 0])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)


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
    Represents a wall clock as a thin square box.
    The object's position is its geometric center.
    """
    def __init__(self, name, position, rotation_z=0, size=0.3, thickness=0.05, material_name="plastic"):
        dims = [size, thickness, size]
        verts, faces = _create_box_vertices_faces(dims, center_bottom_pos=[0, 0, -size / 2])
        super().__init__(name, position, verts.tolist(), faces, get_material(material_name), rotation_z=rotation_z)
