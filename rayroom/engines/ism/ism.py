"""
This module implements the Image Source Method (ISM) for acoustic simulation.

The Image Source Method is a deterministic algorithm used to find the paths of
specular reflections in a room. It is particularly effective for modeling the
early part of a Room Impulse Response (RIR) in rectangular or simple polygonal
geometries.

The core idea is to mirror the sound source across the room's surfaces
(walls) to create "image sources." A valid reflection path exists between the
source and a receiver if a straight line from the receiver to an image source
intersects the chain of reflecting walls that created that image.

This implementation can handle arbitrary polygonal rooms and includes checks for
path validity and occlusion.
"""
import numpy as np

from ...core.constants import C_SOUND
from ...core.physics import air_absorption_coefficient
from ...room.objects import AmbisonicReceiver, Receiver
from ...core.geometry import ray_plane_intersection, is_point_in_polygon, normalize


class ImageSource:
    """Represents an image source in the ISM algorithm.

    An image source is a virtual sound source created by reflecting a real
    source across a surface. This class stores the position of the image
    source, its reflection order, and its lineage (the parent image and
    the wall that generated it).

    :param position: The 3D coordinates of the image source.
    :type position: np.ndarray or list[float]
    :param order: The reflection order of the image source (0 for the real source).
    :type order: int
    :param parent: The parent `ImageSource` object from which this one was generated.
                   `None` for the original source.
    :type parent: ImageSource, optional
    :param generating_wall: The `Wall` object that was used to reflect the parent
                            and create this image source.
    :type generating_wall: rayroom.room.objects.Wall, optional
    """
    def __init__(self, position, order, parent=None, generating_wall=None):
        self.position = np.array(position)
        self.order = order
        self.parent = parent
        self.generating_wall = generating_wall


class ImageSourceEngine:
    """An engine for calculating early specular reflections using the ISM.

    This class manages the process of generating image sources, validating
    reflection paths, and calculating the contribution of each valid path
    to the Room Impulse Response.

    Responsibilities:
      * Recursively generate a tree of image sources up to a specified order.
      * For each receiver, trace paths back from each image source.
      * Validate paths by checking for intersections within wall boundaries
        and for occlusion by other objects.
      * Calculate the arrival time, energy, and direction of each valid reflection.
      * Record the reflection data in the appropriate receiver histograms.

    Example:

        .. code-block:: python

            import rayroom as rt

            # Create a room with a source and receiver
            room = rt.room.ShoeBox([8, 6, 3])
            source = room.add_source([4, 3, 1.5])
            receiver = room.add_receiver([2, 2, 1.5])
            
            # Initialize the ISM engine
            ism_engine = rt.engines.ism.ImageSourceEngine(room)
            
            # Run the simulation for the source
            ism_engine.run(source, max_order=3)
            
            # The receiver's `amplitude_histogram` will now be populated
            # with the calculated early reflections.

    :param room: The `Room` object to be simulated.
    :type room: rayroom.room.Room
    :param temperature: The ambient temperature in Celsius. Defaults to 20.0.
    :type temperature: float, optional
    :param humidity: The relative humidity in percent. Defaults to 50.0.
    :type humidity: float, optional
    """
    def __init__(self, room, temperature=20.0, humidity=50.0):
        self.room = room
        self.temperature = temperature
        self.humidity = humidity
        # Match RayTracer's reference frequency
        self.air_absorption_db_m = air_absorption_coefficient(
            1000.0, temperature, humidity
        )

    def run(self, source, max_order=2, verbose=True):
        """Computes early reflections for a given source.

        This is the main entry point for the engine. It generates the image
        sources and then processes them for each receiver in the room,
        populating the receivers' histograms with the results.

        :param source: The `Source` object for which to calculate reflections.
        :type source: rayroom.room.objects.Source
        :param max_order: The maximum reflection order to compute. Defaults to 2.
        :type max_order: int, optional
        :param verbose: If `True`, print progress information. Defaults to `True`.
        :type verbose: bool, optional
        """
        if verbose:
            print(f"ISM: Processing Source {source.name}...")

            # 1. Generate Image Sources
        images = self._generate_image_sources(source, max_order)

        if verbose:
            print(
                f"  Generated {len(images)} image sources "
                f"(Order <= {max_order})"
            )

        # 2. Check visibility and record for each receiver
        for receiver in self.room.receivers:
            self._process_receiver(source, receiver, images)

    def _generate_image_sources(self, source, max_order):
        """Recursively generates the tree of image sources.

        :param source: The original `Source` object.
        :type source: rayroom.room.objects.Source
        :param max_order: The maximum reflection order.
        :type max_order: int
        :return: A list of all generated `ImageSource` objects.
        :rtype: list[ImageSource]
        """
        images = []

        # Add original source as order 0
        original = ImageSource(source.position, 0, None, None)
        images.append(original)

        self._recursive_images(original, images, max_order)
        return images

    def _recursive_images(self, current_image, all_images, max_depth):
        """The recursive helper function for generating image sources.

        :param current_image: The `ImageSource` to reflect.
        :type current_image: ImageSource
        :param all_images: The list to which new images are added.
        :type all_images: list[ImageSource]
        :param max_depth: The maximum reflection order.
        :type max_depth: int
        """
        if current_image.order >= max_depth:
            return

        for wall in self.room.walls:
            # Don't reflect back across the same wall immediately
            if current_image.generating_wall == wall:
                continue

            # Check if source is in front of the wall
            # Vector from wall point to source
            vec = current_image.position - wall.vertices[0]
            dist = np.dot(vec, wall.normal)

            # Standard ISM reflects everything and filters later.
            if abs(dist) < 1e-6:
                continue  # On the plane

            # Reflect: P' = P - 2 * dist * N
            reflected_pos = current_image.position - 2 * dist * wall.normal

            new_image = ImageSource(
                reflected_pos, current_image.order + 1, current_image, wall
            )
            all_images.append(new_image)

            self._recursive_images(new_image, all_images, max_depth)

    def _process_receiver(self, real_source, receiver, images):
        """Processes all image sources for a single receiver.

        :param real_source: The original `Source` object.
        :type real_source: rayroom.room.objects.Source
        :param receiver: The `Receiver` to process for.
        :type receiver: rayroom.room.objects.Receiver
        :param images: The list of all `ImageSource` objects.
        :type images: list[ImageSource]
        """
        # For each image, check visibility path to receiver
        for img in images:
            result = self._construct_path(img, receiver)

            if result is None:
                continue

            path_points, walls_hit = result

            # Verify validity (intersections within polygons) and Occlusion
            if self._validate_path(path_points, walls_hit):
                # Calculate energy and time
                self._record_reflection(
                    real_source, receiver, img, path_points, walls_hit
                )

    def _construct_path(self, image, receiver):
        """Backtracks from a receiver to an image source to find reflection points.

        This method constructs the geometric path of a potential reflection.
        The path is traced backwards from the receiver towards the image source,
        calculating the intersection point with each parent wall in the
        image's lineage.

        :param image: The `ImageSource` to trace from.
        :type image: ImageSource
        :param receiver: The target `Receiver`.
        :type receiver: rayroom.room.objects.Receiver
        :return: A tuple containing a list of path points and a list of walls hit,
                 or `None` if a valid path cannot be constructed.
        :rtype: tuple[list[np.ndarray], list] or None
        """
        path_points = [receiver.position]
        walls_hit = []

        current_target = receiver.position
        current_img = image

        # Backtrack
        while current_img.parent is not None:
            wall = current_img.generating_wall
            parent = current_img.parent

            vec = current_img.position - current_target
            dist_to_img = np.linalg.norm(vec)
            if dist_to_img < 1e-9:
                return None

            ray_dir = vec / dist_to_img

            t = ray_plane_intersection(
                current_target, ray_dir, wall.vertices[0], wall.normal
            )

            if t is None or t < 1e-5 or t > dist_to_img + 1e-5:
                return None

            intersection_point = current_target + t * ray_dir
            path_points.append(intersection_point)
            walls_hit.append(wall)

            current_target = intersection_point
            current_img = parent

        path_points.append(current_img.position)  # Real source position

        return path_points, walls_hit

    def _validate_path(self, points, walls):
        """Validates a constructed reflection path.

        This method performs two crucial checks:
        1.  It ensures that each reflection point lies within the boundaries
            of its corresponding wall polygon.
        2.  It checks if any segment of the path is occluded by other objects
            in the room (walls or furniture).

        :param points: The list of points defining the path segments.
        :type points: list[np.ndarray]
        :param walls: The list of walls corresponding to the reflection points.
        :type walls: list
        :return: `True` if the path is valid, `False` otherwise.
        :rtype: bool
        """
        # 1. Check Polygons
        # Note: points[1] corresponds to walls[0] (Hit1)
        for i, wall in enumerate(walls):
            hit_point = points[i+1]
            if not is_point_in_polygon(hit_point, wall.vertices, wall.normal):
                return False

        # 2. Check Occlusion
        # Segments: (Points[i] -> Points[i+1])
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            ignore_walls = []
            if i > 0:
                ignore_walls.append(walls[i-1])
            if i < len(walls):
                ignore_walls.append(walls[i])

            if self._is_occluded(p1, p2, ignore_walls):
                return False

        return True

    def _is_occluded(self, p1, p2, ignore_walls):
        """Checks if the line segment between two points is occluded.

        :param p1: The start point of the segment.
        :type p1: np.ndarray
        :param p2: The end point of the segment.
        :type p2: np.ndarray
        :param ignore_walls: A list of walls to ignore during the check (usually
                             the reflecting walls that define the segment).
        :type ignore_walls: list
        :return: `True` if the segment is occluded, `False` otherwise.
        :rtype: bool
        """
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < 1e-5:
            return False

        ray_dir = vec / dist

        # Check Walls
        for wall in self.room.walls:
            if wall in ignore_walls:
                continue

            t = ray_plane_intersection(
                p1, ray_dir, wall.vertices[0], wall.normal
            )
            if t is not None and 1e-4 < t < dist - 1e-4:
                # Hit a wall plane between p1 and p2
                # Check if inside polygon
                hit_p = p1 + t * ray_dir
                if is_point_in_polygon(hit_p, wall.vertices, wall.normal):
                    return True

        # Check Furniture
        for furn in self.room.furniture:
            # Check each face of the furniture
            for f_idx, normal in enumerate(furn.face_normals):
                plane_pt = furn.face_planes[f_idx]
                t = ray_plane_intersection(p1, ray_dir, plane_pt, normal)

                if t is not None and 1e-4 < t < dist - 1e-4:
                    face_verts = furn.vertices[furn.faces[f_idx]]
                    hit_p = p1 + t * ray_dir
                    if is_point_in_polygon(hit_p, face_verts, normal):
                        return True

        return False

    def _record_reflection(
            self, source, receiver, image, path_points, walls_hit
    ):
        """Calculates and records the properties of a valid reflection.

        This method computes the total path length, arrival time, and energy
        of a validated reflection path. It accounts for geometric spreading,
        source directivity, air absorption, and wall absorption. The final
        result is recorded in the receiver's histogram.

        :param source: The original `Source` object.
        :type source: rayroom.room.objects.Source
        :param receiver: The target `Receiver`.
        :type receiver: rayroom.room.objects.Receiver
        :param image: The `ImageSource` corresponding to the reflection.
        :type image: ImageSource
        :param path_points: The points defining the reflection path.
        :type path_points: list[np.ndarray]
        :param walls_hit: The walls involved in the reflection.
        :type walls_hit: list
        """
        # Calculate total distance
        total_dist = 0.0
        for i in range(len(path_points) - 1):
            total_dist += np.linalg.norm(path_points[i+1] - path_points[i])

        time = total_dist / C_SOUND

        # Geometric Spreading: Power * Area / (4 * pi * r^2)
        receiver_area = np.pi * receiver.radius**2
        geom_factor = receiver_area / (4 * np.pi * total_dist**2 + 1e-12)

        # Directivity
        # Direction from source (last point) to first hit (second to last)
        dir_vec = path_points[-2] - path_points[-1]
        dir_vec = normalize(dir_vec)

        gain = 1.0
        if hasattr(source, 'directivity') and \
           source.directivity != "omnidirectional":
            if hasattr(source, 'orientation'):
                cos_theta = np.dot(dir_vec, source.orientation)
                if source.directivity == "cardioid":
                    gain = 0.5 * (1.0 + cos_theta)
                elif source.directivity == "subcardioid":
                    gain = 0.7 + 0.3 * cos_theta
                elif source.directivity == "hypercardioid":
                    gain = np.abs(0.25 + 0.75 * cos_theta)
                elif source.directivity == "bidirectional":
                    gain = np.abs(cos_theta)

        energy = source.power * gain * geom_factor

        # Air Absorption
        # E = E0 * 10^(-alpha * dist / 10)
        energy *= 10**(-self.air_absorption_db_m * total_dist / 10.0)

        # Wall Absorption (Reflection coefficients)
        for wall in walls_hit:
            mat = wall.material
            abs_coeff = np.mean(mat.absorption) \
                if np.ndim(mat.absorption) > 0 else mat.absorption
            energy *= (1.0 - abs_coeff)

        # Record
        if isinstance(receiver, AmbisonicReceiver):
            # Direction from last bounce (or source) to receiver
            arrival_dir = normalize(path_points[0] - path_points[1])
            receiver.record(time, energy, arrival_dir)
        elif isinstance(receiver, Receiver):
            receiver.record(time, energy)
