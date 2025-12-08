import numpy as np


def normalize(v):
    """Normalizes a vector to unit length.

    This function scales the input vector so that its magnitude is 1, which
    is a common operation in vector mathematics and physics simulations.

    Responsibilities:
      * Calculate the L2 norm (magnitude) of the vector.
      * Divide the vector by its norm.
      * Handle the case of a zero-length vector to avoid division by zero.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            vec = np.array([3, 4, 0])
            normalized_vec = rt.core.geometry.normalize(vec)
            # normalized_vec is now array([0.6, 0.8, 0.])

    :param v: The input vector.
    :type v: np.ndarray
    :return: The normalized vector.
    :rtype: np.ndarray
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    """Calculates the intersection of a ray and a plane.

    Determines the point where a ray, defined by an origin and direction,
    intersects with a plane. This is a fundamental operation in ray tracing
    and other geometric calculations.

    Responsibilities:
      * Compute the distance from the ray's origin to the intersection point.
      * Handle cases where the ray is parallel to the plane.
      * Ensure the intersection is in the forward direction of the ray.

    :param ray_origin: The starting point of the ray [x, y, z].
    :type ray_origin: np.ndarray
    :param ray_dir: The direction vector of the ray [x, y, z].
    :type ray_dir: np.ndarray
    :param plane_point: A point that lies on the plane [x, y, z].
    :type plane_point: np.ndarray
    :param plane_normal: The normal vector of the plane [x, y, z].
    :type plane_normal: np.ndarray
    :return: The distance `t` along the ray to the intersection, such that
             `intersection_point = ray_origin + t * ray_dir`. Returns `None`
             if there is no intersection in the forward direction.
    :rtype: float or None
    """
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:
        return None

    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None  # Intersection behind ray

    return t


def is_point_in_polygon(point, vertices, normal):
    """Checks if a point is inside a 3D polygon.

    This function determines if a point, which is assumed to lie on the same
    plane as the polygon, is inside the polygon's boundaries. It does this by
    projecting the point and polygon to 2D and using the crossing number
    algorithm.

    Responsibilities:
      * Project the 3D point and polygon vertices to a 2D plane.
      * Implement the crossing number algorithm to test for inclusion.
      * Correctly handle the projection based on the polygon's normal.

    :param point: The point to check [x, y, z].
    :type point: np.ndarray
    :param vertices: The vertices of the polygon in order.
    :type vertices: np.ndarray
    :param normal: The normal vector of the polygon's plane.
    :type normal: np.ndarray
    :return: `True` if the point is inside the polygon, `False` otherwise.
    :rtype: bool
    """
    # Project 3D to 2D by dropping the dimension with largest normal component
    abs_n = np.abs(normal)
    if abs_n[0] > abs_n[1] and abs_n[0] > abs_n[2]:
        # Drop x, use y, z
        proj_p = point[1:]
        proj_v = vertices[:, 1:]
    elif abs_n[1] > abs_n[0] and abs_n[1] > abs_n[2]:
        # Drop y, use x, z
        proj_p = np.array([point[0], point[2]])
        proj_v = vertices[:, [0, 2]]
    else:
        # Drop z, use x, y
        proj_p = point[:2]
        proj_v = vertices[:, :2]

    # Crossing number algorithm
    inside = False
    n = len(proj_v)
    p1 = proj_v[0]
    for i in range(n + 1):
        p2 = proj_v[i % n]
        if min(p1[1], p2[1]) < proj_p[1] <= max(p1[1], p2[1]):
            if proj_p[0] <= max(p1[0], p2[0]):
                if p1[1] != p2[1]:
                    xinters = (proj_p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if p1[0] == p2[0] or proj_p[0] <= xinters:
                    inside = not inside
        p1 = p2

    return inside


def reflect_vector(incident, normal):
    """Calculates the reflection of a vector across a surface normal.

    This function computes the direction of a vector after a specular
    reflection. This is essential for simulating how rays of light or sound
    bounce off surfaces.

    Responsibilities:
      * Implement the vector reflection formula.
      * Assume both incident and normal vectors are normalized.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            incident_dir = rt.core.geometry.normalize(np.array([1, -1, 0]))
            surface_normal = np.array([0, 1, 0])
            
            reflected_dir = rt.core.geometry.reflect_vector(
                incident_dir, surface_normal
            )
            # reflected_dir is now close to [0.707, 0.707, 0.]

    :param incident: The incident vector.
    :type incident: np.ndarray
    :param normal: The surface normal vector.
    :type normal: np.ndarray
    :return: The reflected vector.
    :rtype: np.ndarray
    """
    return incident - 2 * np.dot(incident, normal) * normal


def random_direction_hemisphere(normal):
    """Generates a random direction in a hemisphere with cosine weighting.

    This is used for simulating diffuse reflections, where an incoming ray
    scatters in a random direction from a surface. The cosine weighting
    ensures that the distribution of reflected rays is physically accurate
    for a Lambertian surface.

    Responsibilities:
      * Create a local coordinate system based on the surface normal.
      * Generate a random direction with a cosine distribution.
      * Transform the direction back to world coordinates.

    :param normal: The normal vector that defines the hemisphere's orientation.
    :type normal: np.ndarray
    :return: A normalized random direction vector within the hemisphere.
    :rtype: np.ndarray
    """
    # Create a random coordinate system
    if abs(normal[0]) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    else:
        u = np.array([1.0, 0.0, 0.0])

    u = normalize(np.cross(u, normal))
    v = np.cross(normal, u)

    # Random samples
    phi = 2 * np.pi * np.random.random()
    r2 = np.random.random()
    r = np.sqrt(r2)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(1 - r2)

    # Transform to world coordinates
    direction = x * u + y * v + z * normal
    return normalize(direction)
