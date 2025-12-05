import numpy as np

class Material:
    def __init__(self, name, absorption=0.1, transmission=0.0, scattering=0.0):
        """
        Initialize a Material.
        
        Args:
            name (str): Name of the material.
            absorption (float or np.array): Absorption coefficient (alpha). 
                                            0 = perfect reflection, 1 = perfect absorption.
                                            Can be a single float or array for frequency bands.
            transmission (float or np.array): Transmission coefficient (tau).
                                              0 = opaque, 1 = fully transparent.
            scattering (float or np.array): Scattering coefficient (s).
                                            0 = specular, 1 = diffuse.
        """
        self.name = name
        self.absorption = np.array(absorption) if isinstance(absorption, (list, tuple)) else np.array([absorption])
        self.transmission = np.array(transmission) if isinstance(transmission, (list, tuple)) else np.array([transmission])
        self.scattering = np.array(scattering) if isinstance(scattering, (list, tuple)) else np.array([scattering])

    def __repr__(self):
        return f"Material({self.name}, abs={self.absorption}, trans={self.transmission})"

# Common Materials Library
def get_material(name):
    # Simplified values, ideally these would be frequency dependent arrays
    materials = {
        "concrete": Material("Concrete", absorption=0.05, transmission=0.0),
        "brick": Material("Brick", absorption=0.03, transmission=0.0),
        "glass": Material("Glass", absorption=0.03, transmission=0.1), # Partial transmission
        "heavy_curtain": Material("Heavy Curtain", absorption=0.6, transmission=0.2),
        "wood": Material("Wood", absorption=0.15, transmission=0.01),
        "plaster": Material("Plaster", absorption=0.1, transmission=0.0),
        "air": Material("Air", absorption=0.0, transmission=1.0), # Fully transparent
        "transparent_wall": Material("TransparentWall", absorption=0.1, transmission=0.8),
    }
    return materials.get(name, Material("Default", 0.1, 0.0))

