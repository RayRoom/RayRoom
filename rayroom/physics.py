import numpy as np

def air_absorption_coefficient(freq, temperature=20.0, humidity=50.0, pressure=101325.0):
    """
    Calculate air absorption coefficient (alpha) in dB/m.

    Simplified approximation proportional to frequency squared.
    Good enough for visualization and basic energy decay.
    
    :param freq: Frequency in Hz.
    :type freq: float
    :param temperature: Temperature in Celsius. Defaults to 20.0. (unused in simplified model)
    :type temperature: float
    :param humidity: Relative humidity in percent. Defaults to 50.0. (unused in simplified model)
    :type humidity: float
    :param pressure: Atmospheric pressure in Pascals. Defaults to 101325.0. (unused)
    :type pressure: float
    :return: Absorption coefficient in dB/m.
    :rtype: float
    """
    # Approx 0.005 dB/m at 1kHz
    # Scales with f^2
    alpha_approx = 5e-9 * (freq**2)
    return alpha_approx
